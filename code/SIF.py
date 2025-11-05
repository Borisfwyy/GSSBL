import json
import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn
from tqdm import tqdm
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cupy as cp
import copy
from sklearn.svm import LinearSVC

patch_sklearn()

# ===== MLP 定义 =====
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

import numba.cuda as nbcuda  # cuML 用的是 Numba，可以直接看当前设备

def set_cuda_device(cuda_device):
    """
    设置 cuML 使用的 GPU 设备。
    """
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        print(f"[INFO] CUDA_VISIBLE_DEVICES 已设置为 {cuda_device}")
    else:
        print("[INFO] 未指定 cuda_device，使用默认 GPU")

    # 检查 cuML / Numba 实际使用的设备
    try:
        current = nbcuda.get_current_device()
        print(f"[INFO] 当前 GPU: {current.id} - {current.name}")
        # 当前逻辑 GPU
        logical_id = torch.cuda.current_device()
        # GPU 名称
        gpu_name = torch.cuda.get_device_name(logical_id)
        # 物理编号
        physical_id = torch.cuda.device_count()
        print(f"[INFO] 当前逻辑 GPU: {logical_id}, 名称: {gpu_name}")
        print(f"[INFO] 可用GPU物理编号: {physical_id}")
    except Exception as e:
        print(f"[WARN] 无法获取当前 GPU 信息: {e}")

def run_sif(
    word_vec_path: str,
    data_paths: list,   # [train_path, test_path]
    output_csv_path: str,
    repeat: int = 5,
    cuda_device: int = 0,
    use_subword: bool = False   # 新增参数
):
    SMOOTHING_A = 1e-3
    EMBED_DIM = 512
    EPOCHS = 100
    PATIENCE = 10
    BATCH_SIZE = 64
    LR = 1e-4
    threshold = 0.5  # 评价指标计算阈值
    print(f"使用子字: {use_subword}, cuda_device: {cuda_device}, SIF...")    
    train_path, test_path = data_paths
    
    #set_cuda_device(cuda_device)
    from cuml.svm import SVC

    # 1. 加载词向量
    with open(word_vec_path, "r", encoding="utf-8") as f:
        word_vec = json.load(f)
        word_vec = {k: np.array(v) for k, v in word_vec.items()}
    print(f"Loaded word vectors from {word_vec_path}, total {len(word_vec)} words.")

    # 2. 正例统计
    def load_positive_sentences(file_path):
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[2] == "1":
                    sentences.append(parts[0])
                    sentences.append(parts[1])
        return sentences

    pos_sents_train = load_positive_sentences(train_path)
    pos_sents_test = load_positive_sentences(test_path)
    positive_sentences = pos_sents_train + pos_sents_test

    def compute_word_freq(sentences, use_subword=False):
        words = []
        for s in sentences:
            for w in s.strip().split():
                if use_subword:
                    # 优先用子字
                    parts = w.split("|")
                    token = parts[1] if len(parts) > 1 else parts[0]
                else:
                    token = w.split("|")[0]
                if token:
                    words.append(token)
        counter = Counter(words)
        total = sum(counter.values())
        prob = {w: c / total for w, c in counter.items()}
        return prob

    word_prob = compute_word_freq(positive_sentences, use_subword=use_subword)

    # 3. SIF embedding
    def compute_sif_embedding(sentence, word_vec, word_prob, a=SMOOTHING_A):
        words = sentence.strip().split()
        vecs = []
        for w in words:
            token = w.split("|")[1] if use_subword else w.split("|")[0]
            if token in word_vec:
                p = word_prob.get(token, 1e-5)
                weight = a / (a + p)
                vecs.append(weight * word_vec[token])
        if not vecs:
            raise ValueError(f"SIF embedding is all zeros! Sentence: '{sentence}' with use_subword={use_subword}")
        return np.mean(np.vstack(vecs), axis=0)


    # 4. 去第一个主成分
    def remove_pc(X, npc=1):
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=42)
        svd.fit(X)
        pc = svd.components_
        if npc == 1:
            pc = pc[0]
            return X - X.dot(pc)[:, None] * pc[None, :]
        else:
            return X - X.dot(pc.T).dot(pc)

    # 5. 生成特征和标签
    def prepare_features_labels(file_path):
        X, y = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"生成特征: {file_path}"):
                a, b, label = line.strip().split("\t")
                va = compute_sif_embedding(a, word_vec, word_prob)
                vb = compute_sif_embedding(b, word_vec, word_prob)
                feat = np.concatenate([va, vb])
                X.append(feat)
                y.append(int(label))
        X = np.array(X)
        X = remove_pc(X, npc=1)
        return X, np.array(y)

    # ======= 后面代码保持不变 =======
    X_train, y_train = prepare_features_labels(train_path)
    X_test, y_test = prepare_features_labels(test_path)

    input_dim = X_train.shape[1]

    # 6. 模型字典
    model_dict = {
        "LR": "lr",
        "SVM": "svm",
        "XGBoost": "xgb",
        "LightGBM": "lgbm",
        "MLP": "mlp"
    }

    all_results = {name: {'auroc': [], 'aupr': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []} for name in model_dict.keys()}

    # 标准化
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("标准化特征...done")

    # 前四个模型只跑一次
    for name, mtype in model_dict.items():
        if mtype != "mlp":
            print(f"\n训练模型: {name}")
            if mtype == "lr":
                m = LogisticRegression(max_iter=2000, solver="saga", class_weight='balanced')
                m.fit(X_train_scaled, y_train)
                y_prob = m.decision_function(X_test_scaled)
            elif mtype == "svm":
                m = LinearSVC(max_iter=10000, class_weight="balanced")
                m.fit(X_train_scaled, y_train)
                y_prob = m.decision_function(X_test_scaled)
            elif mtype == "xgb":
                m = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                m.fit(X_train_scaled, y_train)
                y_prob = m.predict_proba(X_test_scaled)[:, 1]
            elif mtype == "lgbm":
                m = LGBMClassifier()
                m.fit(X_train_scaled, y_train)
                y_prob = m.predict_proba(X_test_scaled)[:, 1]
            else:
                continue

            y_pred = (y_prob >= threshold).astype(int)
            # 打印指标
            print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}, AUPR: {average_precision_score(y_test, y_prob):.4f}, "
                  f"Acc: {accuracy_score(y_test, y_pred):.4f}, Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}, "
                  f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}, F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")

            all_results[name]['auroc'].append(roc_auc_score(y_test, y_prob))
            all_results[name]['aupr'].append(average_precision_score(y_test, y_prob))
            all_results[name]['acc'].append(accuracy_score(y_test, y_pred))
            all_results[name]['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            all_results[name]['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            all_results[name]['f1'].append(f1_score(y_test, y_pred, zero_division=0))

    # MLP 多次实验
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    for run_id in range(repeat):
        print(f"\n=== MLP 第 {run_id + 1} 次实验 ===")
        model = MLPClassifier(input_dim).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        best_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for i, (xb, yb) in enumerate(train_loader, start=1):
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb).squeeze(1)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                print(f"\rBatch {i}/{len(train_loader)} processed...", end="")

            avg_loss = total_loss / len(train_loader.dataset)
            print(f"\nEpoch {epoch}/{EPOCHS} finished, Loss: {avg_loss:.4f}")

            # Early Stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # 载入最佳模型
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            logits = model(X_test_tensor).squeeze(1).cpu().numpy()
            y_prob = 1 / (1 + np.exp(-logits))
            y_pred = (y_prob >= threshold).astype(int)

        # 打印指标
        print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}, AUPR: {average_precision_score(y_test, y_prob):.4f}, "
              f"Acc: {accuracy_score(y_test, y_pred):.4f}, Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}, "
              f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}, F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")

        all_results["MLP"]['auroc'].append(roc_auc_score(y_test, y_prob))
        all_results["MLP"]['aupr'].append(average_precision_score(y_test, y_prob))
        all_results["MLP"]['acc'].append(accuracy_score(y_test, y_pred))
        all_results["MLP"]['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        all_results["MLP"]['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        all_results["MLP"]['f1'].append(f1_score(y_test, y_pred, zero_division=0))

    # 写 CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"])

        for name in model_dict.keys():
            results = all_results[name]
            n_runs = len(results['auroc'])
            # 如果是 MLP，多次实验
            if name == "MLP":
                for idx in range(n_runs):
                    writer.writerow([
                        name, idx + 1,
                        f"{results['auroc'][idx]:.4f}",
                        f"{results['aupr'][idx]:.4f}",
                        f"{results['acc'][idx]:.4f}",
                        f"{results['precision'][idx]:.4f}",
                        f"{results['recall'][idx]:.4f}",
                        f"{results['f1'][idx]:.4f}"
                    ])
            # 其他模型只跑一次，写一次 
            else:
                writer.writerow([
                    name, 1,
                    f"{results['auroc'][0]:.4f}",
                    f"{results['aupr'][0]:.4f}",
                    f"{results['acc'][0]:.4f}",
                    f"{results['precision'][0]:.4f}",
                    f"{results['recall'][0]:.4f}",
                    f"{results['f1'][0]:.4f}"
                ])
            # 均值
            writer.writerow([
                name, "mean",
                f"{np.mean(results['auroc']):.4f}",
                f"{np.mean(results['aupr']):.4f}",
                f"{np.mean(results['acc']):.4f}",
                f"{np.mean(results['precision']):.4f}",
                f"{np.mean(results['recall']):.4f}",
                f"{np.mean(results['f1']):.4f}"
            ])

    print(f"\nMLP均值结果: AUROC={np.mean(all_results['MLP']['auroc']):.4f}, AUPR={np.mean(all_results['MLP']['aupr']):.4f}, acc={np.mean(all_results['MLP']['acc']):.4f}, precision={np.mean(all_results['MLP']['precision']):.4f}, recall={np.mean(all_results['MLP']['recall']):.4f}, f1={np.mean(all_results['MLP']['f1']):.4f}, recall={np.mean(all_results['MLP']['recall']):.4f}, f1={np.mean(all_results['MLP']['f1']):.4f}, F1={np.mean(all_results['MLP']['f1']):.4f}")
    print(f"\n✅ 结果已保存到 {output_csv_path}")
