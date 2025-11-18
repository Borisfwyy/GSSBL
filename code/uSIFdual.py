import json
import numpy as np
from collections import Counter
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
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit  # sigmoid
from sklearn.svm import LinearSVC

patch_sklearn()

# ===== MLP 定义 =====
class MLP(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def run_usifdual(word_vec_path: str, data_paths: list, output_csv_path: str, repeat: int = 5, cuda_device:int = 0, use_subword: bool = False, lr = 1e-4):
    EMBED_DIM = 512
    SMOOTHING_A = 1e-3
    threshold = 0.5
    PATIENCE = 100
    batchsize = 64
    LR = lr

    train_path, test_path = data_paths
    print(f"使用子字: {use_subword}, cuda_device: {cuda_device}, uSIF..., lr={LR}, save：{output_csv_path}, json: {word_vec_path}, repeat={repeat}")    
    #set_cuda_device(cuda_device)


    # ===== 1. 读取词向量 =====
    with open(word_vec_path, "r", encoding="utf-8") as f:
        word_vec = json.load(f)
        word_vec = {k: np.array(v) for k, v in word_vec.items()}

    # ===== 2. 统计正例词频 =====
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
                    parts = w.split("|")
                    sub = parts[1] if len(parts) > 1 else None
                    main = parts[0]
                    token = sub if (sub and sub.strip()) else main
                else:
                    token = w.split("|")[0]
                if token:
                    words.append(token)
        counter = Counter(words)
        total = sum(counter.values())
        return {w: c/total for w, c in counter.items()}

    word_prob = compute_word_freq(positive_sentences, use_subword=use_subword)

    # ===== 3. uSIF embedding（经典方式） =====
    def compute_usif_embedding(sentence, word_vec, word_prob, use_subword=use_subword, a=SMOOTHING_A):
        words = sentence.strip().split()
        vecs = []
        for w in words:
            if use_subword:
                parts = w.split("|")
                sub = parts[1] if len(parts) > 1 else None
                main = parts[0]
                token = sub if (sub and sub in word_vec) else main
            else:
                token = w.split("|")[0]
            if token in word_vec:
                p = word_prob.get(token, 1e-5)
                weight = a / (a + p)
                vecs.append(weight * word_vec[token])
        if not vecs:
            return np.zeros(EMBED_DIM)
        return np.mean(np.vstack(vecs), axis=0)

    # ===== 4. 特征准备 =====
    def prepare_features_labels(file_path, word_vec, word_prob, use_subword=use_subword):
        """
        先对句子 A / B 分别计算 uSIF embedding，再 concat 作为句子对特征
        """
        X, y = [], []
        for line in tqdm(open(file_path, "r", encoding="utf-8"), desc=f"生成特征: {file_path}"):
            a, b, label = line.strip().split("\t")
            va = compute_usif_embedding(a, word_vec, word_prob, use_subword=use_subword)
            vb = compute_usif_embedding(b, word_vec, word_prob, use_subword=use_subword)
            feat = np.concatenate([va, vb])  # concat 两个句子 embedding
            X.append(feat)
            y.append(int(label))
        return np.array(X), np.array(y)



    X_train, y_train = prepare_features_labels(train_path, word_vec, word_prob, use_subword=use_subword)
    X_test, y_test = prepare_features_labels(test_path, word_vec, word_prob, use_subword=use_subword)
    input_dim = X_train.shape[1]  # 自动对应 word_vec 的维度


    # ===== 5. 模型字典 =====
    model_dict = {
        #"LR": "lr",
        "SVM": "svm",
        #"XGBoost": "xgb",
        #"LightGBM": "lgbm",
        #"MLP": "mlp"
    }

    all_results = {name: {'auroc': [], 'aupr': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []} for name in model_dict.keys()}

    # ===== 6. 标准化 =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== 7. 非 MLP 模型 =====
    for name, mtype in model_dict.items():
        if mtype == "mlp":
            continue  # MLP 后面单独处理

        print(f"\n=== 模型 {name} ===")
        run_id = 0  # 只跑一次
        print(f"\n--- 第 {run_id + 1} 次实验 ---")
        
        print(f"\n训练模型: {name}")

        # ===== 初始化模型 =====
        if mtype == "lr":
            max_iter_set = 100
            print(f"LR的max_iter = {max_iter_set}")
            model = LogisticRegression(max_iter=max_iter_set, class_weight='balanced')
        elif mtype == "svm":
            model = LinearSVC(max_iter=1000, class_weight='balanced', C=1, loss="hinge")
        elif mtype == "xgb":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        elif mtype == "lgbm":
            model = LGBMClassifier()
        else:
            continue  # 未知模型跳过

        # ===== 训练 =====
        model.fit(X_train_scaled, y_train)

        # ===== 预测概率/决策值 =====
        if mtype in ["lr", "svm"]:
            y_prob = model.decision_function(X_test_scaled)
        else:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        y_pred = (y_prob >= threshold).astype(int)

        # ===== 指标计算 =====
        auroc = roc_auc_score(y_test, y_prob)
        aupr = average_precision_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        all_results[name]['auroc'].append(auroc)
        all_results[name]['aupr'].append(aupr)
        all_results[name]['acc'].append(acc)
        all_results[name]['precision'].append(precision)
        all_results[name]['recall'].append(recall)
        all_results[name]['f1'].append(f1)

        print(f">>> {name} 指标:")
        print(f"AUROC={auroc:.4f}, AUPR={aupr:.4f}, Accuracy={acc:.4f}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


    # ===== 8. MLP 多次实验 =====
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    if "MLP" in model_dict:
        for run_id in range(repeat):
            print(f"\n=== MLP 第 {run_id + 1} 次实验 ===")
            mlp = MLP(input_dim=input_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
            train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

            best_loss = float('inf')
            best_state = None
            wait = 0

            for epoch in range(1, 101):
                total_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = mlp(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(yb)
                avg_loss = total_loss / len(train_loader.dataset)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = mlp.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break

            mlp.load_state_dict(best_state)
            mlp.eval()
            with torch.no_grad():
                y_prob = torch.sigmoid(mlp(torch.tensor(X_test_scaled, dtype=torch.float32).to(device))).cpu().numpy()
                y_pred = (y_prob >= threshold).astype(int)

            all_results["MLP"]['auroc'].append(roc_auc_score(y_test, y_prob))
            all_results["MLP"]['aupr'].append(average_precision_score(y_test, y_prob))
            all_results["MLP"]['acc'].append(accuracy_score(y_test, y_pred))
            all_results["MLP"]['precision'].append(precision_score(y_test, y_pred))
            all_results["MLP"]['recall'].append(recall_score(y_test, y_pred))
            all_results["MLP"]['f1'].append(f1_score(y_test, y_pred))
    else:
        print("MLP 未包含在模型列表中，跳过 MLP 训练。")

    # ===== 9. 保存 CSV =====
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"])
        for name in model_dict.keys():
            results = all_results[name]
            n_runs = len(results['auroc'])
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
                f"{np.mean(results['auroc']):.8f}",
                f"{np.mean(results['aupr']):.8f}",
                f"{np.mean(results['acc']):.8f}",
                f"{np.mean(results['precision']):.8f}",
                f"{np.mean(results['recall']):.8f}",
                f"{np.mean(results['f1']):.8f}"
            ])
    print(f"\n✅ 结果均值已保存到 {output_csv_path}")
