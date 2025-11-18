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
import copy
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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

def run_sif(
    word_vec_path: str,
    data_paths: list,   # [train_path, test_path]
    output_csv_path: str,
    repeat: int = 5,
    cuda_device: int = 0,
    use_subword: bool = False,   # 新增参数
    lr = 1e-4
):
    SMOOTHING_A = 1e-3
    EPOCHS = 100
    PATIENCE = 100
    BATCH_SIZE = 64
    LR = lr
    setmax = 100
    threshold = 0.5  # 评价指标计算阈值
    print(f"使用子字: {use_subword}, cuda_device: {cuda_device}, SIF..., lr={LR}, save：{output_csv_path}, json: {word_vec_path}, repeat={repeat}, 最大循环={setmax}")    
    train_path, test_path = data_paths

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
                parts = w.split("|")
                token = parts[1] if use_subword and len(parts) > 1 and parts[1] else parts[0]
                if token:
                    words.append(token)
        counter = Counter(words)
        total = sum(counter.values())
        prob = {w: c / total for w, c in counter.items()}
        return prob

    word_prob = compute_word_freq(positive_sentences, use_subword=use_subword)

    # 3. SIF embedding
    def compute_sif_embedding_pair(sent_a, sent_b, word_vec, word_prob, a=SMOOTHING_A, use_subword=False):
        combined_sentence = sent_a + " " + sent_b
        words = combined_sentence.strip().split()
        vecs = []
        for w in words:
            token = w.split("|")[1] if use_subword else w.split("|")[0]
            if token in word_vec:
                p = word_prob.get(token, 1e-5)
                weight = a / (a + p)
                vecs.append(weight * word_vec[token])
        if not vecs:
            raise ValueError(f"SIF embedding is all zeros! Sentence pair: '{sent_a}' + '{sent_b}'")
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
                feat = compute_sif_embedding_pair(a, b, word_vec, word_prob, a=SMOOTHING_A, use_subword=use_subword)
                X.append(feat)
                y.append(int(label))
        X = np.array(X)
        X = remove_pc(X, npc=1)
        return X, np.array(y)

    X_train, y_train = prepare_features_labels(train_path)
    X_test, y_test = prepare_features_labels(test_path)
    input_dim = X_train.shape[1]

    # 6. 模型字典
    model_dict = {
        #"LR": "lr",
        "SVM": "svm",
        #"XGBoost": "xgb",
        #"LightGBM": "lgbm",
        #"MLP": "mlp"
    }

    all_results = {name: {'auroc': [], 'aupr': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []} 
                   for name in model_dict.keys()}

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===== 非 MLP 模型训练（只一次） =====
    for name, mtype in model_dict.items():
        print(f"\n=== 模型 {name} ===")

        if mtype == "lr":
            model = LogisticRegression(max_iter=setmax, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            y_score = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)

        elif mtype == "svm":
            print("模型调整回linearsvc了，可能会不收敛，多放了点小数点，迭代上限1000+平衡")
            model = LinearSVC(max_iter=1000, class_weight='balanced', C=1, loss="hinge")
            model.fit(X_train_scaled, y_train)
            y_score = model.decision_function(X_test_scaled)   # 用于 AUROC / AUPR
            y_pred = model.predict(X_test_scaled)              # 用于 ACC / P / R / F1

        elif mtype == "xgb":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train_scaled, y_train)
            y_score = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)

        elif mtype == "lgbm":
            model = LGBMClassifier()
            model.fit(X_train_scaled, y_train)
            y_score = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)

        else:
            continue

        # ===== 统一指标计算 =====
        auroc = roc_auc_score(y_test, y_score)
        aupr = average_precision_score(y_test, y_score)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # 复制 repeat 次
        for _ in range(repeat):
            all_results[name]['auroc'].append(auroc)
            all_results[name]['aupr'].append(aupr)
            all_results[name]['acc'].append(acc)
            all_results[name]['precision'].append(precision)
            all_results[name]['recall'].append(recall)
            all_results[name]['f1'].append(f1)

        print(f">>> {name} 指标:")
        print(f"AUROC={auroc:.4f}, AUPR={aupr:.4f}, ACC={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


    # ===== MLP 模型训练（PyTorch） =====
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    if "MLP" in model_dict and model_dict["MLP"] == "mlp":
        for run_id in range(repeat):
            print(f"\n--- MLP 第 {run_id+1} 次实验 ---")
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
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    outputs = model(xb).squeeze(1)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(yb)
                avg_loss = total_loss / len(train_loader.dataset)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= PATIENCE:
                        break

            # 载入最佳模型
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                logits = model(X_test_tensor).squeeze(1)
                y_prob = torch.sigmoid(logits).cpu().numpy()
                y_pred = (y_prob >= threshold).astype(int)

            auroc = roc_auc_score(y_test, y_prob)
            aupr = average_precision_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            all_results.setdefault("MLP", {'auroc': [], 'aupr': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []})
            all_results["MLP"]['auroc'].append(auroc)
            all_results["MLP"]['aupr'].append(aupr)
            all_results["MLP"]['acc'].append(acc)
            all_results["MLP"]['precision'].append(precision)
            all_results["MLP"]['recall'].append(recall)
            all_results["MLP"]['f1'].append(f1)

            print(f"MLP 第 {run_id+1} 次指标: AUROC={auroc:.8f}, AUPR={aupr:.8f}, ACC={acc:.8f}, Precision={precision:.8f}, Recall={recall:.8f}, F1={f1:.8f}")

    # 写 CSV
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"])
        for name in all_results:
            results = all_results[name]
            n_runs = len(results['auroc'])
            for idx in range(n_runs):
                writer.writerow([
                    name, idx + 1,
                    f"{results['auroc'][idx]:.8f}",
                    f"{results['aupr'][idx]:.8f}",
                    f"{results['acc'][idx]:.8f}",
                    f"{results['precision'][idx]:.8f}",
                    f"{results['recall'][idx]:.8f}",
                    f"{results['f1'][idx]:.8f}"
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
    print(f"\n✅ 所有结果已保存至 {output_csv_path}")
