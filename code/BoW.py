import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import copy
import os
import pandas as pd

def run_bow(
    train_path: str,
    test_path: str,
    output_csv_path: str,
    repeat: int = 5,
    cuda_device: int | None = None,
    use_subword: bool = False
):
    # ==== 内部配置 ====
    MAX_FEATURES = 2050
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 10
    LR = 1e-4
    THRESHOLD = 0.5
    USE_MLP = True

    if torch.cuda.is_available():
        if cuda_device is None:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device(f"cuda:{cuda_device}")
    else:
        DEVICE = torch.device("cpu")

    # ==== MLP模型 ====
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    # ==== 数据加载 ====
    def extract_main(text):
        if use_subword:
            return " ".join([w.split("|")[1] if "|" in w else w for w in text.split()])
        else:
            return " ".join([w.split("|")[0] for w in text.split()])

    def load_data(path):
        texts, labels = [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                a, b, label = parts
                combined = extract_main(a) + " " + extract_main(b)
                texts.append(combined)
                labels.append(int(label))
        return texts, labels

    print("加载训练集...")
    train_texts, train_labels = load_data(train_path)
    print("加载测试集...")
    test_texts, test_labels = load_data(test_path)

    print("向量化...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    input_dim = X_train.shape[1]
    print(f"训练样本数: {len(X_train)}, 输入维度: {input_dim}")

    def get_batches(X, y, batch_size):
        n = len(X)
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            yield (
                torch.tensor(xb, dtype=torch.float32).to(DEVICE),
                torch.tensor(yb, dtype=torch.float32).to(DEVICE)
            )

    def train_epoch(model, optimizer, criterion, X, y):
        model.train()
        total_loss = 0
        for xb, yb in get_batches(X, y, BATCH_SIZE):
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(y)

    def evaluate_model(model, X, y, threshold=THRESHOLD):
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in get_batches(X, y, BATCH_SIZE):
                output = model(xb).squeeze()
                preds.extend(output.detach().cpu().numpy())
        preds = np.array(preds)
        y = np.array(y)

        # 指标
        auroc = roc_auc_score(y, preds)
        aupr = average_precision_score(y, preds)
        pred_labels = (preds >= threshold).astype(int)
        acc = accuracy_score(y, pred_labels)
        precision = precision_score(y, pred_labels, zero_division=0)
        recall = recall_score(y, pred_labels, zero_division=0)
        f1 = f1_score(y, pred_labels, zero_division=0)

        return auroc, aupr, acc, precision, recall, f1

    all_results = []

    # === MLP重复多次 ===
    for run_id in range(1, repeat + 1):
        print(f"\n=== 第 {run_id} 次 MLP 训练 ===")

        model = MLP(input_dim).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_loss = float('inf')
        best_epoch = 0
        best_state = None
        counter = 0

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, optimizer, criterion, X_train, train_labels)
            print(f"Epoch {epoch}: Train Loss = {loss:.4f}")

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                print(f"→ Loss未下降，连续 {counter}/{PATIENCE} 次")
                if counter >= PATIENCE:
                    print("⚠️ 提前终止训练")
                    break

        model.load_state_dict(best_state)
        auroc, aupr, acc, precision, recall, f1 = evaluate_model(model, X_test, test_labels)
        print(f"MLP Run {run_id} Best Epoch: {best_epoch}, Loss: {best_loss:.4f}, "
              f"AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, "
              f"Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        all_results.append({
            'model': 'MLP', 'run': run_id,
            'best_epoch': best_epoch, 'loss': best_loss,
            'auroc': auroc, 'aupr': aupr,
            'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1
        })

    # === 传统分类器只跑一次 ===
    print("\n=== 基础算法训练 ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_dict = {
        "LogisticRegression": LogisticRegression(max_iter=10000, solver="saga"),
        "PolynomialSVM": SVC(kernel="poly", degree=3, probability=True, max_iter=10000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": LGBMClassifier()
    }

    for name, base_model in model_dict.items():
        print(f"训练模型: {name}")
        if name == "LogisticRegression":
            m = LogisticRegression(max_iter=2000, solver="saga", class_weight='balanced')
            m.fit(X_train_scaled, train_labels)
            y_prob = m.decision_function(X_test_scaled)
        elif name == "PolynomialSVM":
            m = SVC(kernel="poly", degree=3, probability=True,  gamma=0.1)
            m.fit(X_train_scaled, train_labels)
            y_prob = m.predict_proba(X_test_scaled)[:, 1]
        elif name == "XGBoost":
            m = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            m.fit(X_train, train_labels)
            y_prob = m.predict_proba(X_test)[:, 1]
        elif name == "LightGBM":
            m = LGBMClassifier()
            m.fit(X_train, train_labels)
            y_prob = m.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"未知模型: {name}")

        y = np.array(test_labels)
        auroc = roc_auc_score(y, y_prob)
        aupr = average_precision_score(y, y_prob)
        pred_labels = (y_prob >= THRESHOLD).astype(int)
        acc = accuracy_score(y, pred_labels)
        precision = precision_score(y, pred_labels, zero_division=0)
        recall = recall_score(y, pred_labels, zero_division=0)
        f1 = f1_score(y, pred_labels, zero_division=0)

        print(f"{name}: AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, "
              f"Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        all_results.append({
            'model': name, 'run': 1,
            'auroc': auroc, 'aupr': aupr,
            'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1
        })

    # === 保存结果 ===
    df = pd.DataFrame(all_results)
    summary = df.groupby('model').mean(numeric_only=True).reset_index()
    summary['run'] = 'mean'
    final_df = pd.concat([df, summary], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n✅ 结果已保存到 {output_csv_path}")
