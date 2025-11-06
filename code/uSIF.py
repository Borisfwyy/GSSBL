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
from scipy.special import expit  # sigmoid
from cuml.svm import SVC

patch_sklearn()


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


def run_usif(word_vec_path: str, data_paths: list, output_csv_path: str, repeat: int = 5, cuda_device:int = 0, use_subword: bool = False):
    EMBED_DIM = 512
    SMOOTHING_A = 1e-3
    threshold = 0.5
    PATIENCE = 10

    train_path, test_path = data_paths
    print(f"cuda_device: {cuda_device}, uSIF...")


    with open(word_vec_path, "r", encoding="utf-8") as f:
        word_vec = json.load(f)
        word_vec = {k: np.array(v) for k, v in word_vec.items()}


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


    def prepare_features_labels(file_path, use_subword=use_subword):
        X, y = [], []
        for line in tqdm(open(file_path, "r", encoding="utf-8"), desc=f"生成特征: {file_path}"):
            a, b, label = line.strip().split("\t")
            va = compute_usif_embedding(a, word_vec, word_prob, use_subword=use_subword)
            vb = compute_usif_embedding(b, word_vec, word_prob, use_subword=use_subword)
            feat = np.concatenate([va, vb])
            X.append(feat)
            y.append(int(label))
        return np.array(X), np.array(y)

    X_train, y_train = prepare_features_labels(train_path, use_subword=use_subword)
    X_test, y_test = prepare_features_labels(test_path, use_subword=use_subword)
    input_dim = X_train.shape[1]


    model_dict = {
        "LR": "lr",
        "SVM": "svm",
        "XGBoost": "xgb",
        "LightGBM": "lgbm",
        "MLP": "mlp"
    }

    all_results = {name: {'auroc': [], 'aupr': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []} for name in model_dict.keys()}


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

  
    for name, mtype in model_dict.items():
        if mtype != "mlp":
            print(f"\n训练模型: {name}")
            if mtype == "lr":
                m = LogisticRegression(max_iter=2000, solver="saga", class_weight='balanced')
                m.fit(X_train_scaled, y_train)
                y_prob = m.decision_function(X_test_scaled)
            elif mtype == "svm":
                m = SVC(kernel='poly', degree=3, gamma=0.1)  
                m.fit(X_train_scaled, y_train)
                print("SVM训练完成，开始预测...")
                y_prob = m.decision_function(X_test_scaled)
                y_prob = expit(y_prob)
                print("SVM预测完成。")
            elif mtype == "xgb":
                m = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
                m.fit(X_train_scaled, y_train)
                y_prob = m.predict_proba(X_test_scaled)[:,1]
            elif mtype == "lgbm":
                m = LGBMClassifier()
                m.fit(X_train_scaled, y_train)
                y_prob = m.predict_proba(X_test_scaled)[:,1]

            y_pred = (y_prob >= threshold).astype(int)
            print(f"AUROC: {roc_auc_score(y_test, y_prob):.4f}, AUPR: {average_precision_score(y_test, y_prob):.4f}, "
                  f"Acc: {accuracy_score(y_test, y_pred):.4f}, Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}, "
                  f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}, F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
            all_results[name]['auroc'].append(roc_auc_score(y_test, y_prob))
            all_results[name]['aupr'].append(average_precision_score(y_test, y_prob))
            all_results[name]['acc'].append(accuracy_score(y_test, y_pred))
            all_results[name]['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            all_results[name]['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            all_results[name]['f1'].append(f1_score(y_test, y_pred, zero_division=0))


    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    has_mlp = 1 if "MLP" in model_dict else 0
    print(f"Has MLP: {has_mlp}")
    if has_mlp:
        for run_id in range(repeat):
            print(f"\n=== MLP 第 {run_id + 1} 次实验 ===")
            mlp = MLP(input_dim=input_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-4)
            train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            best_loss = float('inf')
            best_state = None
            wait = 0

            for epoch in range(1, 101):
                total_loss = 0
                for i, (xb, yb) in enumerate(train_loader, start=1):
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

            writer.writerow([
                name, "mean",
                f"{np.mean(results['auroc']):.4f}",
                f"{np.mean(results['aupr']):.4f}",
                f"{np.mean(results['acc']):.4f}",
                f"{np.mean(results['precision']):.4f}",
                f"{np.mean(results['recall']):.4f}",
                f"{np.mean(results['f1']):.4f}"
            ])
    print(f"\n✅ 结果均值已保存到 {output_csv_path}")
