def run_bowmixdual(
    train_path: str,
    test_path: str,
    output_csv_path: str,
    repeat: int = 5,
    cuda_device: int | None = None,
    use_subword: bool = False,
    lr=1e-4,
    model_dict=None  # 新增参数：模型词典控制是否跑
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    import copy
    import os
    import pandas as pd

    # ==== 配置 ====
    MAX_FEATURES = 2000
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 100
    LR = lr
    THRESHOLD = 0.5

    if model_dict is None:
        model_dict = {
            "MLP": False,
            "LR": False,
            "SVM": True,
            "XGBoost": False,
            "LightGBM": False
        }

    print(f"使用子字: {use_subword}, cuda_device: {cuda_device}, BOW..., lr={LR}")

    DEVICE = torch.device(f"cuda:{cuda_device}" if cuda_device is not None and torch.cuda.is_available() else "cpu")

    # ==== MLP模型 ====
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    # ==== 数据加载 ====
    def extract_main(text):
        if use_subword:
            return " ".join([w.split("|")[1] if "|" in w else w for w in text.split()])
        else:
            return " ".join([w.split("|")[0] for w in text.split()])

    def load_data(path):
        texts_a, texts_b, labels = [], [], []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                a, b, label = parts
                texts_a.append(a)
                texts_b.append(b)
                labels.append(int(label))
        return texts_a, texts_b, labels

    print("加载训练集...")
    train_texts_a, train_texts_b, train_labels = load_data(train_path)
    print("加载测试集...")
    test_texts_a, test_texts_b, test_labels = load_data(test_path)

    print("向量化...")

    # ==== TF-IDF 向量化，双塔拼接 ====
    # 为每个句子单独生成主字/子字文本
    def build_texts(texts):
        main_texts = [" ".join([w.split("|")[0] for w in extract_main(t).split()]) for t in texts]
        sub_texts  = [" ".join([w.split("|")[1] if "|" in w else w for w in extract_main(t).split()]) for t in texts]
        return main_texts, sub_texts

    train_a_main, train_a_sub = build_texts(train_texts_a)
    train_b_main, train_b_sub = build_texts(train_texts_b)
    test_a_main,  test_a_sub  = build_texts(test_texts_a)
    test_b_main,  test_b_sub  = build_texts(test_texts_b)

    # 构建 TF-IDF 向量器
    vectorizer_a_main = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer_b_main = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer_a_sub  = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer_b_sub  = TfidfVectorizer(max_features=MAX_FEATURES)

    # 拟合训练集
    X_train_a_main = vectorizer_a_main.fit_transform(train_a_main).toarray()
    X_train_b_main = vectorizer_b_main.fit_transform(train_b_main).toarray()
    X_train_a_sub  = vectorizer_a_sub.fit_transform(train_a_sub).toarray()
    X_train_b_sub  = vectorizer_b_sub.fit_transform(train_b_sub).toarray()

    # 测试集 transform
    X_test_a_main = vectorizer_a_main.transform(test_a_main).toarray()
    X_test_b_main = vectorizer_b_main.transform(test_b_main).toarray()
    X_test_a_sub  = vectorizer_a_sub.transform(test_a_sub).toarray()
    X_test_b_sub  = vectorizer_b_sub.transform(test_b_sub).toarray()

    # 拼接四个向量
    X_train = np.concatenate([X_train_a_main, X_train_b_main, X_train_a_sub, X_train_b_sub], axis=1)
    X_test  = np.concatenate([X_test_a_main, X_test_b_main, X_test_a_sub, X_test_b_sub], axis=1)

    input_dim = X_train.shape[1]
    print(f"训练样本数: {len(X_train)}, 输入维度: {input_dim}")

    # ==== 后续训练逻辑不变 ====
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
            output = model(xb)
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
                output = model(xb)
                preds.extend(torch.sigmoid(output).detach().cpu().numpy())
        preds = np.array(preds)
        y = np.array(y)
        auroc = roc_auc_score(y, preds)
        aupr = average_precision_score(y, preds)
        pred_labels = (preds >= threshold).astype(int)
        acc = accuracy_score(y, pred_labels)
        precision = precision_score(y, pred_labels, zero_division=0)
        recall = recall_score(y, pred_labels, zero_division=0)
        f1 = f1_score(y, pred_labels, zero_division=0)
        return auroc, aupr, acc, precision, recall, f1

    all_results = []

    # === MLP训练 ===
    if model_dict.get("MLP", False):
        for run_id in range(1, repeat + 1):
            print(f"\n=== 第 {run_id} 次 MLP 训练 ===")
            model = MLP(input_dim).to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            best_loss = float('inf')
            best_epoch = 0
            best_state = None
            counter = 0

            for epoch in range(1, EPOCHS + 1):
                loss = train_epoch(model, optimizer, criterion, X_train, train_labels)
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= PATIENCE:
                        break

            model.load_state_dict(best_state)
            auroc, aupr, acc, precision, recall, f1 = evaluate_model(model, X_test, test_labels)
            print(f"MLP Run {run_id} Best Epoch: {best_epoch}, Loss: {best_loss:.8f}, "
                  f"AUROC: {auroc:.8f}, AUPR: {aupr:.8f}, "
                  f"Acc: {acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, F1: {f1:.8f}")

            all_results.append({
                'model': 'MLP', 'run': run_id,
                'best_epoch': best_epoch, 'loss': best_loss,
                'auroc': auroc, 'aupr': aupr,
                'acc': acc, 'precision': precision,
                'recall': recall, 'f1': f1
            })

    # === 其他模型训练 ===
    print("\n=== 基础算法训练 ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, run_flag in model_dict.items():
        if name == "MLP" or not run_flag:
            continue

        print(f"训练模型: {name}")
        if name == "LR":
            m = LogisticRegression(max_iter=100, class_weight='balanced')
            m.fit(X_train_scaled, train_labels)
            y_prob = m.decision_function(X_test_scaled)
        elif name == "SVM":
            m = LinearSVC(max_iter=1000, class_weight='balanced', C=1, loss="hinge")
            m.fit(X_train_scaled, train_labels)
            df = m.decision_function(X_test_scaled)
            y_prob = 1 / (1 + np.exp(-df))
        elif name == "XGBoost":
            m = XGBClassifier(use_label_encoder=False)
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

        print(f"{name}: AUROC: {auroc:.8f}, AUPR: {aupr:.8f}, "
              f"Acc: {acc:.8f}, Precision: {precision:.8f}, Recall: {recall:.8f}, F1: {f1:.8f}")

        all_results.append({
            'model': name, 'run': 1,
            'auroc': auroc, 'aupr': aupr,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # === 保存结果 ===
    df = pd.DataFrame(all_results)
    summary = df.groupby('model').mean(numeric_only=True).reset_index()
    summary['run'] = 'mean'
    final_df = pd.concat([df, summary], ignore_index=True)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"\n✅ 结果已保存到 {output_csv_path}")
