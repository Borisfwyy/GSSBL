# 单塔，dougaihaole

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import csv
import copy
import os

def run_bilstm(
    train_path,
    test_path,
    word_vec_path,
    output_csv_path,
    repeat=5,
    cuda_device=0,
    use_subword=False,  # 新增参数，False用主字，True用子字
    lr = 1e-4,     
    savemodel = True,
    savecanshu  = True
):
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    
    # 固定超参
    HIDDEN_SIZE = 256
    DROPOUT = 0.3
    MAX_SEQ_LEN = 30
    BATCH_SIZE = 64
    PATIENCE = 100
    LR = lr
    EPOCHS = 100
    show_progress=False
    print(f"进度条显示？ {show_progress}")
    print("Bilstm", "使用子字" if use_subword else "使用主字", f"词向量: {word_vec_path}, 设备: {device}, LR={LR}, repeat={repeat}")
    # 加载词向量
    with open(word_vec_path, "r", encoding="utf-8") as f:
        raw_vec = json.load(f)
    raw_vec = {k: np.array(v, dtype=np.float32) for k, v in raw_vec.items()}

    # 自动检测嵌入维度
    example_vec = next(iter(raw_vec.values()))
    EMBED_DIM = example_vec.shape[0]
    print(f"自动检测到嵌入维度 EMBED_DIM = {EMBED_DIM}")

    # 构建词表
    def build_vocab(paths):
        vocab = set()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    s1, s2 = parts[0], parts[1]
                    for w in s1.split():
                        vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
                    for w in s2.split():
                        vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
        return sorted(vocab)

    vocab_list = build_vocab([train_path, test_path])
    word2idx = {w: i + 1 for i, w in enumerate(vocab_list)}  # 0留作padding

    embedding_matrix = np.zeros((len(word2idx) + 2, EMBED_DIM), dtype=np.float32)
    for w, idx in word2idx.items():
        if w in raw_vec:
            embedding_matrix[idx] = raw_vec[w]

    # ==== Dataset ====
    SEP_IDX = len(word2idx) + 1  # 全零索引

    class DatasetWrap(Dataset):
        def __init__(self, filepath):
            self.data = []
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    s1, s2, label = parts[0], parts[1], int(parts[2])
                    self.data.append((s1, s2, label))
        def encode(self, s1, s2):
            idxs = []
            for w in s1.strip().split():
                idxs.append(word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], 0))
            idxs.append(SEP_IDX)  # SEP
            for w in s2.strip().split():
                idxs.append(word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], 0))
            return idxs[:MAX_SEQ_LEN*2+1]  # 调整最大长度

        def __getitem__(self, idx):
            s1, s2, label = self.data[idx]
            return self.encode(s1, s2), label
        
        def __len__(self):
            return len(self.data)

    def collate_fn(batch):
        seqs, labels = zip(*batch)
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32)
        return padded.to(device), labels.to(device), lengths


    # ==== dan BiLSTM ====
    class SingleTowerBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            num_embeddings, embed_dim = embedding_matrix.shape
            self.embedding = nn.Embedding(num_embeddings, embed_dim, padding_idx=0)
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False

            self.bilstm = nn.LSTM(embed_dim, HIDDEN_SIZE, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(DROPOUT)
            input_dim = HIDDEN_SIZE * 2  # 单塔输出维度
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(128, 1)
            )

        def forward(self, x, lengths):
            emb = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths=lengths, batch_first=True, enforce_sorted=False)
            _, (h, _) = self.bilstm(packed)
            h_cat = torch.cat((h[-2], h[-1]), dim=1)  # 双向拼接
            h_cat = self.dropout(h_cat)
            logits = self.fc(h_cat).squeeze(1)
            return logits


    # ==== 训练和评估函数 ====

    def train_one_epoch(model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        for seqs, labels, lengths in tqdm(dataloader, desc="训练", disable=not show_progress):
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)
        return total_loss / len(dataloader.dataset)


    def evaluate(model, dataloader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seqs, labels, lengths in tqdm(dataloader, desc="测试", disable=not show_progress):
                logits = model(seqs, lengths)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(labels.cpu().numpy())
        preds_bin = [1 if p >= 0.5 else 0 for p in preds]
        AUROC = roc_auc_score(trues, preds)
        AUPR = average_precision_score(trues, preds)
        ACC = accuracy_score(trues, preds_bin)
        PREC = precision_score(trues, preds_bin, zero_division=0)
        REC = recall_score(trues, preds_bin, zero_division=0)
        F1 = f1_score(trues, preds_bin, zero_division=0)
        return AUROC, AUPR, ACC, PREC, REC, F1

    # ==== 数据加载 ====
    train_dataset = DatasetWrap(train_path)
    test_dataset = DatasetWrap(test_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ==== 多次训练 ====
    results = []

    for run in range(1, repeat + 1):
        print(f"\n🔁 实验 {run}/{repeat}")
        model = SingleTowerBiLSTM().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"⚠️ 提前终止训练 after {epoch} epochs")
                    break

        model.load_state_dict(best_state)
        AUROC, AUPR, ACC, PREC, REC, F1 = evaluate(model, test_loader)
        results.append({
            "AUROC": AUROC, "AUPR": AUPR, "Accuracy": ACC,
            "Precision": PREC, "Recall": REC, "F1": F1
        })
        print(f"✅ Run {run}: AUROC={AUROC:.4f}, AUPR={AUPR:.4f}, ACC={ACC:.4f}, PREC={PREC:.4f}, REC={REC:.4f}, F1={F1:.4f}")

        # 保存模型
        if savemodel:
            model_dir = os.path.dirname(output_csv_path)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"bilstm_run_{run}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ 第 {run} 次训练模型已保存至 {model_path}")

        # 保存阈值指标
        if savecanshu:
            thresholds = np.arange(0.1, 1.0, 0.1)
            rows = []
            preds_all, labels_all = [], []

            model.eval()
            with torch.no_grad():
                for seqs, labels, lengths in test_loader:
                    logits = model(seqs, lengths)
                    preds_all.extend(torch.sigmoid(logits).cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())

            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)

            for t in thresholds:
                y_pred = (preds_all >= t).astype(int)
                rows.append([
                    t,
                    accuracy_score(labels_all, y_pred),
                    precision_score(labels_all, y_pred, zero_division=0),
                    recall_score(labels_all, y_pred, zero_division=0),
                    f1_score(labels_all, y_pred, zero_division=0)
                ])

            thresh_csv_path = os.path.splitext(output_csv_path)[0] + f"_thresholds_run_{run}.csv"
            with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Threshold", "Accuracy", "Precision", "Recall", "F1"])
                writer.writerows(rows)

            print(f"✅ 第 {run} 次训练阈值指标已保存至 {thresh_csv_path}")


    # ==== 保存CSV ====
    fieldnames = ["Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {"Run": i, **r}
            writer.writerow(row)
        # 平均值
        avg_row = {"Run": "Mean"}
        for key in results[0]:
            avg_row[key] = np.mean([r[key] for r in results])
        writer.writerow(avg_row)
    print(f"\n平均结果: " + ", ".join([f"{k}: {v:.4f}" for k,v in avg_row.items() if k != "Run"]))
    print(f"\n📄 结果已保存至 {output_csv_path}")
