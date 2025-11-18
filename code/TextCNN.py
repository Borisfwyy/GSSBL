# 单塔 TextCNN 句子对合并版

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
import csv
import os

def run_textcnn(
    train_path: str,
    test_path: str,
    word_vec_path: str,
    output_csv_path: str,
    repeat: int = 5,
    cuda_device: int | None = None,
    use_subword: bool = True,   # True只用子字，False用主字
    lr = 1e-4,
    savemodel = False,
    savecanshu  = True
):
    # ==== 固定超参数 ====
    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 100
    LEARNING_RATE = lr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 30
    SEQ_LEN = MAX_LEN * 2 + 1  # 两句合并 + 1个SEP
    show_progress = False
    print(f"进度条显示？ {show_progress}")

    if torch.cuda.is_available():
        if cuda_device is None:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device(f"cuda:{cuda_device}")
    else:
        DEVICE = torch.device("cpu")

    print(f"textcnn， 使用设备: {DEVICE}, use_subword: {use_subword}, lr={LEARNING_RATE}")

    # ==== 加载词向量 ====
    with open(word_vec_path, "r", encoding="utf-8") as f:
        word_vec = json.load(f)
    word_vec = {k: torch.tensor(v, dtype=torch.float32) for k, v in word_vec.items()}

    EMBED_DIM = len(next(iter(word_vec.values())))
    print(f"Embedding 维度自动设置为 {EMBED_DIM}")


    # ==== 构建词表 ====
    def build_vocab(paths, use_subword=True):
        vocab = {}
        idx = 1  # 0 用作 padding / SEP
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    for sent in parts[:2]:
                        for w in sent.strip().split():
                            token = w.split("|")[1] if use_subword and len(w.split("|"))>1 else w.split("|")[0]
                            if token not in vocab:
                                vocab[token] = idx
                                idx += 1
        return vocab

    vocab = build_vocab([train_path, test_path], use_subword)
    vocab_size = len(vocab) + 1  # 0 padding/SEP

    # ==== 合并句子生成索引序列 ====
    def sentpair2idx(sent_a, sent_b, vocab, max_len=SEQ_LEN, use_subword=True, sep_idx=0):
        idxs_a = [vocab.get(w.split("|")[1] if use_subword and "|" in w else w.split("|")[0], 0)
                  for w in sent_a.split()]
        idxs_b = [vocab.get(w.split("|")[1] if use_subword and "|" in w else w.split("|")[0], 0)
                  for w in sent_b.split()]
        combined = idxs_a + [sep_idx] + idxs_b
        if len(combined) > max_len:
            combined = combined[:max_len]
        else:
            combined += [0]*(max_len - len(combined))
        return combined

    # ==== 构建 embedding 矩阵 ====
    embedding_matrix = torch.zeros(vocab_size, EMBED_DIM)
    for token, idx in vocab.items():
        if token in word_vec:
            embedding_matrix[idx] = word_vec[token]
        else:
            main = token.split("|")[0]
            if main in word_vec:
                embedding_matrix[idx] = word_vec[main]

    # ==== Dataset ====
    class TextPairDataset(Dataset):
        def __init__(self, path, vocab, max_len=SEQ_LEN, use_subword=True):
            self.data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    sent_a, sent_b, label = parts[0], parts[1], int(parts[2])
                    self.data.append((sentpair2idx(sent_a, sent_b, vocab, max_len, use_subword), label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x, label = self.data[idx]
            return torch.tensor(x), torch.tensor(label, dtype=torch.float32)

    # ==== TextCNN 模型 ====
    class TextCNN(nn.Module):
        def __init__(self, embedding_matrix, n_filters=100, filter_sizes=[3,4,5], output_dim=1, dropout=0.3):
            super().__init__()
            vocab_size, embed_dim = embedding_matrix.shape
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=fs)
                for fs in filter_sizes
            ])
            self.pool_out_dim = len(filter_sizes) * n_filters
            feature_dim = self.pool_out_dim  # 单塔，不再翻倍
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim)
            )

        def forward(self, x):
            x = self.embedding(x).permute(0,2,1)  # [batch, embed_dim, seq_len]
            conved = [torch.relu(conv(x)) for conv in self.convs]
            pooled = [torch.max(c, dim=2)[0] for c in conved]
            features = torch.cat(pooled, dim=1)
            out = self.fc(features)
            return out.squeeze()

    # ==== 训练和测试函数 ====
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        losses = []
        for x, labels in tqdm(loader, desc="训练", disable=not show_progress):
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def test_model(model, loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, labels in tqdm(loader, desc="测试", disable=not show_progress):
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                outputs = model(x)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auroc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)
        y_pred = (np.array(all_probs) >= 0.5).astype(int)
        acc = accuracy_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        return auroc, aupr, acc, precision, recall, f1

    # ==== 多次训练 ====
    all_results = []

    for run_id in range(1, repeat + 1):
        print(f"\n=== 第 {run_id} 次训练 ===")

        train_dataset = TextPairDataset(train_path, vocab, SEQ_LEN, use_subword)
        test_dataset = TextPairDataset(test_path, vocab, SEQ_LEN, use_subword)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = TextCNN(embedding_matrix).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch}/{EPOCHS} | 训练损失: {train_loss:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"训练提前终止，连续{PATIENCE}轮loss未下降")
                    break

        model.load_state_dict(best_state_dict)
        model.to(DEVICE)
        auroc, aupr, acc, precision, recall, f1 = test_model(model, test_loader)

        # 保存阈值指标 CSV
        if savecanshu:
            thresholds = np.arange(0.1, 1.0, 0.1)
            rows = []
            all_probs, all_labels = [], []

            model.eval()
            with torch.no_grad():
                for x, labels in test_loader:
                    x, labels = x.to(DEVICE), labels.to(DEVICE)
                    outputs = model(x)
                    all_probs.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            probs = np.array(all_probs)
            labels = np.array(all_labels)
            for t in thresholds:
                y_pred = (probs >= t).astype(int)
                rows.append([t,
                             accuracy_score(labels, y_pred),
                             precision_score(labels, y_pred, zero_division=0),
                             recall_score(labels, y_pred, zero_division=0),
                             f1_score(labels, y_pred, zero_division=0)])

            thresh_csv_path = os.path.splitext(output_csv_path)[0] + f"_thresholds_run_{run_id}.csv"
            with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Threshold", "Accuracy", "Precision", "Recall", "F1"])
                writer.writerows(rows)
            print(f"✅ 阈值指标已保存到 {thresh_csv_path}")

        if savemodel:
            model_dir = os.path.dirname(output_csv_path)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"textcnn_run_{run_id}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ 第 {run_id} 次训练模型已保存至 {model_path}")

        print(f"Run {run_id} -- AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        all_results.append({'auroc': auroc, 'aupr': aupr, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})

    # ==== 保存 CSV ====
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"])
        for i, r in enumerate(all_results, start=1):
            writer.writerow([i, r['auroc'], r['aupr'], r['acc'], r['precision'], r['recall'], r['f1']])
        mean_auroc = np.mean([r['auroc'] for r in all_results])
        mean_aupr = np.mean([r['aupr'] for r in all_results])
        mean_acc = np.mean([r['acc'] for r in all_results])
        mean_precision = np.mean([r['precision'] for r in all_results])
        mean_recall = np.mean([r['recall'] for r in all_results])
        mean_f1 = np.mean([r['f1'] for r in all_results])
        writer.writerow(["mean", mean_auroc, mean_aupr, mean_acc, mean_precision, mean_recall, mean_f1])

    print(f"平均指标 -- AUROC: {mean_auroc:.4f}, AUPR: {mean_aupr:.4f}, Acc: {mean_acc:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, F1: {mean_f1:.4f}")
    print(f"\n✅ 所有 run 的结果及均值已保存至 {output_csv_path}")
