import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import csv
import os

def run_bert(train_path, test_path, word_vec_path, output_csv, repeat=5, cuda_device=None, use_subword=False):
    # 设备选择
    if torch.cuda.is_available() and cuda_device is not None:
        device = torch.device(f"cuda:{cuda_device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    PAD_TOKEN = "[PAD]"
    EMBED_DIM = 512
    MAX_SEQ_LEN = 60
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-5
    PATIENCE = 5
    NUM_HEADS = 8
    NUM_LAYERS = 6

    print(f"transformer encoder 使用设备: {device}")

    # 加载词向量
    with open(word_vec_path, encoding="utf-8") as f:
        raw_vec = json.load(f)
    raw_vec = {k: np.array(v, dtype=np.float32) for k, v in raw_vec.items()}

    # 构建词表
    def build_vocab(paths):
        vocab = set([CLS_TOKEN, SEP_TOKEN, PAD_TOKEN])
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    s1, s2 = parts[0], parts[1]
                    for w in s1.split():
                        token = w.split("|")[1] if use_subword else w.split("|")[0]
                        vocab.add(token)
                    for w in s2.split():
                        token = w.split("|")[1] if use_subword else w.split("|")[0]
                        vocab.add(token)
        return sorted(vocab)

    vocab_list = build_vocab([train_path, test_path])
    word2idx = {w: i for i, w in enumerate(vocab_list)}

    # 构造embedding矩阵
    embedding_matrix = np.zeros((len(word2idx), EMBED_DIM), dtype=np.float32)
    for w, idx in word2idx.items():
        if w in raw_vec:
            embedding_matrix[idx] = raw_vec[w]
        elif w in [CLS_TOKEN, SEP_TOKEN]:
            embedding_matrix[idx] = np.random.normal(scale=0.02, size=EMBED_DIM)
        elif w == PAD_TOKEN:
            embedding_matrix[idx] = np.zeros(EMBED_DIM)
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.02, size=EMBED_DIM)

    class DatasetWrap(Dataset):
        def __init__(self, filepath):
            self.data = []
            self.cls_id = word2idx[CLS_TOKEN]
            self.sep_id = word2idx[SEP_TOKEN]
            self.pad_id = word2idx[PAD_TOKEN]
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    s1, s2, label = parts[0], parts[1], int(parts[2])
                    self.data.append((s1, s2, label))

        def __len__(self):
            return len(self.data)

        def encode(self, s1, s2):
            t1 = [word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], self.pad_id)
                  for w in s1.strip().split()]
            t2 = [word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], self.pad_id)
                  for w in s2.strip().split()]

            t1 = t1[:(MAX_SEQ_LEN // 2 - 2)]
            t2 = t2[:(MAX_SEQ_LEN // 2 - 1)]

            input_ids = [self.cls_id] + t1 + [self.sep_id] + t2 + [self.sep_id]
            token_type_ids = [0] * (len(t1) + 2) + [1] * (len(t2) + 1)
            attention_mask = [1] * len(input_ids)

            pad_len = MAX_SEQ_LEN - len(input_ids)
            input_ids += [self.pad_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len

            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)

        def __getitem__(self, idx):
            s1, s2, label = self.data[idx]
            input_ids, token_type_ids, attention_mask = self.encode(s1, s2)
            return input_ids, token_type_ids, attention_mask, torch.tensor(label, dtype=torch.float32)

    # 模型结构，BERT简化版
    class BERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = embedding_matrix.shape[0]
            self.embed_dim = EMBED_DIM
            self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=True)
            self.segment_embeddings = nn.Embedding(2, EMBED_DIM)
            self.position_embeddings = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
            self.dropout = nn.Dropout(0.3)

            encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM,
                                                       nhead=NUM_HEADS,
                                                       dim_feedforward=EMBED_DIM * 4,
                                                       dropout=0.3,
                                                       activation='gelu')
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

            self.classifier = nn.Sequential(
                nn.Linear(EMBED_DIM, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, input_ids, token_type_ids, attention_mask):
            positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = self.word_embeddings(input_ids) + \
                self.segment_embeddings(token_type_ids) + \
                self.position_embeddings(positions)
            x = self.dropout(x).transpose(0, 1)

            key_padding_mask = (attention_mask == 0)
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
            x = x.transpose(0, 1)

            cls_vec = x[:, 0]
            logits = self.classifier(self.dropout(cls_vec)).squeeze(1)
            return logits

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for input_ids, token_type_ids, attention_mask, labels in tqdm(loader, desc="训练"):
            input_ids, token_type_ids, attention_mask, labels = \
                input_ids.to(device), token_type_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
        return total_loss / len(loader.dataset)

    def eval_model(model, loader):
        model.eval()
        probs_all = []
        labels_all = []
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, labels in tqdm(loader, desc="测试"):
                input_ids, token_type_ids, attention_mask = \
                    input_ids.to(device), token_type_ids.to(device), attention_mask.to(device)
                logits = model(input_ids, token_type_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                labels = labels.cpu().numpy()
                probs_all.extend(probs)
                labels_all.extend(labels)
        y_pred = (np.array(probs_all) >= 0.5).astype(int)
        return {
            "auroc": roc_auc_score(labels_all, probs_all),
            "aupr": average_precision_score(labels_all, probs_all),
            "acc": accuracy_score(labels_all, y_pred),
            "precision": precision_score(labels_all, y_pred, zero_division=0),
            "recall": recall_score(labels_all, y_pred, zero_division=0),
            "f1": f1_score(labels_all, y_pred, zero_division=0)
        }

    # 多次训练
    results_all = []

    for run_id in range(1, repeat + 1):
        print(f"\n=== 第 {run_id} 次训练 ===")
        train_data = DatasetWrap(train_path)
        test_data = DatasetWrap(test_path)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

        model = BERTClassifier().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        patience_count = 0

        for _epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f"Epoch {_epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}")
            if train_loss < best_loss:
                best_loss = train_loss
                patience_count = 0
                best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    break

        model.load_state_dict(best_weights)
        metrics = eval_model(model, test_loader)
        print(f"Run {run_id} -- " + ", ".join([f"{k}: {v:.4f}" for k,v in metrics.items()]))
        results_all.append(metrics)

    # 写入 CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"])
        for i, m in enumerate(results_all, 1):
            writer.writerow([i, m["auroc"], m["aupr"], m["acc"], m["precision"], m["recall"], m["f1"]])
        # 平均值
        avg_metrics = {k: np.mean([r[k] for r in results_all]) for k in results_all[0]}
        writer.writerow(["mean", avg_metrics["auroc"], avg_metrics["aupr"], avg_metrics["acc"],
                         avg_metrics["precision"], avg_metrics["recall"], avg_metrics["f1"]])
    print(f"\nMEAN结果: " + ", ".join([f"{k}: {v:.4f}" for k,v in avg_metrics.items()]))
    print(f"\n✅ 所有结果及平均值已保存至 {output_csv}")
