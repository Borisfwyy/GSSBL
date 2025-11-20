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

def run_lstmdual(
    train_path,
    test_path,
    word_vec_path,
    output_csv_path,
    repeat=5,
    cuda_device=0,
    use_subword=False,
    lr = 1e-4,     
    savemodel = True,
    savecanshu  = True,
    avg_tokens=None
):
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    
    HIDDEN_SIZE = 256
    DROPOUT = 0.3
    MAX_SEQ_LEN = 30
    BATCH_SIZE = 64
    PATIENCE = 100
    LR = lr
    EPOCHS = 100
    show_progress=False

    print(f"DualTower LSTM", "Using subword" if use_subword else "Using main word", f"Word vectors: {word_vec_path}, Device: {device}, LR={LR}, repeat={repeat}")
    
    with open(word_vec_path, "r", encoding="utf-8") as f:
        raw_vec = json.load(f)
    raw_vec = {k: np.array(v, dtype=np.float32) for k, v in raw_vec.items()}

    EMBED_DIM = next(iter(raw_vec.values())).shape[0]
    print(f"Detected embedding dimension EMBED_DIM = {EMBED_DIM}")

    def build_vocab(paths):
        vocab = set()
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3: continue
                    s1, s2 = parts[0], parts[1]
                    for w in s1.split():
                        vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
                    for w in s2.split():
                        vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
        return sorted(vocab)

    vocab_list = build_vocab([train_path, test_path])
    word2idx = {w: i + 1 for i, w in enumerate(vocab_list)}

    embedding_matrix = np.zeros((len(word2idx)+1, EMBED_DIM), dtype=np.float32)
    for w, idx in word2idx.items():
        if w in raw_vec:
            embedding_matrix[idx] = raw_vec[w]

    class DatasetWrap(Dataset):
        def __init__(self, filepath):
            self.data = []
            self.max_len = MAX_SEQ_LEN
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3: continue
                    s1, s2, label = parts[0], parts[1], int(parts[2])
                    self.data.append((s1, s2, label))

        def __len__(self):
            return len(self.data)

        def encode(self, sent):
            idxs = []
            for w in sent.strip().split():
                main_or_sub = w.split("|")[1] if use_subword else w.split("|")[0]
                idxs.append(word2idx.get(main_or_sub, 0))
            return idxs[:self.max_len]

        def __getitem__(self, idx):
            s1, s2, label = self.data[idx]
            return self.encode(s1), self.encode(s2), label

    def collate_fn(batch):
        s1s, s2s, labels = zip(*batch)
        len1s = [len(s) for s in s1s]
        len2s = [len(s) for s in s2s]
        max_len1 = max(len1s)
        max_len2 = max(len2s)

        s1_padded = torch.zeros(len(batch), max_len1, dtype=torch.long)
        s2_padded = torch.zeros(len(batch), max_len2, dtype=torch.long)
        for i, s in enumerate(s1s):
            s1_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        for i, s in enumerate(s2s):
            s2_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)

        labels = torch.tensor(labels, dtype=torch.float32)
        return s1_padded.to(device), s2_padded.to(device), labels.to(device), len1s, len2s

    class DualTowerLSTM(nn.Module):
        def __init__(self, embedding_matrix, hidden_size=256, dropout=0.3, avg_tokens=None):
            super().__init__()
            num_embeddings, embed_dim = embedding_matrix.shape
            self.embedding = nn.Embedding(num_embeddings, embed_dim, padding_idx=0)
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False

            self.lstm1 = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=False)
            self.lstm2 = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=False)
            self.dropout = nn.Dropout(dropout)
            self.avg_tokens = avg_tokens

            self.fc = nn.Sequential(
                nn.Linear(hidden_size*2, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )

        def encode(self, x, lengths, lstm):
            emb = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths=lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            if self.avg_tokens is None:
                mask = (x != 0).float().unsqueeze(2)
                h_avg = (out * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                n = min(self.avg_tokens, out.size(1))
                h_avg = out[:, :n, :].mean(dim=1)
            h_avg = self.dropout(h_avg)
            return h_avg

        def forward(self, s1, s2, len1, len2):
            u = self.encode(s1, len1, self.lstm1)
            v = self.encode(s2, len2, self.lstm2)
            features = torch.cat([u, v], dim=1)
            logits = self.fc(features).squeeze(1)
            return logits

    def train_one_epoch(model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0
        for s1, s2, labels, len1, len2 in tqdm(dataloader, desc="Training", disable=not show_progress):
            optimizer.zero_grad()
            logits = model(s1, s2, len1, len2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s1.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(model, dataloader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s1, s2, labels, len1, len2 in tqdm(dataloader, desc="Testing", disable=not show_progress):
                logits = model(s1, s2, len1, len2)
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

    train_dataset = DatasetWrap(train_path)
    test_dataset = DatasetWrap(test_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    results = []

    for run in range(1, repeat + 1):
        print(f"\nExperiment {run}/{repeat}")
        model = DualTowerLSTM(embedding_matrix, hidden_size=HIDDEN_SIZE, dropout=DROPOUT, avg_tokens=avg_tokens).to(device)
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
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        model.load_state_dict(best_state)
        AUROC, AUPR, ACC, PREC, REC, F1 = evaluate(model, test_loader)
        results.append({
            "AUROC": AUROC, "AUPR": AUPR, "Accuracy": ACC,
            "Precision": PREC, "Recall": REC, "F1": F1
        })
        print(f"Run {run}: AUROC={AUROC:.4f}, AUPR={AUPR:.4f}, ACC={ACC:.4f}, PREC={PREC:.4f}, REC={REC:.4f}, F1={F1:.4f}")

        if savemodel:
            model_dir = os.path.dirname(output_csv_path)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"dual_lstm_run_{run}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Model for run {run} saved to {model_path}")

        if savecanshu:
            thresholds = np.arange(0.1, 1.0, 0.1)
            rows = []
            preds_all, labels_all = [], []

            model.eval()
            with torch.no_grad():
                for s1, s2, labels, len1, len2 in test_loader:
                    logits = model(s1, s2, len1, len2)
                    preds_all.extend(torch.sigmoid(logits).cpu().numpy())
                    labels_all.extend(labels.cpu().numpy())

            preds_all = np.array(preds_all)
            labels_all = np.array(labels_all)
            for t in thresholds:
                y_pred = (preds_all >= t).astype(int)
                rows.append([t,
                            accuracy_score(labels_all, y_pred),
                            precision_score(labels_all, y_pred, zero_division=0),
                            recall_score(labels_all, y_pred, zero_division=0),
                            f1_score(labels_all, y_pred, zero_division=0)])

            thresh_csv_path = os.path.splitext(output_csv_path)[0] + f"_thresholds_run_{run}.csv"
            with open(thresh_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Threshold", "Accuracy", "Precision", "Recall", "F1"])
                writer.writerows(rows)
            print(f"Threshold metrics for run {run} saved to {thresh_csv_path}")

    fieldnames = ["Run", "AUROC", "AUPR", "Accuracy", "Precision", "Recall", "F1"]
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {"Run": i, **r}
            writer.writerow(row)
        avg_row = {"Run": "Mean"}
        for key in results[0]:
            avg_row[key] = np.mean([r[key] for r in results])
        writer.writerow(avg_row)
    print(f"\nAverage results: " + ", ".join([f"{k}: {v:.4f}" for k,v in avg_row.items() if k != "Run"]))
    print(f"\nResults saved to {output_csv_path}")
