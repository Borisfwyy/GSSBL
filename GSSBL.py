import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import copy
import csv
from tqdm import tqdm

def run_weighted_fusion_siamese(
    train_path,
    test_path,
    word_vec_path_meaning,
    word_vec_path_shape,
    output_csv_path,
    repeat,
    use_subword,
    cuda_device
):

    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 20
    LR = 1e-3
    EMBED_DIM = 512
    HIDDEN_SIZE = 256
    DROPOUT = 0.3
    MAX_SEQ_LEN = 30

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    def load_word_vec(path):
        with open(path, "r", encoding="utf-8") as f:
            return {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

    word_vec_meaning = load_word_vec(word_vec_path_meaning)
    word_vec_shape = load_word_vec(word_vec_path_shape)

    def build_vocab(paths):
        vocab = set()
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3: continue
                    for i in [0, 1]:
                        for w in parts[i].split():
                            vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
        return {w: i + 1 for i, w in enumerate(sorted(vocab))}

    word2idx = build_vocab([train_path, test_path])

    def create_embedding_matrix(word2idx, vec_dict):
        mat = np.zeros((len(word2idx) + 1, EMBED_DIM), dtype=np.float32)
        for w, idx in word2idx.items():
            if w in vec_dict:
                mat[idx] = vec_dict[w]
        return mat

    embedding_meaning = create_embedding_matrix(word2idx, word_vec_meaning)
    embedding_shape = create_embedding_matrix(word2idx, word_vec_shape)

    class DatasetSiamese(Dataset):
        def __init__(self, path):
            self.data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3: continue
                    s1, s2, label = parts
                    self.data.append((s1, s2, int(label)))

        def encode(self, sent):
            return [word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], 0)
                    for w in sent.strip().split()][:MAX_SEQ_LEN]

        def __getitem__(self, idx):
            s1, s2, label = self.data[idx]
            return self.encode(s1), self.encode(s2), label

        def __len__(self):
            return len(self.data)

    def collate_fn(batch):
        s1s, s2s, labels = zip(*batch)
        len1 = [len(s) for s in s1s]
        len2 = [len(s) for s in s2s]
        max1 = max(len1)
        max2 = max(len2)
        pad = lambda seqs, maxlen: torch.stack([torch.cat([torch.tensor(s), torch.zeros(maxlen - len(s))]) for s in seqs])
        return pad(s1s, max1).long().to(device), pad(s2s, max2).long().to(device), torch.tensor(labels).float().to(device), len1, len2

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = embedding_meaning.shape[0]

            self.emb_meaning = nn.Embedding.from_pretrained(torch.tensor(embedding_meaning), freeze=True, padding_idx=0)
            self.emb_shape = nn.Embedding.from_pretrained(torch.tensor(embedding_shape), freeze=True, padding_idx=0)
            self.bilstm_meaning = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)
            self.bilstm_shape = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, batch_first=True, bidirectional=True)

            self.dropout = nn.Dropout(DROPOUT)
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.fc = nn.Sequential(
                nn.Linear(HIDDEN_SIZE * 4, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(128, 1)
            )

        def encode(self, x, lens, emb, lstm):
            packed = nn.utils.rnn.pack_padded_sequence(emb(x), lens, batch_first=True, enforce_sorted=False)
            _, (h, _) = lstm(packed)
            return torch.cat([h[-2], h[-1]], dim=1)

        def forward(self, s1, s2, l1, l2):
            u_meaning = self.encode(s1, l1, self.emb_meaning, self.bilstm_meaning)
            v_meaning = self.encode(s2, l2, self.emb_meaning, self.bilstm_meaning)
            u_shape = self.encode(s1, l1, self.emb_shape, self.bilstm_shape)
            v_shape = self.encode(s2, l2, self.emb_shape, self.bilstm_shape)
            u = self.alpha * u_shape + (1 - self.alpha) * u_meaning
            v = self.alpha * v_shape + (1 - self.alpha) * v_meaning
            return self.fc(self.dropout(torch.cat([u, v], dim=1))).squeeze(1)

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for s1, s2, labels, l1, l2 in loader:
            optimizer.zero_grad()
            loss = criterion(model(s1, s2, l1, l2), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(loader.dataset)

    def evaluate(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s1, s2, labels, l1, l2 in loader:
                logits = model(s1, s2, l1, l2)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(labels.cpu().numpy())
        return roc_auc_score(trues, preds), average_precision_score(trues, preds)

    train_loader = DataLoader(DatasetSiamese(train_path), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(DatasetSiamese(test_path), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    auroc_list, aupr_list = [], []

    for run in range(repeat):
        model = Model().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, criterion, optimizer)
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        auroc, aupr = evaluate(model, test_loader)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
        print(f"✅ Run {run + 1}/{repeat}: AUROC = {auroc:.4f}, AUPR = {aupr:.4f}")

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Mean_AUROC", "Mean_AUPR"])
        writer.writerow([np.mean(auroc_list), np.mean(aupr_list)])

    print(f"\n✅ Done, avg AUROC = {np.mean(auroc_list):.4f}, avg AUPR = {np.mean(aupr_list):.4f}")
    print(f"📄 Results saved to: {output_csv_path}")
