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
    use_subword: bool = True   
):

    BATCH_SIZE = 64
    EPOCHS = 100
    PATIENCE = 10
    LEARNING_RATE = 5e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 30
    EMBED_DIM = 512
    FEATURE_MODE = "concat"

    if torch.cuda.is_available():
        if cuda_device is None:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device(f"cuda:{cuda_device}")
    else:
        DEVICE = torch.device("cpu")

    with open(word_vec_path, "r", encoding="utf-8") as f:
        word_vec = json.load(f)
    word_vec = {k: torch.tensor(v, dtype=torch.float32) for k, v in word_vec.items()}

    # ==== 构建词表 ====
    def build_vocab(paths, use_subword=True):
        vocab = {}
        idx = 1
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    for sent in parts[:2]:
                        words = sent.strip().split()
                        for w in words:
                            token = w.split("|")[1] if use_subword and len(w.split("|"))>1 else w.split("|")[0]
                            if token not in vocab:
                                vocab[token] = idx
                                idx += 1
        return vocab

    vocab = build_vocab([train_path, test_path], use_subword)
    vocab_size = len(vocab) + 1 


    def sent2idx(sent, vocab, max_len=MAX_LEN, use_subword=True):
        idxs = []
        words = sent.strip().split()
        for w in words:
            token = w.split("|")[1] if use_subword and len(w.split("|"))>1 else w.split("|")[0]
            idxs.append(vocab.get(token, 0))
        if len(idxs) > max_len:
            idxs = idxs[:max_len]
        else:
            idxs += [0] * (max_len - len(idxs))
        return idxs

    
    embedding_matrix = torch.zeros(vocab_size, EMBED_DIM)

    for token, idx in vocab.items():

        if token in word_vec:
            embedding_matrix[idx] = word_vec[token]
        else:
          
            main = token.split("|")[0]  
            if main in word_vec:
                embedding_matrix[idx] = word_vec[main]



    class TextPairDataset(Dataset):
        def __init__(self, path, vocab, max_len=MAX_LEN, use_subword=True):
            self.data = []
            self.use_subword = use_subword
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    sent_a, sent_b, label = parts[0], parts[1], int(parts[2])
                    self.data.append((sent2idx(sent_a, vocab, max_len, use_subword),
                                      sent2idx(sent_b, vocab, max_len, use_subword),
                                      label))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            a, b, label = self.data[idx]
            return torch.tensor(a), torch.tensor(b), torch.tensor(label, dtype=torch.float32)

   
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
            feature_dim = 2 * self.pool_out_dim

            self.fc = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim)
            )

        def forward(self, a, b):
            a_emb = self.embedding(a).permute(0, 2, 1)
            b_emb = self.embedding(b).permute(0, 2, 1)

            a_conved = [torch.relu(conv(a_emb)) for conv in self.convs]
            b_conved = [torch.relu(conv(b_emb)) for conv in self.convs]

            a_pooled = [torch.max(conv, dim=2)[0] for conv in a_conved]
            b_pooled = [torch.max(conv, dim=2)[0] for conv in b_conved]

            u = torch.cat(a_pooled, dim=1)
            v = torch.cat(b_pooled, dim=1)

            features = torch.cat([u, v], dim=1)
            out = self.fc(features)
            return out.squeeze()


    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        losses = []
        for a, b, labels in tqdm(loader, desc="训练"):
            a, b, labels = a.to(DEVICE), b.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(a, b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def test_model(model, loader):
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for a, b, labels in tqdm(loader, desc="测试"):
                a, b, labels = a.to(DEVICE), b.to(DEVICE), labels.to(DEVICE)
                outputs = model(a, b)
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


    all_results = []

    for run_id in range(1, repeat + 1):
        print(f"\n=== 第 {run_id} 次训练 ===")

        train_dataset = TextPairDataset(train_path, vocab, MAX_LEN, use_subword)
        test_dataset = TextPairDataset(test_path, vocab, MAX_LEN, use_subword)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = TextCNN(embedding_matrix).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()

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
                print(f"训练集loss未下降（{patience_counter}/{PATIENCE}）")
                if patience_counter >= PATIENCE:
                    print(f"训练提前终止，连续{PATIENCE}轮loss未下降")
                    break

        model.load_state_dict(best_state_dict)
        model.to(DEVICE)
        auroc, aupr, acc, precision, recall, f1 = test_model(model, test_loader)

        print(f"Run {run_id} -- AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        all_results.append({'auroc': auroc, 'aupr': aupr, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1})


   
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