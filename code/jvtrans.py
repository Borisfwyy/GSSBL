
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import copy
import csv
from tqdm import tqdm

def run_weighted_fusion_siamese_trans(
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
    EPOCHS = 50
    PATIENCE = 5
    LR = 1e-6
    EMBED_DIM = 512
    MAX_SEQ_LEN = 30
    NUM_HEADS = 8
    NUM_LAYERS = 6
    DROPOUT = 0.3
    THRESHOLD = 0.5  # 分类阈值

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")


    def load_word_vec(path):
        with open(path, "r", encoding="utf-8") as f:
            vec_dict = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
        print(f"Loaded {len(vec_dict)} word vectors from {path}")
        return vec_dict

    word_vec_meaning = load_word_vec(word_vec_path_meaning)
    word_vec_shape = load_word_vec(word_vec_path_shape)

    SPECIAL_TOKENS = ["<PAD>", "<CLS>", "<SEP>"]
    def build_vocab(paths):
        vocab = set(SPECIAL_TOKENS)
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3: continue
                    for i in [0, 1]:
                        for w in parts[i].split():
                            vocab.add(w.split("|")[1] if use_subword else w.split("|")[0])
        print(f"Vocab size: {len(vocab)}")
        return {w: i for i, w in enumerate(sorted(vocab))}

    word2idx = build_vocab([train_path, test_path])
    PAD_ID = word2idx["<PAD>"]
    CLS_ID = word2idx["<CLS>"]
    SEP_ID = word2idx["<SEP>"]


    def create_embedding_matrix(word2idx, vec_dict):
        mat = np.zeros((len(word2idx), EMBED_DIM), dtype=np.float32)
        for w, idx in word2idx.items():
            if w in vec_dict:
                mat[idx] = vec_dict[w]
        print(f"Embedding matrix shape: {mat.shape}")
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
            tokens = [CLS_ID] + [
                word2idx.get(w.split("|")[1] if use_subword else w.split("|")[0], 0)
                for w in sent.strip().split()
            ][:MAX_SEQ_LEN] + [SEP_ID]
            return tokens

        def __getitem__(self, idx):
            s1, s2, label = self.data[idx]
            return self.encode(s1), self.encode(s2), label

        def __len__(self):
            return len(self.data)

    def pad_seq(seqs, maxlen):
        padded = torch.full((len(seqs), maxlen), PAD_ID, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return padded

    def collate_fn(batch):
        s1s, s2s, labels = zip(*batch)
        len1 = [len(s) for s in s1s]
        len2 = [len(s) for s in s2s]
        max1, max2 = max(len1), max(len2)
        return pad_seq(s1s, max1).to(device), pad_seq(s2s, max2).to(device), torch.tensor(labels).float().to(device), len1, len2


    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = embedding_meaning.shape[0]

            # embedding
            self.emb_meaning = nn.Embedding.from_pretrained(torch.tensor(embedding_meaning), freeze=True, padding_idx=PAD_ID)
            self.emb_shape = nn.Embedding.from_pretrained(torch.tensor(embedding_shape), freeze=True, padding_idx=PAD_ID)
            self.position_embeddings = nn.Embedding(MAX_SEQ_LEN + 2, EMBED_DIM)  

            self.dropout = nn.Dropout(DROPOUT)
            self.alpha = nn.Parameter(torch.tensor(0.5))  

            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NUM_HEADS,
                dim_feedforward=EMBED_DIM * 4,
                dropout=DROPOUT,
                activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

            # MLP 分类器
            self.fc = nn.Sequential(
                nn.Linear(EMBED_DIM * 2, 512), 
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(512, 1)
            )

        def encode(self, x, lens):
            batch_size, seq_len = x.size()
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            x_emb = (1 - self.alpha) * self.emb_meaning(x) + self.alpha * self.emb_shape(x) + self.position_embeddings(positions)
            x_emb = self.dropout(x_emb).transpose(0,1)  

            key_padding_mask = (x == PAD_ID)
            x_enc = self.encoder(x_emb, src_key_padding_mask=key_padding_mask)
            x_enc = x_enc.transpose(0,1)  
            return x_enc[:, 0, :]  

        def forward(self, s1, s2, l1, l2):
            u = self.encode(s1, l1)
            v = self.encode(s2, l2)
            return self.fc(self.dropout(torch.cat([u, v], dim=1))).squeeze(1)


    def train_one_epoch(model, loader, criterion, optimizer, epoch_num, total_epochs):
        model.train()
        total_loss = 0
        for i, (s1, s2, labels, l1, l2) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = criterion(model(s1, s2, l1, l2), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            print(f"\rBatch {i}/{len(loader)} processed...", end="")
        epoch_loss = total_loss / len(loader.dataset)
        print(f"\nEpoch {epoch_num}/{total_epochs} finished, Loss: {epoch_loss:.4f}")
        return epoch_loss

    def evaluate(model, loader):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s1, s2, labels, l1, l2 in loader:
                logits = model(s1, s2, l1, l2)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(labels.cpu().numpy())
        auroc = roc_auc_score(trues, preds)
        aupr = average_precision_score(trues, preds)
        pred_labels = [1 if p >= THRESHOLD else 0 for p in preds]
        acc = accuracy_score(trues, pred_labels)
        precision = precision_score(trues, pred_labels, zero_division=0)
        recall = recall_score(trues, pred_labels, zero_division=0)
        f1 = f1_score(trues, pred_labels, zero_division=0)
        return auroc, aupr, acc, precision, recall, f1


    train_loader = DataLoader(DatasetSiamese(train_path), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(DatasetSiamese(test_path), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


    auroc_list, aupr_list = [], []
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for run in range(repeat):
        model = Model().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch_num=epoch+1, total_epochs=EPOCHS)
            if loss < best_loss:
                best_loss = loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        auroc, aupr, acc, prec, rec, f1 = evaluate(model, test_loader)
        auroc_list.append(auroc)
        aupr_list.append(aupr)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        print(f"✅ Run {run + 1}/{repeat}: AUROC={auroc:.4f}, AUPR={aupr:.4f}, Acc={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

  
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Run", "AUROC", "AUPR", "Acc", "Precision", "Recall", "F1"])
      
        for i in range(repeat):
            writer.writerow([
                i+1,
                auroc_list[i],
                aupr_list[i],
                acc_list[i],
                prec_list[i],
                rec_list[i],
                f1_list[i]
            ])
  
        writer.writerow([
            "Mean",
            np.mean(auroc_list),
            np.mean(aupr_list),
            np.mean(acc_list),
            np.mean(prec_list),
            np.mean(rec_list),
            np.mean(f1_list)
        ])


    print(f"\n✅ 所有实验完成，平均 AUROC={np.mean(auroc_list):.4f}, 平均 AUPR={np.mean(aupr_list):.4f}")
    print(f"平均指标: Acc={np.mean(acc_list):.4f}, Precision={np.mean(prec_list):.4f}, Recall={np.mean(rec_list):.4f}, F1={np.mean(f1_list):.4f}")
    print(f"📄 结果保存至: {output_csv_path}")
