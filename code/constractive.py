import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

class SharedMLP(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def load_embeddings(path):
    with open(path, 'r', encoding='utf8') as f:
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in json.load(f).items()}

def load_pairs(path):
    pairs = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) > 1:
                anchor, pos_list = items[0], items[1].split()
                pairs[anchor] = pos_list
    return pairs

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    neg_sims = torch.matmul(anchor, negatives.T) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)
    return F.cross_entropy(logits, labels)

def train(
    glyph_json,
    pos_file,
    neg_file,
    output_glyph_json,
    dim=512,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    temperature=0.07,
    patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    glyph_embed = load_embeddings(glyph_json)
    pos_pairs = load_pairs(pos_file)
    neg_pairs = load_pairs(neg_file)

    shared_mlp = SharedMLP(input_dim=dim, output_dim=512).to(device)
    optimizer = Adam(shared_mlp.parameters(), lr=lr)

    all_ids = list(glyph_embed.keys())

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(epochs):
        shared_mlp.train()
        random.shuffle(all_ids)
        total_loss = 0
        count_batches = 0
        for i in tqdm(range(0, len(all_ids), batch_size), desc=f"Epoch {epoch+1}"):
            batch_ids = all_ids[i:i+batch_size]
            anchors = []
            positives = []
            negatives = []

            for anchor_id in batch_ids:
                if anchor_id not in pos_pairs or anchor_id not in neg_pairs:
                    continue
                pos_ids = pos_pairs[anchor_id]
                neg_ids = neg_pairs[anchor_id]
                if not pos_ids or not neg_ids:
                    continue

                pos_id = random.choice(pos_ids)
                sampled_negs = random.sample(neg_ids, min(len(neg_ids), 10))

                anchors.append(glyph_embed[anchor_id])
                positives.append(glyph_embed[pos_id])
                negatives.append(torch.stack([glyph_embed[nid] for nid in sampled_negs]))

            if not anchors:
                continue

            anchor_batch = shared_mlp(torch.stack(anchors).to(device))
            positive_batch = shared_mlp(torch.stack(positives).to(device))
            negative_batch = torch.cat(negatives).to(device)

            loss = contrastive_loss(anchor_batch, positive_batch, negative_batch, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count_batches += 1

        if count_batches > 0:
            avg_loss = total_loss / count_batches
        else:
            avg_loss = float('inf')  

        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.20f} | Best Loss: {best_loss:.30f} (Epoch {best_epoch})")


        if 0:
            print(f"Epoch {epoch+1} loss is 0, skipping model update.")
            epochs_no_improve += 1
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = shared_mlp.state_dict()
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    if best_state is not None:
        shared_mlp.load_state_dict(best_state)

    shared_mlp.eval()
    mapped_glyph = {}
    with torch.no_grad():
        for char_id, vec in glyph_embed.items():
            mapped_vec = shared_mlp(vec.to(device)).cpu().tolist()
            mapped_glyph[char_id] = mapped_vec

    with open(output_glyph_json, 'w', encoding='utf8') as f:
        json.dump(mapped_glyph, f, ensure_ascii=False, indent=2)

    print("✅ Saved contrastive glyphic embeddings!")

if __name__ == "__main__":
    train(
        glyph_json="model/glyph_vae_features.json",
        pos_file="dataset/constractive/positive_pairs.txt",
        neg_file="dataset/constractive/negative_pairs.txt",
        output_glyph_json="model/glyph_constractive.json",
        dim=512,
        epochs=100,
        batch_size=64,
        lr=1e-5,
        patience=10
    )
