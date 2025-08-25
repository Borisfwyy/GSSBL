import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GlyphVAE(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, feature_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, feature_dim)
        self.fc_decode = nn.Sequential(
            nn.Linear(feature_dim, 512 * 8 * 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.fc_decode(z).view(-1, 512, 8, 8)
        out = self.decoder(x_rec)
        return z, out, mu, logvar


def vae_loss(recon, x, mu, logvar, l1_weight=1.0, kl_weight=1.0):
    recon_loss = nn.functional.l1_loss(recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return l1_weight * recon_loss + kl_weight * kl


def kl_annealing(epoch, total_epochs, max_kl=0.01):
    return max_kl * epoch / total_epochs


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class GlyphDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.RandomRotation(5),
            transforms.ToTensor()
        ])
        self.image_paths = image_paths

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        return self.transform(img), self.image_paths[idx]


def collect_image_paths(root_dir, exts={".png", ".jpg", ".jpeg", ".bmp", ".tiff"}):
    return [os.path.join(dp, f)
            for dp, _, fnames in os.walk(root_dir)
            for f in fnames if os.path.splitext(f)[1].lower() in exts]


def train_vae(image_paths, model_path, epochs=10, batch_size=64, num_workers=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GlyphDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = GlyphVAE()
    model.apply(weights_init)  # Xavier初始化
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 降低学习率

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        kl_weight = kl_annealing(epoch, epochs)  # 动态KL权重
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            _, recons, mu, logvar = model(imgs)
            loss = vae_loss(recons, imgs, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} 平均损失: {avg_loss:.6f} | KL权重: {kl_weight:.6f}")

        if epoch <= 3 or epoch % 5 == 0:
            os.makedirs("debug_output", exist_ok=True)
            save_image(imgs[:4].cpu(), f"debug_output/input_epoch{epoch}.png")
            save_image(recons[:4].cpu(), f"debug_output/output_epoch{epoch}.png")
    model_save_path = model_path or "model/images/glyph_vae_autoencoder.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")
    return model


def extract_and_save_features(model, image_paths, feature_dir, decode_dir, batch_size=64, num_workers=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = GlyphDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(decode_dir, exist_ok=True)
    feature_dict = {}
    common_root = os.path.commonpath(image_paths)

    with torch.no_grad():
        for imgs, paths in tqdm(dataloader, desc="提取特征"):
            imgs = imgs.to(device)
            zs, recons, _, _ = model(imgs)
            for i, path in enumerate(paths):
                name = os.path.splitext(os.path.basename(path))[0]
                feature_dict[name] = zs[i].cpu().tolist()

                save_path = os.path.join(decode_dir, os.path.relpath(path, start=common_root))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image(recons[i].cpu(), save_path)

    with open(os.path.join(feature_dir, "glyph_vae_features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_dict, f, ensure_ascii=False, indent=2)

    print(f"共保存 {len(feature_dict)} 个字形特征至 {feature_dir}/glyph_vae_features.json")


def run_pipeline(
    image_root_folder,
    feature_root="features",
    decoded_root="decoded_images",
    model_weights_path="glyph_vae.pt",
    epochs=10,
    batch_size=64,
    num_workers=8,
    device=None
):
    seed_everything(42)
    torch.cuda.empty_cache()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = collect_image_paths(image_root_folder)
    print(f"共加载 {len(image_paths)} 张图像")

    if model_weights_path and os.path.isfile(model_weights_path):
        print(f"加载已有模型权重 {model_weights_path}")
        model = GlyphVAE()
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.to(device)
    else:
        print(f"未检测到权重，开始训练")
        model = train_vae(image_paths, model_weights_path, epochs, batch_size, num_workers, device)

    extract_and_save_features(model, image_paths, feature_root, decoded_root, batch_size, num_workers, device)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_pipeline(
        image_root_folder="images",
        feature_root="features",
        decoded_root="decoded_images",
        model_weights_path="glyph_vae.pt",
        epochs=10,
        batch_size=64,
        num_workers=8,
        device=None
    )
