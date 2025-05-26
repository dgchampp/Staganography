#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:18:25 2025

@author: rio
"""
# StampOne GAN-style Encoder + Discriminator Setup (U-Net with skip connections)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
import lpips
import pywt
import numpy as np

# 1. Load image as tensor
def load_image_tensor(image_path, size=(256, 256)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return tensor, image

# 2. Prepare binary message tensor
def prepare_binary_message(batch_size=1, bit_len=256):
    message = torch.randint(0, 2, (batch_size, bit_len)).float()
    message = message.view(batch_size, 1, 16, 16)  # grayscale
    message_rgb = message.repeat(1, 3, 1, 1)       # fake RGB
    return message_rgb, message

# 3. Sobel layer
class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)

        self.sobel_x = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)
        self.sobel_y = nn.Conv2d(3, 3, 3, padding=1, bias=False, groups=3)

        self.sobel_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

# 4. Wavelet transform with resizing
class WaveletTransform(nn.Module):
    def __init__(self):
        super(WaveletTransform, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        out = []
        for b in range(B):
            bands = []
            for c in range(C):
                channel_np = x[b, c].cpu().numpy()
                IG = channel_np
                LL, (LH, HL, HH) = pywt.dwt2(channel_np, 'haar')

                IG = self._resize(IG, H, W)
                LL = self._resize(LL, H, W)
                LH = self._resize(LH, H, W)
                HL = self._resize(HL, H, W)
                HH = self._resize(HH, H, W)

                bands.extend([IG, LL, LH, HL, HH])
            stacked = torch.tensor(np.stack(bands), dtype=torch.float32)
            out.append(stacked)
        return torch.stack(out).to(x.device)

    def _resize(self, arr, H, W):
        arr_t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(arr_t, size=(H, W), mode='bilinear', align_corners=False)
        return resized.squeeze()

# 3. Depthwise convolution for 15 channels
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels=15):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels
        )

    def forward(self, x):
        return self.depthwise(x)

# 4. MPN Network (Figure 3-style)
class MessagePreparationNetwork(nn.Module):
    def __init__(self, in_channels=15, out_channels=32):
        super(MessagePreparationNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

    # -----------------------------
# Generator (Encoder) - U-Net with skip connections
# -----------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(UNetEncoder, self).__init__()

        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + 128 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)          # 64
        x2 = self.enc2(self.pool(x1))  # 128
        x3 = self.enc3(self.pool(x2))  # 256

        x2_up = self.up2(x2)       # Upsample to match x1
        x3_up = self.up3(x3)       # Upsample to match x1

        x_cat = torch.cat([x1, x2_up, x3_up], dim=1)
        return self.final_conv(x_cat)

# -----------------------------
# Discriminator - Patch-style
# -----------------------------
class StegoDiscriminator(nn.Module):
    def __init__(self):
        super(StegoDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1).mean(dim=1)

# -----------------------------
# Real-data based training loop with LPIPS and Color Histogram Loss
# -----------------------------

def color_histogram_loss(img1, img2, bins=64):
    B, C, H, W = img1.shape
    loss = 0
    for b in range(B):
        for c in range(C):
            h1 = torch.histc(img1[b, c], bins=bins, min=0.0, max=1.0)
            h2 = torch.histc(img2[b, c], bins=bins, min=0.0, max=1.0)
            h1 = h1 / (h1.sum() + 1e-6)
            h2 = h2 / (h2.sum() + 1e-6)
            loss += F.mse_loss(h1, h2)
    return loss / (B * C)

# ... [keep your earlier functions: load_image_tensor, SobelLayer, WaveletTransform, etc.] ...

# Combined Generator: MPN + U-Net Encoder
class GeneratorWithMPN(nn.Module):
    def __init__(self):
        super(GeneratorWithMPN, self).__init__()
        self.gradient = SobelLayer()
        self.wavelet = WaveletTransform()
        self.depthwise = DepthwiseConv()
        self.mpn = MessagePreparationNetwork()
        self.image_proj = nn.Conv2d(15, 32, kernel_size=1)
        self.encoder = UNetEncoder()

    def forward(self, cover_img, binary_message):
        # Cover image path: [gradient → wavelet → depthwise → proj]
        grad_img = self.gradient(cover_img)
        wave_img = self.wavelet(grad_img)
        depth_img = self.depthwise(wave_img)
        image_feat = self.image_proj(depth_img)  # (B, 32, H, W)

        # Message path: [reshape → fake RGB → gradient → wavelet → depthwise → MPN]
        B = binary_message.shape[0]
        message = binary_message.view(B, 1, 16, 16).float()
        message_rgb = message.repeat(1, 3, 1, 1)
        grad_msg = self.gradient(message_rgb)
        wave_msg = self.wavelet(grad_msg)
        depth_msg = self.depthwise(wave_msg)
        msg_feat = self.mpn(depth_msg)  # (B, 32, 16, 16)
        msg_feat_up = F.interpolate(msg_feat, size=image_feat.shape[2:], mode='bilinear', align_corners=False)

        # Fuse and encode
        fused_input = torch.cat([image_feat, msg_feat_up], dim=1)
        encoded_img = self.encoder(fused_input)  # Output: (B, 3, H, W)
        return encoded_img

# Discriminator remains the same
class StegoDiscriminator(nn.Module):
    def __init__(self):
        super(StegoDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1).mean(dim=1)

# ... other classes and imports remain unchanged ...

loss_history = {"epoch": [], "G_loss": [], "D_loss": [], "LPIPS": [], "Color": [], "L1": []}

# Save loss plots
def plot_losses():
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history["epoch"], loss_history["G_loss"], label="G Loss")
    plt.plot(loss_history["epoch"], loss_history["D_loss"], label="D Loss")
    plt.legend()
    plt.title("Generator vs Discriminator Loss")

    plt.subplot(1, 3, 2)
    plt.plot(loss_history["epoch"], loss_history["LPIPS"], label="LPIPS")
    plt.plot(loss_history["epoch"], loss_history["Color"], label="Color Histogram")
    plt.legend()
    plt.title("LPIPS and Color Loss")

    plt.subplot(1, 3, 3)
    plt.plot(loss_history["epoch"], loss_history["L1"], label="L1 Loss")
    plt.legend()
    plt.title("L1 Loss")

    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.close()

# StampOne GAN-style Encoder + Discriminator Setup (U-Net with skip connections)

import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import lpips
from torch.utils.data import DataLoader, TensorDataset

# Updated training loop to handle multiple input images and save all encoded outputs

def real_training_loop_on_directory(image_dir, num_epochs=100, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = GeneratorWithMPN().to(device)
    D = StegoDiscriminator().to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    λColor, λP, λL1 = 1.0, 2.0, 0.1
    encoded_results = []
    


    # Load all images and generate messages
    image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
    images, messages, names = [], [], []
    for img_path in image_paths:
        tensor, _ = load_image_tensor(img_path)
        msg = torch.randint(0, 2, (256,), dtype=torch.float32)
        images.append(tensor.squeeze(0))
        messages.append(msg)
        names.append(os.path.basename(img_path))

    dataset = list(zip(images, messages, names))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    checkpoint_path = 'encoder/stampone_checkpoint.pth'
    start_epoch = 1  # default

    if os.path.exists(checkpoint_path):
        print("🔁 Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
    
        G.load_state_dict(checkpoint['generator_state_dict'])
        D.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        for batch in dataloader:
            cover_batch, msg_batch, name_batch = batch
            cover_batch = cover_batch.to(device)
            msg_batch = msg_batch.to(device)

            G.train()
            D.train()

            msg_input = msg_batch.view(-1, 256)
            fake_img_G = G(cover_batch, msg_input)
            fake_img_D = fake_img_G.detach()

            # Discriminator step
            D.zero_grad()
            d_real = D(cover_batch)
            d_fake = D(fake_img_D)
            d_loss = d_fake.mean() - d_real.mean()
            d_loss.backward()
            d_optimizer.step()

            # Generator step
            G.zero_grad()
            g_fake = D(fake_img_G)
            adv_loss = -g_fake.mean()
            lpips_loss = lpips_fn(fake_img_G * 2 - 1, cover_batch * 2 - 1).mean()
            color_loss = color_histogram_loss(fake_img_G, cover_batch)
            l1_loss = F.l1_loss(fake_img_G, cover_batch)

            g_loss = adv_loss + λP * lpips_loss + λColor * color_loss + λL1 * l1_loss
            g_loss.backward()
            g_optimizer.step()

            print(f"Epoch {epoch:03} | Batch size {len(cover_batch)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | LPIPS: {lpips_loss.item():.4f} | ColorHist: {color_loss.item():.4f} | L1: {l1_loss.item():.4f}")

            if epoch == num_epochs:
                for i, name in enumerate(name_batch):
                    encoded_results.append((name, fake_img_G[i].detach().cpu()))

    # Save all final encoded images after last epoch
    os.makedirs("/home/rio/encoder/final_encoded_outputs", exist_ok=True)
    for name, tensor in encoded_results:
        img = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
        plt.imsave(f"/home/rio/encoder/final_encoded_outputs/encoded_{name}", img)
        
        # Save model checkpoints after training
    torch.save(G.state_dict(), '/home/rio/encoder/generator_stampone.pth')
    torch.save(D.state_dict(), '/home/rio/encoder/discriminator_stampone.pth')
    torch.save({
    'epoch': num_epochs,
    'generator_state_dict': G.state_dict(),
    'discriminator_state_dict': D.state_dict(),
    'g_optimizer_state_dict': g_optimizer.state_dict(),
    'd_optimizer_state_dict': d_optimizer.state_dict(),
}, '/home/rio/encoder/stampone_checkpoint.pth')

    print("✅ Models saved successfully.")


# Example usage:
real_training_loop_on_directory("encoder/data/m1", num_epochs=10000, batch_size=4)
