

pip install torch torchvision matplotlib numpy

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
import matplotlib.pyplot as plt
import numpy as np

# ================= USER INPUTS =================
dataset_choice = input("Enter dataset (mnist/fashion): ").lower()
epochs = int(input("Enter epochs (30–100): "))
batch_size = int(input("Enter batch size (64 or 128): "))
noise_dim = int(input("Enter noise dimension (50 or 100): "))
learning_rate = float(input("Enter learning rate (e.g., 0.0002): "))
save_interval = int(input("Save samples every k epochs: "))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("generated_samples", exist_ok=True)
os.makedirs("final_generated_images", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_choice == "mnist":
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
elif dataset_choice == "fashion":
    dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
else:
    raise ValueError("Invalid dataset choice")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch = real_imgs.size(0)

        real_labels = torch.ones(batch, 1).to(device)
        fake_labels = torch.zeros(batch, 1).to(device)

        # ========== Train Discriminator ==========
        noise = torch.randn(batch, noise_dim).to(device)
        fake_imgs = G(noise)

        D_real = D(real_imgs)
        D_fake = D(fake_imgs.detach())

        D_loss = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        D_acc = ((D_real > 0.5).sum() + (D_fake < 0.5).sum()).item() / (2 * batch) * 100

        # ========== Train Generator ==========
        G_fake = D(fake_imgs)
        G_loss = criterion(G_fake, real_labels)

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}/{epochs} | D_loss: {D_loss:.2f} | D_acc: {D_acc:.2f}% | G_loss: {G_loss:.2f}")

    # Save images
    if epoch % save_interval == 0:
        utils.save_image(fake_imgs[:25], f"generated_samples/epoch_{epoch:02d}.png",
                         nrow=5, normalize=True)

noise = torch.randn(100, noise_dim).to(device)
final_images = G(noise)

for i in range(100):
    utils.save_image(final_images[i], f"final_generated_images/img_{i}.png", normalize=True)

classifier = models.resnet18(pretrained=True)
classifier.fc = nn.Linear(classifier.fc.in_features, 10)
classifier = classifier.to(device)
classifier.eval()

predicted_labels = []

with torch.no_grad():
    for img in final_images:
        img = img.repeat(3, 1, 1).unsqueeze(0)  # convert 1→3 channel
        img = nn.functional.interpolate(img, size=(224, 224))
        output = classifier(img)
        predicted_labels.append(torch.argmax(output).item())

# Label distribution
from collections import Counter
print("Label Distribution:", Counter(predicted_labels))