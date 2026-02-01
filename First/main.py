# =========================
# Install Dependencies
# =========================
# NOTE: run manually in terminal or Colab, NOT required in repo
# pip install diffusers transformers accelerate safetensors torch torchvision seaborn scikit-learn

# =========================
# Imports
# =========================
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from diffusers import StableDiffusionPipeline

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# Dog Breeds
# =========================
DOG_BREEDS = [
    "Labrador Retriever","Golden Retriever","German Shepherd","French Bulldog",
    "Bulldog","Poodle (Standard)","Beagle","Rottweiler","Dachshund",
    "Yorkshire Terrier","Boxer","Doberman Pinscher","Siberian Husky",
    "Great Dane","Shih Tzu","Chihuahua","Pug","Border Collie",
    "Australian Shepherd","Belgian Malinois","Akita","Alaskan Malamute",
    "Samoyed","Bernese Mountain Dog","Saint Bernard","Newfoundland",
    "Cane Corso","Greyhound","Whippet","Bloodhound","Basset Hound",
    "Rhodesian Ridgeback","Weimaraner","Vizsla","Cocker Spaniel",
    "Cavalier King Charles Spaniel","Pomeranian","Chow Chow",
    "Shiba Inu","Basenji"
]

# =========================
# Stable Diffusion Loader
# =========================
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=True  # token picked from env variable
    )

    pipe = pipe.to(device)
    return pipe

# =========================
# Generate Dataset
# =========================
def generate_dataset(pipe, breeds, output_dir, images_per_class=3):
    os.makedirs(output_dir, exist_ok=True)

    for breed in breeds:
        class_dir = os.path.join(output_dir, breed.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)

        for i in range(images_per_class):
            prompt = f"a high quality photo of a {breed}, ultra realistic, cinematic lighting, 4k"

            image = pipe(
                prompt,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]

            path = os.path.join(class_dir, f"{breed.replace(' ', '_')}_{i}.png")
            image.save(path)
            print("Saved:", path)

# =========================
# Dataloaders
# =========================
def get_loaders(data_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return dataset, train_loader, val_loader

# =========================
# Model
# =========================
def build_model(num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# =========================
# Train & Evaluate
# =========================
def train(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        acc = evaluate(model, val_loader)
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(train_loader):.4f} "
            f"Val Acc: {acc:.2f}%"
        )

# =========================
# Evaluation
# =========================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# =========================
# Confusion Matrix
# =========================
def plot_confusion(model, loader, class_names):
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix – ResNet18")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

# =========================
# Run Pipeline
# =========================
DATASET_DIR = "40_dog_dataset"

pipe = load_pipeline()
generate_dataset(pipe, DOG_BREEDS, DATASET_DIR, images_per_class=3)

dataset, train_loader, val_loader = get_loaders(DATASET_DIR)

model = build_model(len(dataset.classes))
train(model, train_loader, val_loader, epochs=5)

torch.save(model.state_dict(), "resnet18_40_dog_breeds.pth")
plot_confusion(model, val_loader, dataset.classes)
