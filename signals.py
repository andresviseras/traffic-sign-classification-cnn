#
# PROJECT: SIGNAL CLASSIFICATION SYSTEM
# Dataset: GTSRB (German Traffic Sign Recognition Benchmark) - 43 Clases
#

# 1. Configuration and Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Class Names for better visuzalization
CLASS_NAMES = [
    "Speed limit 20",  "Speed limit 30",   "Speed limit 50",  "Speed limit 60", "Speed limit 70",
    "Speed limit 80",  "End speed limit 80", "Speed limit 100", "Speed limit 120", "No overtaking",
    "No overtaking >3.5t", "Priority road", "Main road", "Give way", "Stop",
    "No entry for vehicles", "No entry for vehicles >3.5t", "No entry", "General danger", "Dangerous curve left",
    "Dangerous curve right", "Double curve", "Uneven road", "Slippery road", "Road narrows on right",
    "Roadworks", "Traffic signals", "Pedestrians", "Children", "Cyclists",
    "Ice/Snow", "Wild animals", "End of all restrictions", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Ahead or right only", "Ahead or left only", "Keep right", "Keep left",
    "Roundabout", "End of no overtaking", "End of no overtaking >3.5t"
    ]

# 3. Data Loading and Augmentation
image_size = 32
batch_size = 64

transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Downloading GTSRB dataset...")
trainset = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=transform_train)
valset = torchvision.datasets.GTSRB(root='./data', split='test',  download=True, transform=transform_val)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader = torch.utils.data.DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = 43
print(f"Training images: {len(trainset)}")
print(f"Validation images: {len(valset)}")

# 4. Visualization of samples per class, for only the 10 first classes
def show_random_samples(loader, title="Dataset samples"):
    data_iter  = iter(loader)
    images, labels = next(data_iter)
    num_show   = min(10, len(images))

    fig, axes = plt.subplots(1, num_show, figsize=(16, 3))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    for i, ax in enumerate(axes):
        img = images[i] / 2 + 0.5
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)
        ax.imshow(img)
        ax.set_title(CLASS_NAMES[labels[i].item()], fontsize=7)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# 5. CNN Architecture
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignCNN, self).__init__()

        # Block 1: 3x32x32 -> 32x16x16     LOW LEVEL
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32x16x16 -> 64x8x8   MID LEVEL
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64x8x8 -> 128x4x4    HIGH LEVEL
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 canales * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model     = TrafficSignCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 6. Training with Early Stopping
def train_model(model, epochs=20):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 5

    for epoch in range(epochs):
        # TRAINING PHASE
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            _, pred = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (pred == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # VALIDATION PHASE
        model.eval()
        val_loss     = 0.0
        correct_val  = 0
        total_val    = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, pred   = torch.max(outputs, 1)
                total_val   += labels.size(0)
                correct_val += (pred == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = 100 * correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.1f}%")

        # Save best model and implement early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_traffic_model.pth')
            print(f"   ✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly Stopping in epoch {epoch+1}. No improvement in {PATIENCE} epochs.")
                break

    return train_losses, val_losses, train_accs, val_accs

# 7. Learning Curves Visualization
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Curvas de Aprendizaje - Detección de Señales", fontweight='bold')

    axes[0].plot(train_losses, label='Train', color='blue')
    axes[0].plot(val_losses,   label='Val',   color='red', linestyle='--')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label='Train', color='blue')
    axes[1].plot(val_accs,   label='Val',   color='red', linestyle='--')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# 8. Visual Predictions with green and red
def show_predictions(model, val_loader, num_samples=8):
    model.eval()
    data_iter = iter(val_loader)
    images, labels = next(data_iter)

    fig, axes = plt.subplots(1, num_samples, figsize=(18, 3))
    fig.suptitle("Predictions for Validation Samples", fontweight='bold')

    for i, ax in enumerate(axes):
        img = images[i] / 2 + 0.5
        img = np.clip(img.permute(1, 2, 0).numpy(), 0, 1)

        with torch.no_grad():
            output = model(images[i:i+1].to(device))
            pred_idx = torch.argmax(output, 1).item()

        real_name = CLASS_NAMES[labels[i].item()]
        pred_name = CLASS_NAMES[pred_idx]
        color = 'green' if pred_idx == labels[i].item() else 'red'

        ax.imshow(img)
        ax.set_title(f"Real: {real_name}\nPred: {pred_name}", color=color, fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 9. Final Evaluation with Confusion Matrix with the top 10 most frequent classes
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nGlobal Accuracy in Validation: {100 * correct / total:.2f}%")

    # 15x15 Matrix
    from collections import Counter
    top_classes  = [c for c, _ in Counter(all_labels).most_common(15)]
    mask         = [i for i, l in enumerate(all_labels) if l in top_classes]
    filtered_labels = [all_labels[i] for i in mask]
    filtered_preds  = [all_preds[i]  for i in mask]
    top_names    = [CLASS_NAMES[c] for c in sorted(top_classes)]

    cm = confusion_matrix(filtered_labels, filtered_preds, labels=sorted(top_classes))
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_names, yticklabels=top_names)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


# EXECUTION
if __name__ == '__main__':

    # 1. Show random samples from training set
    show_random_samples(train_loader, "Training samples with Augmentation")

    # 2. Training
    print("\nTRAINING STARTED...")
    train_losses, val_losses, train_accs, val_accs = train_model(model, epochs=20)

    model.load_state_dict(torch.load('best_traffic_model.pth'))
    print("\nBest model loaded.")

    # 3. Plot learning curves
    plot_curves(train_losses, val_losses, train_accs, val_accs)

    # 4. Show predictions on validation samples
    show_predictions(model, val_loader)

    # 5. Final Evaluation with Confusion Matrix
    evaluate_model(model, val_loader)