import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from tqdm import tqdm  # Import tqdm

# Configuration parameters
batch_size = 256
epochs = 200 # Increased for early stopping
lr = 3e-4
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
log_interval = 100  # Log every 100 steps

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = "/data/"

# Custom dataset class for loading data from CSV
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])  # Assuming image path is in the first column
        image = Image.open(img_name).convert('RGB')
        label = self.data_frame.iloc[idx, 1]  # Assuming label is in the second column

        if self.transform:
            image = self.transform(image)

        return image, label

# Load data
train_dataset = CustomImageDataset(csv_file=os.path.join(data_dir, 'TRAIN.csv'), root_dir=data_dir, transform=transform)
val_dataset = CustomImageDataset(csv_file=os.path.join(data_dir, 'VAL.csv'), root_dir=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = timm.create_model('gcvit_tiny', pretrained=True, num_classes=2)
# model = nn.DataParallel(model)  # Use DataParallel to utilize multiple GPUs
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)  # Reduce LR on validation accuracy plateau

# Early stopping parameters
early_stopping_patience = 20
best_accuracy = 0
epochs_without_improvement = 0

# Training and validation functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    # Wrap train_loader with tqdm
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            progress_bar.set_postfix(loss=loss.item(), accuracy=100. * correct / ((batch_idx + 1) * train_loader.batch_size))

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)')
    return accuracy

# Main training and validation loop
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    accuracy = validate(model, device, val_loader)
    scheduler.step(accuracy)  # Step the scheduler with validation accuracy

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), "ckpt_\GCViT_xtiny_Model.pth")
        best_accuracy = accuracy
        epochs_without_improvement = 0
        print(f'Saved better model with accuracy: {accuracy:.2f}%')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered")
            break

print("Training complete")
