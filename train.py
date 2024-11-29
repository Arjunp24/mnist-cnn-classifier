import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from model import SimpleCNN
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define transforms with augmentations and normalization (mean=0, std=1)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])

# Separate transform for saving augmented samples
augmentation_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
])

# Check if MNIST data directory exists
data_path = './data'
if not os.path.exists(data_path):
    print("Downloading MNIST dataset...")
    download = True
else:
    print("MNIST dataset already exists, skipping download...")
    download = False

# Load MNIST dataset
try:
    train_dataset = datasets.MNIST(data_path, train=True, download=download, transform=train_transform)
    batch_size = 16
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    print(f"Dataset loaded successfully with {len(train_dataset)} samples")
    print(f"Number of mini-batches: {len(train_loader)}")
    print(f"Using {num_workers} workers for data loading")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Initialize model, loss function and optimizer
model = SimpleCNN().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0034)

# Save augmented samples
if not os.path.exists('augmented_samples'):
    os.makedirs('augmented_samples')
else:
    print("Saving augmented samples...")
    
    # Get original dataset without any transforms
    original_dataset = datasets.MNIST(data_path, train=True, download=False)
    
    for i in range(5):
        # Get original image and convert to PIL Image
        orig_img = original_dataset.data[i].numpy()
        pil_img = transforms.ToPILImage()(orig_img)
        
        # Create augmented version
        torch.manual_seed(i)  # For reproducible augmentations
        aug_img = augmentation_transform(pil_img)
        
        # Create subplot with original and augmented
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot original
        ax1.imshow(orig_img, cmap='gray')
        ax1.set_title(f'Original - Label: {original_dataset.targets[i].item()}')
        ax1.axis('off')
        
        # Plot augmented
        ax2.imshow(aug_img.squeeze(), cmap='gray')
        ax2.set_title(f'Augmented - Label: {original_dataset.targets[i].item()}')
        ax2.axis('off')
        
        # Add spacing between subplots
        plt.tight_layout()
        
        # Save with higher DPI for better quality
        plt.savefig(f'augmented_samples/sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print("Augmented samples saved successfully!")

# Training loop
def train_one_epoch():
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        avg_loss = running_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'accuracy': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_acc, epoch_loss

print("Starting training...")
accuracy, loss = train_one_epoch()
print(f'Final Training Accuracy: {accuracy}%')
print(f'Final Training Loss: {loss:.4f}')

# Save the model
model_path = 'mnist_cnn.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_accuracy': accuracy,
    'final_loss': loss
}, model_path)
print(f"Model saved to {model_path}") 