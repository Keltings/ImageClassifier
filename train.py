# train.py

import argparse
import torch
from torch import optim
from torchvision import datasets, models, transforms
from model_utils import save_checkpoint  # Assuming save_checkpoint is in model_utils.py
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a new network")
    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13', 'resnet18'],
                        help='Model architecture to use (default: vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units (default: 512)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint (default: current directory)')
    return parser.parse_args()

def train_model(args):
    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Prepare data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)

    # Choose model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, args.hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(args.hidden_units, 102),  # 102 output classes (flowers)
        torch.nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    model.to(device)
    
    # Set up the criterion and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train the model
    steps = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Print validation accuracy every 40 steps
            if steps % 40 == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        valid_loss += criterion(outputs, labels).item()
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}, "
                      f"Train Loss: {running_loss/40:.3f}, "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f}, "
                      f"Validation Accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': args.epochs,
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'learning_rate': args.learning_rate,
        'class_to_idx': train_dataset.class_to_idx,
        'classifier': model.classifier,
    }
    save_checkpoint(checkpoint, args.save_dir)

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
