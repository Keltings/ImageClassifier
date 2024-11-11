# model_utils.py

import torch
from torchvision import models
from PIL import Image
import numpy as np
import os

def save_checkpoint(checkpoint, save_dir):
    """Save the model checkpoint to the specified directory."""
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_model(checkpoint_path, gpu=False):
    """Load a model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model

def process_image(image_path):
    """Process an image for use with the model."""
    image = Image.open(image_path)
    image = image.resize((256, 256))
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image).float()
