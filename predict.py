# predict.py

import torch
from model_utils import load_model, process_image  # Assuming you put the load_model and process_image in model_utils.py
from PIL import Image
import json
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower type from an image")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('checkpoint', type=str, help="Path to the saved model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top K predictions to return")
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="JSON file mapping category to real names")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")
    return parser.parse_args()

def predict_image(image_path, model, topk=5):
    image = process_image(image_path).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    probs = torch.exp(output)
    top_probs, top_classes = probs.topk(topk)
    return top_probs.cpu().numpy().flatten(), top_classes.cpu().numpy().flatten()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, gpu=args.gpu)
    top_probs, top_classes = predict_image(args.image_path, model, topk=args.top_k)

    # Load category names if available
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        class_names = [cat_to_name[str(c)] for c in top_classes]
    else:
        class_names = [str(c) for c in top_classes]
    
    print("Top K Predictions:")
    for i in range(len(top_probs)):
        print(f"Class: {class_names[i]}, Probability: {top_probs[i]:.4f}")
