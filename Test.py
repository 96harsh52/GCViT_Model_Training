import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def load_model(model_path, device):
    model = timm.create_model('gcvit_tiny', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def predict_single_image(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        probability = softmax(output.cpu().numpy())[0]  # Get softmax result and select first item
        return probability

def main():
    csv_file = "/data/VAL_DATA.csv"
    df = pd.read_csv(csv_file)

    model_path = "ckpt_\GCViT_xtiny_Model.pth"
    device = "cuda:3"
    model = load_model(model_path, device)

    predictions = []

    image_paths = df['Image_Path'].tolist()

    # Process images sequentially
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image_tensor = preprocess_image(image_path)
            probability = predict_single_image(model, image_tensor, device)
            predictions.append((probability[0], probability[1]))  # (Pred_Spoof, Pred_Live)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            predictions.append((None, None))  # Handle failed predictions

    # Convert predictions to DataFrame
    df_predictions = pd.DataFrame(predictions, columns=['Pred_Spoof', 'Pred_Live'])

    # Combine predictions with the original dataframe
    df_combined = pd.concat([df, df_predictions], axis=1)

    output_dir = "/data/WORKSPACE_HS/GCViT/Train_code_2209"
    output_csv_file_with_length = os.path.join(output_dir, f"551_Pred_GCViT_3009_old_code_{len(df_combined)}.csv")
    df_combined.to_csv(output_csv_file_with_length, index=False)

if __name__ == "__main__":
    main()
