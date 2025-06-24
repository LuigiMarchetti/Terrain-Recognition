import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from typing import Tuple, Dict
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn.functional as F

# Custom Dataset for EuroSAT
class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# Fine-tune GoogleNet
def fine_tune_googlenet(model, train_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

    return model

# Preprocess image for inference
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = transform(image)
    assert isinstance(processed_image, torch.Tensor)
    image_tensor = processed_image.unsqueeze(0)
    return image_tensor

# Classify deforestation with fine-tuned model
def classify_deforestation(image: np.ndarray, model: torch.nn.Module, device: torch.device, eurosat_classes) -> Tuple[np.ndarray, Dict]:
    image_tensor = preprocess_image(image).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # EuroSAT classes: 0=AnnualCrop, 1=Forest, 2=HerbaceousVegetation, 3=Highway, 4=Industrial,
    # 5=Pasture, 6=PermanentCrop, 7=Residential, 8=River, 9=SeaLake
    forest_class = 1  # Forest
    non_forest_classes = [0, 2, 3, 4, 5, 6, 7, 8, 9]  # All other classes as potential deforested

    forest_score = probs[forest_class]
    deforestation_score = sum(probs[i] for i in non_forest_classes)

    original_shape = image.shape[:2]
    classification = np.zeros(original_shape, dtype=np.uint8)
    threshold = 0.1  # Adjusted threshold for EuroSAT
    if deforestation_score > forest_score and deforestation_score > threshold:
        classification[:, :] = 1

    total_pixels = classification.size
    deforested_pixels = np.sum(classification == 1)
    forest_pixels = np.sum(classification == 0)
    stats = {
        'Deforested': {'pixels': int(deforested_pixels), 'percentage': (deforested_pixels / total_pixels) * 100},
        'Forest': {'pixels': int(forest_pixels), 'percentage': (forest_pixels / total_pixels) * 100}
    }

    # Save analysis
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_indices]
    top5_labels = [eurosat_classes[i] for i in top5_indices]
    analysis_text = f"GoogleNet Analysis for {image.shape}:\n"
    analysis_text += f"Top 5 Predictions: {dict(zip(top5_labels, [f'{p:.3f}' for p in top5_probs]))}\n"
    analysis_text += f"Forest Score: {forest_score:.3f}\n"
    analysis_text += f"Deforestation Score: {deforestation_score:.3f}\n"
    analysis_text += f"Threshold Used: {threshold:.3f}\n"
    analysis_text += f"Classification: {'Deforested' if deforestation_score > forest_score and deforestation_score > threshold else 'Forest'}\n"
    with open('googlenet_analysis.txt', 'a') as f:
        f.write(analysis_text + "\n")

    return classification, stats

def create_comparison_visualization(
        before_image, before_classification, before_stats,
        after_image, after_classification, after_stats,
        coordinates, output_path
):
    fig = plt.figure(figsize=(15, 10))
    cmap = ListedColormap(['#228B22', '#8B4513'])

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(before_image)
    ax1.set_title(f'Before RGB\n2022-06-13\nLat: {coordinates[0]:.6f}, Lon: {coordinates[1]:.6f}', fontsize=10)
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(before_classification, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title('Before Classification\n2022-06-13', fontsize=10)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, ticks=[0, 1], label='Class (0=Forest, 1=Deforested)')

    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    table_data = [[k, f"{v['percentage']:.2f}%", f"{v['pixels']:,}"] for k, v in before_stats.items()]
    table = ax3.table(cellText=table_data, colLabels=['Class', '%', 'Pixels'], cellLoc='center', loc='center')
    table.auto_set_fontsize(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax3.set_title('Before Statistics', fontsize=10)

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(after_image)
    ax4.set_title(f'After RGB\n2025-03-09', fontsize=10)
    ax4.axis('off')

    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(after_classification, cmap=cmap, vmin=0, vmax=1)
    ax5.set_title('After Classification\n2025-03-09', fontsize=10)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, ticks=[0, 1], label='Class (0=Forest, 1=Deforested)')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    deforestation_increase = after_stats['Deforested']['percentage'] - before_stats['Deforested']['percentage']
    forest_loss = before_stats['Forest']['percentage'] - after_stats['Forest']['percentage']
    risk_level = 'High' if deforestation_increase > 10 else 'Medium' if deforestation_increase > 5 else 'Low'
    change_text = f"""
Deforestation Change Summary
Coordinates: ({coordinates[0]:.6f}, {coordinates[1]:.6f})
Period: 2022-06-13 to 2025-03-09

Deforestation Change:
• Before: {before_stats['Deforested']['percentage']:.2f}%
• After: {after_stats['Deforested']['percentage']:.2f}%
• Increase: {deforestation_increase:.2f}%

Forest Loss: {forest_loss:.2f}%

Risk Level: {risk_level}
Recommendation: {'Immediate action' if risk_level == 'High' else 'Enhanced monitoring' if risk_level == 'Medium' else 'Continue monitoring'}
"""
    ax6.text(0.05, 0.95, change_text, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax6.set_title('Change Summary', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def compare_deforestation(
        before_image_path, after_image_path, coordinates,
        output_dir="deforestation_comparison", model_path="fine_tuned_googlenet.pth"
):
    os.makedirs(output_dir, exist_ok=True)

    before_image = cv2.imread(before_image_path, cv2.IMREAD_COLOR)
    after_image = cv2.imread(after_image_path, cv2.IMREAD_COLOR)

    if before_image is None or after_image is None:
        raise ValueError("Failed to load one or both images.")

    before_image = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
    after_image = cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 10 EuroSAT classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    eurosat_classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                       'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    before_classification, before_stats = classify_deforestation(before_image, model, device, eurosat_classes)
    after_classification, after_stats = classify_deforestation(after_image, model, device, eurosat_classes)

    deforestation_change = {
        'before_deforested_pct': before_stats['Deforested']['percentage'],
        'after_deforested_pct': after_stats['Deforested']['percentage'],
        'deforestation_increase': after_stats['Deforested']['percentage'] - before_stats['Deforested']['percentage'],
        'before_forest_pct': before_stats['Forest']['percentage'],
        'after_forest_pct': after_stats['Forest']['percentage'],
        'forest_loss': before_stats['Forest']['percentage'] - after_stats['Forest']['percentage']
    }

    output_path = os.path.join(output_dir, f"deforestation_comparison_{coordinates[0]:.6f}_{coordinates[1]:.6f}.png")
    create_comparison_visualization(
        before_image, before_classification, before_stats,
        after_image, after_classification, after_stats,
        coordinates, output_path
    )

    summary_file = os.path.join(output_dir, "comparison_summary.json")
    summary_stats = {
        'before_image': before_image_path,
        'after_image': after_image_path,
        'coordinates': coordinates,
        'before_stats': before_stats,
        'after_stats': after_stats,
        'deforestation_change': deforestation_change,
        'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Comparison image saved: {output_path}")
    print(f"Summary statistics saved: {summary_file}")
    print(f"Deforestation Increase: {deforestation_change['deforestation_increase']:.2f}%")
    print(f"Forest Loss: {deforestation_change['forest_loss']:.2f}%")

    return deforestation_change

if __name__ == "__main__":
    # Step 1: Fine-tune the model (run this once)
    base_dir = "C:\\Users\\luigi\\Downloads\\EUROSAT"
    eurosat_dir = os.path.join(base_dir, "EuroSAT_RGB")
    if os.path.exists(eurosat_dir):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = EuroSATDataset(eurosat_dir, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # 10 EuroSAT classes
        model = fine_tune_googlenet(model, train_loader, num_epochs=10, device=device)
        torch.save(model.state_dict(), "fine_tuned_googlenet.pth")
        print("Fine-tuning complete. Model saved as fine_tuned_googlenet.pth")
    else:
        print("EuroSAT dataset not found at", eurosat_dir)

    # Step 2: Compare images with fine-tuned model
    before_image_path = os.path.join(base_dir, "CNN", "after", "rgb_1_2022-06-13.png")
    after_image_path = os.path.join(base_dir, "CNN", "before", "rgb_1_2025-03-09.png")
    coordinates = (-3.4653, -62.2159)

    if not (os.path.exists(before_image_path) and os.path.exists(after_image_path)):
        print("One or both image files not found. Please check the paths:")
        print(f"Before: {before_image_path}")
        print(f"After: {after_image_path}")
    else:
        print("Starting deforestation comparison with fine-tuned model...")
        results = compare_deforestation(before_image_path, after_image_path, coordinates, model_path="fine_tuned_googlenet.pth")