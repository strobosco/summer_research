"""
Current state:

Missing:
  The pipeline to extract FLAME parameters from RGB video
  The training data pairs of (frame, FLAME parameters)

TODO:
- modify transform and ResNet to handle 320x240 instead of 224x224 images

"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ResNet-18 Feature Extractor
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for grayscale input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
        # self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification head
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-2],  # Remove avgpool & FC layer
            nn.AdaptiveAvgPool2d((1, 1))  # Dynamically pool features
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # Output shape: (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        return x

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model for Predicting FLAME Parameters
class TransformerFLAMEModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, output_dim=156, seq_length=5):
        super(TransformerFLAMEModel, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Training Function
def train_model(model, dataloader, num_epochs=10, learning_rate=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for frames, flame_params in dataloader:
            frames, flame_params = frames.to(device), flame_params.to(device)
            optimizer.zero_grad()
            predictions = model(frames)
            loss = criterion(predictions, flame_params)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    print("Training complete!")

# Validation Function
def validate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for frames, flame_params in dataloader:
            frames, flame_params = frames.to(device), flame_params.to(device)
            predictions = model(frames)
            loss = criterion(predictions, flame_params)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(dataloader)}")

# Testing Function
def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    results = []
    with torch.no_grad():
        for frames, _ in dataloader:
            frames = frames.to(device)
            predictions = model(frames)
            results.append(predictions.cpu().numpy())
    return results

# Real-Time Prediction Function
def predict_real_time(model, frame):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    frame = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(frame)
    return prediction.cpu().numpy()
