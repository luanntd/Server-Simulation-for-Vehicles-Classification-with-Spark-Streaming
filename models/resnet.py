import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyspark.sql.dataframe import DataFrame
from torchvision import models, transforms
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from joblibspark import register_spark

register_spark()

class ResnetClassifier:
    def __init__(self, num_classes=4, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def _prepare_batch(self, df: DataFrame):
        X = np.array(df.select("image").collect()).reshape(-1, 32, 32, 3)
        y = np.array(df.select("label").collect()).reshape(-1)

        X = np.transpose(X, (0, 3, 1, 2)) / 255.0
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        return X_tensor, y_tensor

    def train(self, df: DataFrame):
        self.model.train()
        X_tensor, y_tensor = self._prepare_batch(df)

        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()

        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        y_true = y_tensor.cpu().numpy()

        accuracy = np.mean(preds == y_true)
        precision = precision_score(y_true, preds, average="macro", zero_division=0)
        recall = recall_score(y_true, preds, average="macro", zero_division=0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return preds, accuracy, precision, recall, f1

    def predict(self, df: DataFrame):
        self.model.eval()
        X_tensor, y_tensor = self._prepare_batch(df)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)

        preds = preds.cpu().numpy()
        y_true = y_tensor.cpu().numpy()

        accuracy = np.mean(preds == y_true)
        precision = precision_score(y_true, preds, average="macro", zero_division=0)
        recall = recall_score(y_true, preds, average="macro", zero_division=0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        cm = confusion_matrix(y_true, preds)

        return preds, accuracy, precision, recall, f1, cm

    def save(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Model checkpoint saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        print(f"[INFO] Model checkpoint loaded from {path}")
