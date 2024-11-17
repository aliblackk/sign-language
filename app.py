import streamlit as st
import torch
import gdown
import zipfile
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set parameters
img_size = (224, 224)
batch_size = 32
learning_rate = 0.001
epochs = 10
classes = [f"Class {i}" for i in range(26)]  # Replace with your class names

# Define data transforms
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def download_and_extract_data():
    # Hardcoded Google Drive link for data_split.zip
    drive_link = "https://drive.google.com/file/d/1EEPJeNoL_r9nVTCyCEKNiBx371dG_SJx/view?usp=drive_link"
    output_file = "data_split.zip"

    try:
        # Convert Google Drive shareable link to direct download link
        file_id = drive_link.split('/')[-2]
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        
        st.write("Downloading dataset from Google Drive...")
        gdown.download(gdrive_url, output_file, quiet=False)

        # Unzip dataset
        st.write("Extracting dataset...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall("./data_split")
        
        st.write("Dataset downloaded and extracted successfully!")
        return "./data_split"
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        return None

# Main Streamlit app
st.title("Model Training and Testing")

# Download the dataset
data_dir = download_and_extract_data()

if data_dir:
    # Check if there is a nested data_split folder after extraction and adjust paths accordingly
    if os.path.exists(os.path.join(data_dir, "data_split")):
        data_dir = os.path.join(data_dir, "data_split")
    
    # Set paths to train, validation, and test datasets
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")
    st.success("Dataset paths set up successfully.")
    
    st.write("Train Directory:", train_dir)
    st.write("Validation Directory:", val_dir)
    st.write("Test Directory:", test_dir)

    # Add additional training/testing code here
else:
    st.error("Dataset could not be loaded.")

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)  # Adjust size dynamically if needed
        self.fc2 = nn.Linear(512, len(classes))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs):
    st.write("### Training Progress")
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        st.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc*100:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        st.write(f"Validation Accuracy: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    st.write("### Training Complete!")
    return model

# Evaluation function
def evaluate_model(model, test_loader):
    st.write("### Testing Results")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    st.write("#### Classification Report")
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    st.dataframe(report)

    st.write("#### Confusion Matrix")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    st.pyplot(plt)

# Streamlit UI
st.title("Image Classification Training Dashboard")

model = CustomCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if st.button("Start Training"):
    model = train_model(model, train_loader, val_loader, criterion, optimizer)

if st.button("Evaluate Model"):
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader)
