import streamlit as st
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import io  

wandb.login(key='e6bbb13bc6a48abd9cddaf89523b51f76fe4dbd1')

# Initialize wandb for retrieving the specific run's data
wandb.init(project="sign-language", anonymous="allow")

# Fetch the specific run using the run ID
run_id = "lvsatyew"  # Replace this with your actual run ID
api = wandb.Api()
run = api.run(f"alibek-musabek-aitu/sign-language/{run_id}")  # Get the specific run by ID

# Display some basic information about the run
st.title(f"Sign Language Model Training Results - {run.name}")
st.write(f"Run ID: {run.id}")
st.write(f"Created at: {run.created_at}")

# Display the training parameters used in this run
st.subheader("Training Hyperparameters")
st.write(f"Learning Rate: {run.config['learning_rate']}")
st.write(f"Batch Size: {run.config['batch_size']}")
st.write(f"Weight Decay: {run.config['weight_decay']}")
st.write(f"Epochs: {run.config['epochs']}")
st.write(f"Model Architecture: {run.config['architecture']}")

# Fetch the metrics (loss, accuracy, precision, recall, etc.)
history = run.history(keys=["train_loss", "train_accuracy", "val_loss", "val_accuracy", "train_precision", "train_recall", "train_f1", "val_precision", "val_recall", "val_f1"])

# Display the metrics in a plot
st.subheader("Training and Validation Metrics")
metrics = history.dropna(subset=["train_loss", "val_loss"])

# Plotting train and validation loss
fig, ax = plt.subplots()
ax.plot(metrics["train_loss"], label="Train Loss")
ax.plot(metrics["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

# Plotting train and validation accuracy
fig, ax = plt.subplots()
ax.plot(metrics["train_accuracy"], label="Train Accuracy")
ax.plot(metrics["val_accuracy"], label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# Plotting F1 score for both train and validation
fig, ax = plt.subplots()
ax.plot(metrics["train_f1"], label="Train F1 Score")
ax.plot(metrics["val_f1"], label="Validation F1 Score")
ax.set_xlabel("Epoch")
ax.set_ylabel("F1 Score")
ax.legend()
st.pyplot(fig)

# Display confusion matrices
st.subheader("Confusion Matrix")
confusion_matrix_images = run.files()  # Get all the files uploaded in the run
for file in confusion_matrix_images:
    if "confusion_matrix" in file.name and file.name.endswith('.png'):
        # Download the image as binary data and open it using PIL
        img_data = file.download()  # Download the image file as binary data
        img = Image.open(io.BytesIO(img_data))  # Open the image from the binary data using BytesIO
        st.image(img, caption=f"Confusion Matrix - Epoch {file.name.split('_')[-1].split('.')[0]}")

# Finish the wandb session
wandb.finish()

