import streamlit as st
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import io  
# Authenticate with Wandb
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

# Display precision and recall for both train and validation
st.subheader("Precision and Recall")
st.write("**Train Precision:**", metrics["train_precision"].iloc[-1])
st.write("**Train Recall:**", metrics["train_recall"].iloc[-1])
st.write("**Train F1 Score:**", metrics["train_f1"].iloc[-1])
st.write("**Validation Precision:**", metrics["val_precision"].iloc[-1])
st.write("**Validation Recall:**", metrics["val_recall"].iloc[-1])
st.write("**Validation F1 Score:**", metrics["val_f1"].iloc[-1])

# Display confusion matrices
st.subheader("Confusion Matrix")

# Get all the files uploaded in this run
confusion_matrix_images = run.files()  # Corrected: call the method to get the files

# Create a specific folder if it doesn't exist
output_folder = './media/images/'
os.makedirs(output_folder, exist_ok=True)

# Iterate through the files to find confusion matrix image
for file in confusion_matrix_images:
    # Check for a file name that matches the confusion matrix image (for example, ending with .png or .jpg)
    if "confusion_matrix" in file.name.lower() and (file.name.endswith('.png') or file.name.endswith('.jpg')):
        # Download the image to the specified folder
        image_path = file.download(root=output_folder)

        # Open the image using PIL after download
        image = Image.open(image_path)

        # Display the image in Streamlit
        st.image(image, caption="Confusion Matrix", use_column_width=True)

# Finish the wandb session
wandb.finish()
