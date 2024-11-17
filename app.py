import streamlit as st
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Connect to your Wandb project
wandb.init(project="sign-language", entity="alibek-musabek-aitu")

# Simulate logging metrics in Wandb during training
epochs = 10
for epoch in range(epochs):
    # Simulate metrics (replace with actual values from your model)
    acc = 1 - 2 ** -epoch
    loss = 2 ** -epoch
    precision = 0.5 + 0.05 * epoch
    recall = 0.5 + 0.05 * epoch
    f1 = 2 * (precision * recall) / (precision + recall)  # Simple F1 calculation
    wandb.log({"acc": acc, "loss": loss, "precision": precision, "recall": recall, "f1": f1})

# After the training loop, you can retrieve the run data
run = wandb.Api().run("alibek-musabek-aitu/sign-language/balmy-moon-1")

# Fetching all the metrics (acc, loss, precision, recall, f1)
history = run.history()

# Streamlit dashboard
st.title('Model Training Metrics')
st.write("Here are the metrics logged during the training process.")

# Displaying the learning curve
st.subheader("Learning Curve")

# Plot Loss and Accuracy vs Epoch
fig, ax = plt.subplots()
ax.plot(history['acc'], label="Training Accuracy", color='blue')
ax.plot(history['loss'], label="Training Loss", color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('Learning Curve')
ax.legend()

st.pyplot(fig)

# Display Metrics in a table
st.subheader("Training Metrics")
st.dataframe(history[['acc', 'loss', 'precision', 'recall', 'f1']])

# Fetch confusion matrix from Wandb
# Simulating confusion matrix (replace with actual confusion matrix from your model)
true_labels = np.random.randint(0, 3, 100)  # Simulated true labels
pred_labels = np.random.randint(0, 3, 100)  # Simulated predicted labels

cm = confusion_matrix(true_labels, pred_labels)

# Plotting confusion matrix
st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(cm.shape[0]), yticklabels=np.arange(cm.shape[0]))
ax_cm.set_xlabel("Predicted Labels")
ax_cm.set_ylabel("True Labels")
ax_cm.set_title("Confusion Matrix")

st.pyplot(fig_cm)

# You can also display other hyperparameters logged in wandb
st.write(f"Hyperparameters used: ")
st.write(f"Learning Rate: {run.config['learning_rate']}")
st.write(f"Batch Size: {run.config['batch_size']}")
st.write(f"Epochs: {run.config['epochs']}")
