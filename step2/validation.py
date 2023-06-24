import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from step1.cnn import Net
from dataProcessor.dsLoader import DatasetLoader
import torch
import mapping

# Initialize DatasetLoader
datasetLoader = DatasetLoader(csv_file='/Users/oha/Desktop/assessment/chinese_mnist.csv', 
                              img_dir='/Users/oha/Desktop/assessment/data/data', 
                              batch_size=1)

# Get the validation loader
val_loader = datasetLoader.val_loader

# Load the trained model
model = Net()
model.load_state_dict(torch.load('/Users/oha/Desktop/assessment/best_model.pth'))
model.eval()

# Map for label visualization


# Perform validation and visualization
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)

        # Class with the highest probability is our predicted class
        _, predicted = torch.max(outputs.data, 1)

        # Plot the image and the actual and predicted labels
        img = images[0].numpy().squeeze()  # Remove batch dimension and convert to numpy
        labels_map = mapping.labels_map
        plt.imshow(img, cmap='gray')
        plt.title(f'Actual: {labels_map[labels.item()]}, Predicted: {labels_map[predicted.item()]}')
        plt.show()
