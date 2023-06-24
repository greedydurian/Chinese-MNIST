import sys
import os
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from step1.cnn import Net
from dataProcessor.dsLoader import DatasetLoader

datasetLoader = DatasetLoader(csv_file='/Users/oha/Desktop/assessment/chinese_mnist.csv',
                              img_dir='/Users/oha/Desktop/assessment/data/data',
                              batch_size=32,
                              ratios=[0.7, 0.2, 0.1])  # using the split ratios you defined

# use the test_loader
test_loader = datasetLoader.test_loader

# calling loaders from datasetLoader
model = Net()
model.load_state_dict(torch.load('/Users/oha/Desktop/assessment/best_model.pth'))
model.eval()

# Perform testing
correct = 0
total = 0
all_labels = []
all_predictions = []
all_probabilities = []
number_of_classes = 10 

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predicted.numpy().tolist())
        all_probabilities.extend(probabilities.numpy().tolist())

print('Test accuracy: %d %%' % (100 * correct / total))
print('Classification Report:')
print(classification_report(all_labels, all_predictions))

# calc AUC 
y_bin = label_binarize(all_labels, classes=list(range(number_of_classes)))
auc_score = roc_auc_score(y_bin, np.array(all_probabilities), multi_class='ovo') 
print(f'AUC Score: {auc_score}')
