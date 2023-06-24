from dataProcessor.loader import ChineseMnistDataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Normalize, Compose
from dataProcessor.transformData import TransformedChineseMnistDataset

dataset = ChineseMnistDataset(csv_file='/Users/oha/Desktop/assessment/chinese_mnist.csv', img_dir='/Users/oha/Desktop/assessment/data/data')

image, label = dataset[0]

# Display the image
plt.imshow(image)
plt.show()

# Access other properties of the image
image_size = image.size
image_format = image.format

transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

dataset = ChineseMnistDataset(csv_file='chinese_mnist.csv', img_dir='data')
dataset = TransformedChineseMnistDataset(dataset, transform)