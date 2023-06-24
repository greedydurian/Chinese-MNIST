import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image
from torchvision import transforms
from step1.cnn import Net
import matplotlib.pyplot as plt
import mapping


image = Image.open('/Users/oha/Desktop/assessment/step2/wu.png')

# convert the image to grayscale if it has more than one color channel
if image.mode != 'L':
    image = image.convert('L')

model = Net()
model.load_state_dict(torch.load('/Users/oha/Desktop/assessment/best_model.pth'))
model.eval()

#transform here again 
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to the same size as your training images
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Apply the same normalization as your training images
])
inverted_image = Image.eval(image, lambda x: 255 - x)

# convert the inverted image back to grayscale
inverted_image = inverted_image.convert('L')

inverted_image_tensor = transform(inverted_image).unsqueeze(0)

plt.imshow(inverted_image, cmap='gray')
plt.show()

# use the model to predict the class
output = model(inverted_image_tensor)
print(output)


# get the predicted class
_, predicted_class = torch.max(output.data, 1)

# map each color to a class

predicted_class = predicted_class.item()

# call labels from the mapping script
class_mapping = mapping.class_mapping
predicted_character = class_mapping[predicted_class]

print(f'The model predicts that the image is: {predicted_character}')
