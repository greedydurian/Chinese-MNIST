import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch

class ChineseMnistDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        # only get 0-9
        self.df = self.df[self.df['code'] <= 10]
        self.img_dir = img_dir
        self.cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Use cached image if it exists, else open and cache
        if img_name in self.cache:
            image = self.cache[img_name]
        else:
            try:
                image = Image.open(img_path)
                self.cache[img_name] = image
            except Exception as e:
                print(f'Error occurred when loading image {img_path}: {str(e)}')
                # Return a default sample
                return Image.new('RGB', (28, 28)), -1

        label = row['code'] - 1 #using zero-based indexing
        
        return image, label
    
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def visualize_samples(self, num_samples=9):
        labels_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }

        figure = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, num_samples + 1):
            sample_idx = torch.randint(len(self), size=(1,)).item()
            img, label = self[sample_idx]
            figure.add_subplot(rows, cols, i)
            plt.title(labels_map[label])
            plt.axis("off")
            plt.imshow(img, cmap="gray")
        plt.show()

