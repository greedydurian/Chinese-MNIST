from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torch.utils.data import DataLoader, random_split
from step1.cnn import Net
from dataProcessor.loader import ChineseMnistDataset
from dataProcessor.transformData import TransformedChineseMnistDataset
import torch

class DatasetLoader:
    def __init__(self, csv_file, img_dir, batch_size, ratios=[0.7, 0.2, 0.1], seed=0):
        # transformation to be applied on the images
        self.transform = Compose([
            Resize((64, 64)),  # Resize images to 64x64
            ToTensor(),
            Normalize((0.5,), (0.5,))  # Adjust these values if necessary
        ])

        self.dataset = ChineseMnistDataset(csv_file=csv_file, 
                                            img_dir=img_dir)

        self.dataset = TransformedChineseMnistDataset(self.dataset, self.transform)

        # set the random seed for reproducible results
        torch.manual_seed(seed)

        # weights calculated above (0.7, 0.2, 0.1 respectively)
        train_size = int(ratios[0] * len(self.dataset))  
        val_size = int(ratios[1] * len(self.dataset))  
        test_size = len(self.dataset) - train_size - val_size 
        
        #random split here
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])


        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)