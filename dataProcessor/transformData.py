from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset
from dataProcessor.loader import ChineseMnistDataset

class TransformedChineseMnistDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if label == -1:
            # image to a tensor
            image = ToTensor()(image)
        elif self.transform:
            image = self.transform(image)

        return image, label
