import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        super(MyDataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.y is not None:
            y = self.y[idx]
        else:
            y = None
        return x, y

def create_dataloader(dataset, batch_size, num_workers,
                      shuffle=True, use_gpu=False):
    dl = DataLoader(dataset, batch_size=batch_size,
                    num_workers=num_workers, shuffle=shuffle,
                    pin_memory=use_gpu if num_workers>0 else False)
    return dl

if __name__ == "__main__":
    x = torch.load("../data/input_data.pt", weights_only=False)
    y = torch.load("../data/labels.pt", weights_only=False)
    dataset = MyDataset(x, y)
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])