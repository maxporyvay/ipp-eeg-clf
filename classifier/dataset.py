import torch.utils.data


class MorletDataset(torch.utils.data.Dataset):
    """
    Dataset class for morlet data
    """

    def __init__(self, labels, morlets, transform=None, target_transform=None):
        self.labels = labels
        self.morlets = morlets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        morlet, label = self.morlets[idx], self.labels[idx]
        if self.transform:
            morlet = self.transform(morlet)
        if self.target_transform:
            label = self.target_transform(label)
        return morlet, label
