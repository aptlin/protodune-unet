import torch
from torch.utils.data import Dataset


class PlaneLoader(Dataset):
    def __init__(self, gt_data, noisy_data):
        self.gt_data = torch.load(gt_data)
        self.noisy_data = torch.load(noisy_data)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, index):
        noisy_image = self.noisy_data[index]
        if len(noisy_image.shape) == 2:
            noisy_image = noisy_image[None, :, :]
        gt_image = self.gt_data[index]
        if len(gt_image.shape) == 2:
            gt_image = gt_image[None, :, :]

        noise = noisy_image - gt_image
        return noisy_image, noise

