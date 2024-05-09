import torch

import utils.globals as uglobals

# Placeholder dataset
class PlaceholderDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = self.gen_data()

    def gen_data(self):
        # y = x1^2 + x2 + noise
        x = torch.randn(1000, 2)
        noise = 0.1 * torch.randn(1000, 1)
        y = x[: ,0] ** 2 + x[:, 1] + noise[:, 0]
        
        data = torch.cat([x, y.unsqueeze(1)], dim=1)
        return data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def placeholder_collate(batch):
    x = torch.stack([item[: -1] for item in batch])
    y = torch.stack([item[-1:] for item in batch])
    return x, y

def make_placeholder_loader(batch_size, shuffle=True, single_worker=False):
    dataset = PlaceholderDataset()
    if single_worker:
        n_workers = 1
        persistent_workers = False
    else:
        n_workers = uglobals.NUM_WORKERS
        persistent_workers = True
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=placeholder_collate, 
        num_workers=n_workers,
        persistent_workers=persistent_workers,
        )
    return loader