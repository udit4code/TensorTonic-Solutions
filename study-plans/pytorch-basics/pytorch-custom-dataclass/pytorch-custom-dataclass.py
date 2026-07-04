import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        data_tensor = torch.tensor(data, dtype=torch.float32)
        feature_cols = [idx for idx in range(data_tensor.shape[1]) if idx != label_col]
        self.features = data_tensor[:, feature_cols]
        # unsqueeze() inserts a new dimension of size 1 at the specified position.
        # Why ? Because we expect labels to be of the shape (N,1) instead of (N,)
        self.labels = data_tensor[:, label_col].unsqueeze(1)
        assert  len(self.features) == len(self.labels), "Mismatch in count of features and labels"
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
