import torch
from torch.utils.data import TensorDataset

class C2P_Dataset(TensorDataset):
    """Dataset class with normalization capabilities"""
    
    def __init__(self, C, Z, normalize_data=False, margin=0.0, dtype=float):
        self.C = C.to(dtype)
        self.Z = Z.to(dtype)

        self.margin = margin

        self.C_max = torch.max(self.C, dim=0, keepdim=True)[0]
        self.C_min = torch.min(self.C, dim=0, keepdim=True)[0]
        self.Z_max = torch.max(self.Z, dim=0, keepdim=True)[0]
        self.Z_min = torch.min(self.Z, dim=0, keepdim=True)[0]

        
        if normalize_data:
            self.normalize()
        
    def normalize(self):
        """Min-max normalization"""
        self.C = (self.C - self.C_min) / (self.C_max - self.C_min)
        
        Z_range = self.Z_max - self.Z_min
        self.Z_min -= self.margin * Z_range
        self.Z_max += self.margin * Z_range

        self.Z = (self.Z - self.Z_min) / (self.Z_max - self.Z_min)
    
    def denormalize(self):
        """Reverse normalization"""
        self.C = self.C * (self.C_max - self.C_min) + self.C_min
        self.Z = self.Z * (self.Z_max - self.Z_min) + self.Z_min
    
    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, idx):
        return self.C[idx, :], self.Z[idx, :]