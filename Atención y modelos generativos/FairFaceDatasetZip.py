from torch.utils.data import Dataset
import os, PIL.Image

class FairFaceDatasetZip(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir      
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.df.loc[idx, "file"]

        img_path = os.path.join(self.root_dir, file_name)
        img = PIL.Image.open(img_path).convert("RGB")
        cat = self.df.loc[idx, "age_CG"]
        if self.transform:
            img = self.transform(img)

        return img, cat

    def __len__(self):
        return len(self.df)