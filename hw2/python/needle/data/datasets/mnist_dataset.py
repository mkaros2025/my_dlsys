from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as f:
            header = f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.astype(np.float32) / 255.0
            self.images = images.reshape((-1, 28, 28, 1))

        with gzip.open(label_filename, "rb") as f:
            header = f.read(8)
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.images[index]
        labels = self.labels[index]

        if self.transforms is not None:
            if len(imgs.shape) == 3:
                imgs = self.apply_transforms(imgs)
            else:
                imgs = np.array([self.apply_transforms(img) for img in imgs])

        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.__len__()
        ### END YOUR SOLUTION