import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.X, self.y = [], []
        if train:
            file_names = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            file_names = ["test_batch"]

        data_list, labels_list = [], []

        for name in file_names:
            file_path = os.path.join(base_folder, name)
            with open(file_path, 'rb') as fo:
                batch_dict = pickle.load(fo, encoding='bytes')
                data_list.append(batch_dict[b'data'])
                labels_list.extend(batch_dict[b'labels'])

        X_raw = np.concatenate(data_list, axis=0)
        self.X = X_raw.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.array(labels_list)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img = self.X[index]
        label = self.y[index]
        if self.transforms:
            for t in self.transforms:
                img = t(img)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
