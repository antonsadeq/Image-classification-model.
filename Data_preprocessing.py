from glob import glob
from os import path
import torch
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional, List, Tuple


class ImagesDataset(Dataset):
    def __init__(self, image_dir=None, csv_file_path=None, width: int = 100, height: int = 100,
                 dtype: Optional[type] = None, filepaths: Optional[List[str]] = None, transform=None):
        self.transform = transform
        if filepaths is None:
            if image_dir is None:
                raise ValueError("Either image_dir or filepaths must be provided")
            self.image_filepaths = sorted(path.relpath(f) for f in glob(path.join(image_dir, "*.jpg")))
        else:
            self.image_filepaths = filepaths

        if not self.image_filepaths:
            raise FileNotFoundError("No image files found in the specified directory")

        if csv_file_path:
            self.class_filepath = csv_file_path
        else:
            if image_dir is None:
                raise ValueError("Either csv_file_path or image_dir must be provided to find the CSV file")
            class_filepath_list = [path.relpath(f) for f in glob(path.join(image_dir, "*.csv"))]
            if not class_filepath_list:
                raise FileNotFoundError("No CSV file found in the specified directory")
            self.class_filepath = class_filepath_list[0]

        self.filenames_classnames, self.classnames_to_ids = ImagesDataset.load_classnames(self.class_filepath)
        if width < 100 or height < 100:
            raise ValueError
        self.width = width
        self.height = height
        self.dtype = dtype

        max_label = max(self.classnames_to_ids.values())
        if max_label >= 20:
            raise ValueError

    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()
        classnames_to_ids = {classname: index for index, classname in enumerate(classnames)}
        return filenames_classnames, classnames_to_ids

    def __getitem__(self, index):
        with Image.open(self.image_filepaths[index]) as im:
            image = np.array(im, dtype=self.dtype)
        image = to_grayscale(image)
        resized_image, _ = prepare_image(image, self.width, self.height, 0, 0, 32)
        resized_image = torch.tensor(resized_image, dtype=torch.float32) / 255.0

        if self.transform:
            resized_image = self.transform(resized_image)

        filename = path.basename(self.image_filepaths[index])
        classname = self.filenames_classnames[self.filenames_classnames[:, 0] == filename][0][1]
        classid = self.classnames_to_ids[classname]

        # Check classid is in valid range
        if classid >= 20:  # Ensure this matches your actual number of classes
            raise ValueError(f"Class ID {classid} is out of bounds for num_classes=20")

        return resized_image, classid

    def __len__(self):
        return len(self.image_filepaths)

    def split_data(self, test_size: float) -> Tuple['ImagesDataset', 'ImagesDataset']:
        train_filepaths, test_filepaths = train_test_split(self.image_filepaths, test_size=test_size)
        train_dataset = ImagesDataset(image_dir=None, csv_file_path=self.class_filepath, width=self.width,
                                      height=self.height, dtype=self.dtype, filepaths=train_filepaths,
                                      transform=self.transform)
        test_dataset = ImagesDataset(image_dir=None, csv_file_path=self.class_filepath, width=self.width,
                                     height=self.height, dtype=self.dtype, filepaths=test_filepaths)
        return train_dataset, test_dataset
def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int):
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("image must have shape (1, H, W)")
    if width < 32 or height < 32 or size < 32:
        raise ValueError("width/height/size must be >= 32")
    if x < 0 or (x + size) > width:
        raise ValueError(f"x={x} and size={size} do not fit into the resized image width={width}")
    if y < 0 or (y + size) > height:
        raise ValueError(f"y={y} and size={size} do not fit into the resized image height={height}")

    image = image.copy()

    if image.shape[1] > height:
        image = image[:, (image.shape[1] - height) // 2: (image.shape[1] - height) // 2 + height, :]
    else:
        image = np.pad(image,
                       ((0, 0), ((height - image.shape[1]) // 2, math.ceil((height - image.shape[1]) / 2)), (0, 0)),
                       mode='edge')

    if image.shape[2] > width:
        image = image[:, :, (image.shape[2] - width) // 2: (image.shape[2] - width) // 2 + width]
    else:
        image = np.pad(image,
                       ((0, 0), (0, 0), ((width - image.shape[2]) // 2, math.ceil((width - image.shape[2]) / 2))),
                       mode='edge')

    subarea = image[:, y:y + size, x:x + size]
    return image, subarea


# Example usage
image_dir = "./training_data"
csv_file_path = "./training_data/labels.csv"
full_dataset = ImagesDataset(image_dir, csv_file_path, 100, 100, int)
train_dataset, test_dataset = full_dataset.split_data(test_size=1.0 / 5.0)

print(f"Number of full data samples: {len(full_dataset)}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
