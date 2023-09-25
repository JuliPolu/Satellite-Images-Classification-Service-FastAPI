import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import numpy as np


def preprocess_image(image: np.ndarray, target_image_size: int) -> torch.Tensor:
    """Препроцессинг имаджнетом.

    :param image: RGB-изображение;
    :param target_image_size: целевой размер изображения;
    :return: обработанный тензор.
    """
    image = image.astype(np.float32)

    preprocess = albu.Compose(
        [
            albu.Resize(height=target_image_size, width=target_image_size),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )

    image = preprocess(image=image)['image']

    return image[None]
