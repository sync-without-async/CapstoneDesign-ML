from typing import Any

import torch.nn.functional as F
import torch.nn as nn
import base64
import torch
import cv2

import numpy as np

class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_shape: tuple = (28, 28),
            output_shape: tuple = (28, 28),
    ):  
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.flatten = nn.Flatten()
        self.en_linear1 = nn.Linear(in_features=self.input_shape[0] * self.input_shape[1], out_features=128)
        self.en_linear2 = nn.Linear(in_features=128, out_features=64)
        self.en_linear3 = nn.Linear(in_features=64, out_features=32)
        self.en_linear4 = nn.Linear(in_features=32, out_features=16)

        self.de_linear1 = nn.Linear(in_features=16, out_features=32)
        self.de_linear2 = nn.Linear(in_features=32, out_features=64)
        self.de_linear3 = nn.Linear(in_features=64, out_features=128)
        self.de_linear4 = nn.Linear(in_features=128, out_features=self.output_shape[0] * self.output_shape[1])

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.flatten(x)
        x = F.sigmoid(self.en_linear1(x))
        x = F.sigmoid(self.en_linear2(x))
        x = F.sigmoid(self.en_linear3(x))
        x = F.sigmoid(self.en_linear4(x))

        x = F.sigmoid(self.de_linear1(x))
        x = F.sigmoid(self.de_linear2(x))
        x = F.sigmoid(self.de_linear3(x))
        x = F.sigmoid(self.de_linear4(x))

        return x

class DataPreprocessing:
    def __init__(
            self, 
            target_datatype: np.float32 = None, 
            image_width: int = 28,
            image_height: int = 28,
            image_channel: int = 1
        ):
        self.target_datatype = target_datatype
        if self.target_datatype is None: ValueError(f"target_datatype must be specified. (e.g. np.float32)\nExcepted: {np.float32}, Input: {self.target_datatype}")

        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        if self.image_width is not int or self.image_height is not int or self.image_channel is not int: 
            ValueError(f"image_width, image_height, image_channel must be specified. (e.g. 28, 28, 1)\nExcepted: {int}, Input: {self.image_width, self.image_height, self.image_channel}")

    def __call__(self, image: np.ndarray) -> torch.tensor:
        image = base64.b64decode(image)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.resize(image, (28, 28))
        image = image.reshape(-1, self.image_channel, self.image_width, self.image_height)
        image = image / 255.0
        image = image.astype(self.target_datatype)

        return image