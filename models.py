from tqdm import tqdm

import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import time
import cv2

class SkeletonExtractor:
    def __init__(
            self, 
            pretrained_bool: bool = True, 
            number_of_keypoints: int = 17,
            device: str = 'cpu'):
        """SkeletonExtractor class for extracting skeletons from videos.
        Models are loaded from torchvision.models.detection.keypointrcnn_resnet50_fpn.
        The model is loaded onto the device specified by the device parameter.

        Args:
            pretrained_bool (bool, optional): Whether to load a pretrained model. Defaults to True.
            number_of_keypoints (int, optional): The number of keypoints to extract. Defaults to 17.
            device (str, optional): The device to load the model onto. Defaults to 'cpu'.

        Raises:
            ValueError: If the device is not 'cpu' or 'cuda'.

        Examples:
            >>> from models import SkeletonExtractor
            >>> extractor = SkeletonExtractor()
            >>> video = cv2.VideoCapture("videos/webm/ec07c4c7eb818d6c.webm")
            >>> skeletons = extractor.extract(video)
            >>> print(skeletons)"""
        self.pretrained_bool = pretrained_bool
        self.number_of_keypoints = number_of_keypoints
        self.device = device
        if self.device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {self.device}")
        
        self.model = models.detection.keypointrcnn_resnet50_fpn(pretrained=self.pretrained_bool, num_keypoints=self.number_of_keypoints)
        self.model.to(self.device).eval()

    def extract(self, video_tensor: cv2.VideoCapture, score_threshold: float = 0.9) -> list:
        """Extracts skeletons from a video using the model loaded onto the device specified in the constructor.

        Args:
            video_tensor (cv2.VideoCapture): The video to extract skeletons from.   
            score_threshold (float, optional): The minimum score for a skeleton to be extracted. Defaults to 0.9.

        Returns:
            list: A list of skeletons extracted from the video.

        Examples:
            >>> from models import SkeletonExtractor
            >>> extractor = SkeletonExtractor()
            >>> video = cv2.VideoCapture("videos/webm/ec07c4c7eb818d6c.webm")
            >>> skeletons = extractor.extract(video)
            >>> print(skeletons)"""
        total_fps, frame_count = 0., 0.
        extracted_skeletons = []
        pbar = tqdm(desc=f"Extracting skeletons from video", total=int(video_tensor.get(cv2.CAP_PROP_FRAME_COUNT)), unit="frames")

        while True:
            ret, frame = video_tensor.read()
            if not ret: break

            frame_from_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_from_video = np.array(frame, dtype=np.float64) / 255.0
            frame_from_video = torch.from_numpy(frame_from_video).permute(2, 0, 1)
            frame_from_video = frame_from_video.unsqueeze(0).float().to(self.device)

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(frame_from_video)[0]
            inference_time = time.time() - start_time

            # try:                keypoints = utils.get_keypoints(outputs, score_threshold)
            # except KeyError:    return "NO SKELETONS FOUND"

            # try:
            #     keypoints = utils.get_keypoints(outputs, score_threshold)
            #     extracted_skeletons.append(keypoints)
            # except KeyError:
            #     extracted_skeletons.append(None)     

            keypoints = utils.get_keypoints(outputs, score_threshold)
            if keypoints is not None:
                extracted_skeletons.append(keypoints)
            else:
                plt.imshow(frame_from_video)
                plt.show()

            fps = 1.0 / inference_time
            total_fps += fps
            frame_count += 1
            pbar.set_postfix({"FPS": f"{fps:.2f}", "Average FPS": f"{total_fps / frame_count:.2f}"})
            pbar.update(1)
        pbar.close()

        return extracted_skeletons