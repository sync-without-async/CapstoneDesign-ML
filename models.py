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
        
        self.model = models.detection.keypointrcnn_resnet50_fpn(weight=self.pretrained_bool, num_keypoints=self.number_of_keypoints)
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
        extracted_skeletons = self.__extract_keypoint_mapping({})
        pbar = tqdm(desc=f"Extracting skeletons from video", total=int(video_tensor.get(cv2.CAP_PROP_FRAME_COUNT)), unit="frames")

        while True:
            ret, frame = video_tensor.read()
            if not ret: break

            # Converts the frame from BGR to RGB and normalizes the values to be between 0 and 1
            frame_from_video = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_from_video = np.array(frame, dtype=np.float64) / 255.0

            # Transforms the frame into a tensor (1, 3, height, width)
            frame_from_video = torch.from_numpy(frame_from_video).permute(2, 0, 1)
            frame_from_video = frame_from_video.unsqueeze(0).float().to(self.device)

            # Runs the model on the frame and gets the keypoints
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(frame_from_video)
            inference_time = time.time() - start_time

            keypoints = utils.get_keypoints(outputs, score_threshold)

            if keypoints == "NO SKELETONS FOUND":
                extracted_skeletons = self.__add_none_keypoints(extracted_skeletons)
            else:
                extracted_skeletons = self.__add_keypoints(keypoints, extracted_skeletons)

            fps = 1.0 / inference_time
            total_fps += fps
            frame_count += 1
            pbar.set_postfix({"FPS": f"{fps:.2f}", "Average FPS": f"{total_fps / frame_count:.2f}"})
            pbar.update(1)
        pbar.close()

        return extracted_skeletons

    def __add_none_keypoints(self, input_mapping: dict) -> dict:
        """Adds None keypoints to the input mapping.
        Keypoints are indexed from 0 to 16. The index number corresponds to the index of the keypoint in the list of keypoints.

        Args:
            input_mapping (dict): The input mapping to add the None keypoints to.

        Returns:
            dict: The input mapping with the None keypoints added."""
        for idx in range(17):
            input_mapping[self.__return_keypoint_name_from_index(idx)].append((None, None))
        return input_mapping

    def __add_keypoints(self, keypoints: list, input_mapping: dict) -> dict:
        """Adds the keypoints to the input mapping.
        Keypoints are indexed from 0 to 16. The index number corresponds to the index of the keypoint in the list of keypoints.
        
        Args:
            keypoints (list): The list of keypoints to add to the input mapping.
            input_mapping (dict): The input mapping to add the keypoints to.
            
        Returns:
            dict: The input mapping with the keypoints added."""
        for idx in range(len(keypoints)):
            x, y = keypoints[idx][0], keypoints[idx][1]
            input_mapping[self.__return_keypoint_name_from_index(idx)].append((x.item(), y.item()))
        return input_mapping

    def __extract_keypoint_mapping(self, input_mapping: dict) -> dict:
        """Returns a dictionary with the keypoint names as keys and empty lists as values.
        Keypoints are indexed from 0 to 16. The index number corresponds to the index of the keypoint in the list of keypoints.
        Keypoint names are as follows:
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear',
            4: 'right_ear',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle',
            
        Args:
            input_mapping (dict): The input dictionary to add the keypoint names to.
            
        Returns:
            dict: The input dictionary with the keypoint names added."""
        keypoint_names = {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear', 
            4: 'right_ear', 
            5: 'left_shoulder', 
            6: 'right_shoulder',
            7: 'left_elbow', 
            8: 'right_elbow',
            9: 'left_wrist', 
            10: 'right_wrist',
            11: 'left_hip', 
            12: 'right_hip',
            13: 'left_knee', 
            14: 'right_knee',
            15: 'left_ankle', 
            16: 'right_ankle',
        }

        for key in keypoint_names.keys():
            input_mapping[keypoint_names[key]] = []

        return input_mapping
    
    def __return_keypoint_name_from_index(self, index_number: int) -> str:
        """Returns the name of the keypoint from the index number.
        Keypoints are indexed from 0 to 16. The index number corresponds to the index of the keypoint in the list of keypoints.
        
        Args:
            index_number (int): The index number of the keypoint.
            
        Returns:
            str: The name of the keypoint."""
        keypoint_names = {
            0: 'nose',
            1: 'left_eye',
            2: 'right_eye',
            3: 'left_ear', 
            4: 'right_ear', 
            5: 'left_shoulder', 
            6: 'right_shoulder',
            7: 'left_elbow', 
            8: 'right_elbow',
            9: 'left_wrist', 
            10: 'right_wrist',
            11: 'left_hip', 
            12: 'right_hip',
            13: 'left_knee', 
            14: 'right_knee',
            15: 'left_ankle', 
            16: 'right_ankle',
        }
        return keypoint_names[index_number]