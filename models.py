from tqdm import tqdm

import torchvision.models as models
import skvideo.io as skvideo
import numpy as np
import torch
import utils
import time
import cv2
import os

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
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {self.device}")

        self.model = getattr(
            models.detection, 
            "keypointrcnn_resnet50_fpn"
        )(
            pretrained=self.pretrained_bool,
            num_keypoints=self.number_of_keypoints,
            progress=False
        ).to(self.device).eval()
   
    def extract(self, video_tensor: cv2.VideoCapture, score_threshold: float = 0.9, video_length: float = None) -> dict:
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
        if video_length is None: video_length = int(video_tensor.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO/EXTRACT] Extracting skeletons from video. Video length: {video_length} frames.")

        total_fps, frame_count = 0., 0.
        extracted_skeletons = self.__extract_keypoint_mapping({})
        pbar = tqdm(desc=f"Extracting skeletons from video", total=video_length, unit="frames")

        while True:
            ret, frame = video_tensor.read()
            if not ret: break
            if frame_count == video_length: break
            
            # Preprocesses the frame
            frame_from_video = np.array(frame, dtype=np.float32) / 255.0
            frame_from_video = torch.Tensor(frame_from_video).permute(2, 0, 1)
            frame_from_video = frame_from_video.unsqueeze(0)
            frame_from_video = frame_from_video.float().to(self.device)

            # Runs the model on the frame and gets the keypoints
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(frame_from_video)
            inference_time = time.time() - start_time

            # Gets the keypoints from the outputs
            keypoints = utils.get_keypoints(outputs, None, threshold=score_threshold)
            output_image = utils.draw_keypoints(outputs, frame)

            try:
                extracted_skeletons = self.__add_keypoints(keypoints, extracted_skeletons)
            except:
                extracted_skeletons = self.__add_none_keypoints(extracted_skeletons)
            
            fps = 1.0 / inference_time
            total_fps += fps
            frame_count += 1

            pbar.set_postfix({"FPS": f"{fps:.2f}", "Average FPS": f"{total_fps / frame_count:.2f}"})
            pbar.update(1)

            cv2.imshow("Output", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        pbar.close()
        cv2.destroyAllWindows()

        return extracted_skeletons, frame_count

    def __add_none_keypoints(self, input_mapping: dict) -> dict:
        """Adds None keypoints to the input mapping.
        Keypoints are indexed from 0 to 16. The index number corresponds to the index of the keypoint in the list of keypoints.

        Args:
            input_mapping (dict): The input mapping to add the None keypoints to.

        Returns:
            dict: The input mapping with the None keypoints added."""
        for idx in range(17):
            input_mapping[self.__return_keypoint_name_from_index(idx)].append((0, 0))
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
    
class DataPreprocessing:
    def __save_and_read_video_file(self, video, temp_video_file_path):
        with open(temp_video_file_path, "wb+") as f:
            for chunk in video.file: f.write(chunk)
        video.file.close()
        video = cv2.VideoCapture(temp_video_file_path)

        return video

    def processing(self, video_file, temp_video_file_path: str = "temp.webm"):
        """Processes the video file and returns the video tensor.
        Save the video file to the temp_video_file_path and read it. Then, convert it to the video tensor.
        
        Args:
            video_file (UploadFile): The video file to process.
            temp_video_file_path (str, optional): The path to save the video file to. Defaults to "temp.webm".
            
        Returns:
            torch.Tensor: The video tensor."""
        file_ext = video_file.filename.split(".")[-1]
        file_ext = temp_video_file_path.split(".")[0] + "." + file_ext
        video = self.__save_and_read_video_file(video_file, file_ext)
        return video

class Metrics:
    def __jaccard_score(self, y_true: list, y_pred: list) -> float:
        """Returns the jaccard score of the two arrays.
        The jaccard score is calculated as follows:
            jaccard_score = (y_true & y_pred).sum() / (y_true | y_pred).sum()

        Args:
            y_true (np.ndarray): The ground truth array.
            y_pred (np.ndarray): The predicted array.

        Returns:
            float: The jaccard score of the two arrays."""
        metrics = np.sum(np.min([y_true, y_pred], axis=0)) / np.sum(np.max([y_true, y_pred], axis=0))
        return metrics

    def __normalized_mean_squared_error(self, y_true: list, y_pred: list) -> float:
        """Returns the normalized mean squared error of the two arrays.
        The normalized mean squared error is calculated as follows:
            normalized_mean_squared_error = (y_true - y_pred)^2 / (y_true - y_true.mean())^2

        Args:
            y_true (np.ndarray): The ground truth array.
            y_pred (np.ndarray): The predicted array.

        Returns:
            float: The normalized mean squared error of the two arrays."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        metrics = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        return metrics

    def score(self, y_true: dict, y_pred: dict) -> float:
        """Returns the score of the two arrays.
        The score is calculated as follows:
            score = jaccard_score(y_true, y_pred)
            
        Args:
            y_true (np.ndarray): The ground truth array.
            y_pred (np.ndarray): The predicted array.
            
        Returns:
            float: The score of the two arrays."""
        scores = []
        for key in y_true:
            scores.append(
                self.__normalized_mean_squared_error(y_true[key], y_pred[key])
            )

        score = np.mean(scores)
        return score

