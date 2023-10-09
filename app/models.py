from tqdm import tqdm

import torchvision.models as models
import skvideo.io as skvideo
import numpy as np
import time
import cv2

import torch.nn as nn
import torch

import utils

from sklearn.metrics import jaccard_score

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

        self.bounding_box_model = getattr(
            models.detection,
            "fasterrcnn_resnet50_fpn"
        )(
            pretrained=self.pretrained_bool,
            progress=False
        ).to(self.device).eval()

    def __bounding_box(self, video_tensor: torch.Tensor, score_threshold: float = 0.9) -> torch.Tensor:
        """Returns the bounding box of the video.
        The bounding box is calculated as follows:
            bounding_box = (x1, y1, x2, y2)

        Args:
            video_tensor (np.ndarray): The video to extract the bounding box from.
            score_threshold (float, optional): The minimum score for a bounding box to be extracted. Defaults to 0.9.

        Returns:
            list: The bounding box of the video."""
        video_tensor = torch.from_numpy(video_tensor).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        video_tensor = video_tensor / 255.0

        with torch.no_grad():
            outputs = self.bounding_box_model(video_tensor)

        cropped_image = []
        for idx in range(len(outputs[0])):
            box_coord = outputs[0]['boxes'][idx]
            box_score = outputs[0]['scores'][idx]
            box_label = outputs[0]['labels'][idx]

            if box_score > score_threshold and box_label == 1:
                box_coord = box_coord.cpu().numpy()
                box_coord = box_coord.astype(int)

                x1, y1, x2, y2 = box_coord[0], box_coord[1], box_coord[2], box_coord[3]
                cropped_image = video_tensor[0, :, y1:y2, x1:x2]
                cropped_image = cropped_image.permute(1, 2, 0).cpu().numpy()
                break

        cropped_image = cv2.resize(cropped_image, (256, 512))
        cropped_image = torch.Tensor(cropped_image).permute(2, 0, 1)
        cropped_image = cropped_image.unsqueeze(0).float().to(self.device)

        return cropped_image
   
    def extract(self, video_tensor: cv2.VideoCapture, score_threshold: float = 0.93, video_length: float = 0.0) -> tuple:
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
                # cropped_human = self.__bounding_box(frame, score_threshold=score_threshold)
                outputs = self.model(frame_from_video)
            inference_time = time.time() - start_time

            # Gets the keypoints from the outputs
            # cropped_human = cropped_human.squeeze(0).permute(1, 2, 0).cpu().numpy()

            keypoints = utils.get_keypoints(outputs, None, threshold=score_threshold)

            try:
                extracted_skeletons = self.__add_keypoints(keypoints, extracted_skeletons)
            except:
                extracted_skeletons = self.__add_none_keypoints(extracted_skeletons)
            
            fps = 1.0 / inference_time
            total_fps += fps
            frame_count += 1

            pbar.set_postfix({"FPS": f"{fps:.2f}", "Average FPS": f"{total_fps / frame_count:.2f}"})
            pbar.update(1)

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

    def __add_keypoints(self, keypoints, input_mapping):
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
        file_ext = temp_video_file_path.split(".")[0] + "." + "mp4"
        video = self.__save_and_read_video_file(video_file, file_ext)
        video_height, video_width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        return video, video_height, video_width

class MetricsModel(nn.Module):
        def __init__(self):
            super(MetricsModel, self).__init__()
            self.guide_points_skeleton = nn.Linear(34, 64)
            self.consumer_points_skeleton = nn.Linear(34, 64)

            self.hidden_1 = nn.Linear(128, 64)

            self.score = nn.Sequential(
                nn.Linear(64, 16),
                nn.Linear(16, 1)
            )

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, guide_X, user_X):
            guide_X = self.relu(self.guide_points_skeleton(guide_X))
            user_X = self.relu(self.consumer_points_skeleton(user_X))

            x = torch.cat((guide_X, user_X), dim=1)
            x = self.relu(self.hidden_1(x))
            
            x = self.sigmoid(self.score(x))

            return x

class Metrics:
    def __video_normalize(self, skeleton: dict, video_height: int, video_width: int, cut_point: int):
        """Normalizes the skeleton to the video height and width.
        The skeleton is normalized as follows:
            normalized_skeleton = skeleton / video_height or video_width

        Args:
            skeleton (dict): The skeleton to normalize.
            video_height (int): The height of the video that the skeleton is extracted from.
            video_width (int): The width of the video that the skeleton is extracted from.
            cut_point (int): The cut point of the video that the skeleton is extracted from.

        Returns:
            dict: The normalized skeleton."""
        for key in skeleton.keys():
            coordinate = []
            for idx in range(int(cut_point)):
                x, y = skeleton[key][idx]
                x, y = x / video_width, y / video_height
                coordinate.append((x, y))
            skeleton[key] = coordinate

        return skeleton

    def __jaccard_score(self, 
                        y_true: torch.Tensor, 
                        y_pred: torch.Tensor) -> float:
        """Returns the jaccard score of the two arrays.
        The jaccard score is calculated as follows:
            jaccard_score = (y_true & y_pred).sum() / (y_true | y_pred).sum()

        Args:
            y_true (np.ndarray): The ground truth array.
            y_pred (np.ndarray): The predicted array.

        Returns:
            float: The jaccard score of the two arrays."""
        metrics = np.sum(np.min([y_true, y_pred], axis=0)) / np.sum(np.max([y_true, y_pred], axis=0))
        return float(metrics)
    
    def __linear_model(self, 
                       y_true: torch.Tensor, 
                       y_pred: torch.Tensor) -> float:
        model = MetricsModel()
        model.load_state_dict(
            torch.load("model.pth", 
                       map_location=torch.device('cpu')
        ))
        model.eval()

        with torch.no_grad():
            metrics = model(y_true, y_pred)
        metrics = metrics.cpu().numpy()
        metrics = np.sum(metrics) / len(metrics)
        return metrics

    def __normalized_mean_squared_error(self, y_true, y_pred) -> float:
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

    def score(self, 
              y_true: dict, true_video_height: int, true_video_width: int, true_cut_point: int,
              y_pred: dict, pred_video_height: int, pred_video_width: int) -> float:
        """Returns the score of the two arrays.
        The score is calculated as follows:
            score = (jaccard_score + normalized_mean_squared_error) / 2

        Args:
            y_true (np.ndarray): The ground truth array.
            true_video_height (int): The height of the video that the ground truth array is extracted from.
            true_video_width (int): The width of the video that the ground truth array is extracted from.
            y_pred (np.ndarray): The predicted array.
            pred_video_height (int): The height of the video that the predicted array is extracted from.
            pred_video_width (int): The width of the video that the predicted array is extracted from.

        Returns:
            float: The score of the two arrays.""" 
        
        # y_true = self.__video_normalize(y_true, true_video_height, true_video_width, true_cut_point)
        # y_pred = self.__video_normalize(y_pred, pred_video_height, pred_video_width, true_cut_point)

        y_true_values, y_pred_values = [], []
        for key in y_true.keys():
            y_true_value, y_pred_value = y_true[key], y_pred[key]
            y_true_values.extend(y_true_value)
            y_pred_values.extend(y_pred_value)
        y_true_values, y_pred_values = torch.Tensor(y_true_values), torch.Tensor(y_pred_values)
        y_true_values, y_pred_values = y_true_values.view(-1, 34), y_pred_values.view(-1, 34)

        minmum_length = min(y_true_values.shape[0], y_pred_values.shape[0])
        y_true_values, y_pred_values = y_true_values[:minmum_length, :], y_pred_values[:minmum_length, :]

        # metrics_score = self.__linear_model(y_true_values, y_pred_values)
        metrics_score = self.__jaccard_score(y_true_values, y_pred_values)

        return metrics_score