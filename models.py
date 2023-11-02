from tqdm import tqdm

import torchvision.models as models
import numpy as np
import torch
import time
import cv2

import logging
import utils

logging.basicConfig(level=logging.INFO)

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
        extracted_skeletons_cropped = self.__extract_keypoint_mapping({})
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
                cropped_human = self.__bounding_box(frame, score_threshold=score_threshold)

                cropped_output = self.model(cropped_human)
                outputs = self.model(frame_from_video)
            inference_time = time.time() - start_time

            keypoints = utils.get_keypoints(outputs, None, threshold=score_threshold)
            keypoints_cropped = utils.get_keypoints(cropped_output, None, threshold=score_threshold)

            try:
                extracted_skeletons = self.__add_keypoints(keypoints, extracted_skeletons)
                extracted_skeletons_cropped = self.__add_keypoints(keypoints_cropped, extracted_skeletons_cropped)
            except:
                extracted_skeletons = self.__add_none_keypoints(extracted_skeletons)
                extracted_skeletons_cropped = self.__add_none_keypoints(extracted_skeletons_cropped)
            
            fps = 1.0 / inference_time
            total_fps += fps
            frame_count += 1

            pbar.set_postfix({"FPS": f"{fps:.2f}", "Average FPS": f"{total_fps / frame_count:.2f}"})
            pbar.update(1)

        pbar.close()

        return extracted_skeletons, extracted_skeletons_cropped, frame_count

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
                        y_true, y_pred) -> float:
        """Returns the jaccard score of the two arrays.
        The jaccard score is calculated as follows:
            jaccard_score = (y_true & y_pred).sum() / (y_true | y_pred).sum()

        Args:
            y_true (np.ndarray): The ground truth array.
            y_pred (np.ndarray): The predicted array.

        Returns:
            float: The jaccard score of the two arrays."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)        
        metrics = np.sum(np.min([y_true, y_pred], axis=0)) / np.sum(np.max([y_true, y_pred], axis=0))
        metrics = float(metrics)

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
    
    def weighted_score(self,
                       wegiht_target_part: str,
                       y_true: dict, true_video_height: int, true_video_width: int, true_cut_point: int,
                       y_pred: dict, pred_video_height: int, pred_video_width: int) -> float:
        wegiht_target_part = wegiht_target_part.upper()
        weighted_y_true, weighted_y_pred = [], []

        if wegiht_target_part == "SHOULDER":
            for key in y_true.keys():
                if key == "left_shoulder"   \
                or key == "right_shoulder"  \
                or key == "left_elbow"      \
                or key == "right_elbow"     \
                or key == "left_wrist"      \
                or key == "right_wrist":
                    weighted_y_true.extend(y_true[key] * 0.9)
                    weighted_y_pred.extend(y_pred[key] * 0.9)
                else:
                    weighted_y_true.extend(y_true[key] * 0.1)
                    weighted_y_pred.extend(y_pred[key] * 0.1)           

        elif wegiht_target_part == "KNEE":
            for key in y_true.keys():
                if key == "left_knee"   \
                or key == "right_knee"  \
                or key == "left_hip"    \
                or key == "right_hip"   \
                or key == "left_ankle"  \
                or key == "right_ankle":
                    weighted_y_true.extend(y_true[key] * 0.9)
                    weighted_y_pred.extend(y_pred[key] * 0.9)
                else:
                    weighted_y_true.extend(y_true[key] * 0.1)
                    weighted_y_pred.extend(y_pred[key] * 0.1)

        elif wegiht_target_part == "THIGHS":
            for key in y_true.keys():
                if key == "left_knee"   \
                or key == "right_knee"  \
                or key == "left_hip"    \
                or key == "right_hip":
                    weighted_y_true.extend(y_true[key] * 0.9)
                    weighted_y_pred.extend(y_pred[key] * 0.9)
                else:
                    weighted_y_true.extend(y_true[key] * 0.1)
                    weighted_y_pred.extend(y_pred[key] * 0.1)

        elif wegiht_target_part == "ARMS":
            for key in y_true.keys():
                if key == "left_shoulder"   \
                or key == "right_shoulder"  \
                or key == "left_elbow"      \
                or key == "right_elbow"     \
                or key == "left_wrist"      \
                or key == "right_wrist":
                    weighted_y_true.extend(y_true[key] * 0.9)
                    weighted_y_pred.extend(y_pred[key] * 0.9)
                else:
                    weighted_y_true.extend(y_true[key] * 0.1)
                    weighted_y_pred.extend(y_pred[key] * 0.1)

        else:
            raise ValueError(f"Invalid target part: {wegiht_target_part}")

        minimum_length = min(len(weighted_y_true), len(weighted_y_pred))
        weighted_y_true, weighted_y_pred = weighted_y_true[:minimum_length], weighted_y_pred[:minimum_length]

        metrics_score = self.__jaccard_score(weighted_y_true, weighted_y_pred)
        return metrics_score

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
        
        y_true = self.__video_normalize(y_true, true_video_height, true_video_width, true_cut_point)
        y_pred = self.__video_normalize(y_pred, pred_video_height, pred_video_width, true_cut_point)

        y_true_values, y_pred_values = [], []
        for key in y_true.keys():
            y_true_value, y_pred_value = y_true[key], y_pred[key]
            y_true_values.extend(y_true_value)
            y_pred_values.extend(y_pred_value)
        y_true_values, y_pred_values = torch.Tensor(y_true_values), torch.Tensor(y_pred_values)
        y_true_values, y_pred_values = y_true_values.view(-1, 34), y_pred_values.view(-1, 34)

        minmum_length = min(y_true_values.shape[0], y_pred_values.shape[0])
        y_true_values, y_pred_values = y_true_values[:minmum_length, :], y_pred_values[:minmum_length, :]

        metrics_score = self.__jaccard_score(y_true_values, y_pred_values)

        return metrics_score
    
class MMPoseStyleSimilarty:
    def score(self, 
              guide_skeleton, 
              consumer_skeleton,
              execrise_points: str = "NONE"
              ) -> float:
        guide_skeleton = self.__get_valid_incidences(skeleton=guide_skeleton, execute_points=execrise_points)
        consumer_skeleton = self.__get_valid_incidences(skeleton=consumer_skeleton, execute_points=execrise_points)
        guide_skeleton, consumer_skeleton = self.__cut_minimum_length(guide_skeleton, consumer_skeleton)

        matrix = torch.stack([guide_skeleton, consumer_skeleton], dim=3)
        matrix_clone = matrix.clone()
        matrix_clone[matrix == 0.0] = 256.0
        x_min, y_min = matrix_clone.narrow(3, 0, 1).min(dim=2).values, matrix_clone.narrow(3, 1, 1).min(dim=2).values
        x_max, y_max = matrix_clone.narrow(3, 0, 1).max(dim=2).values, matrix_clone.narrow(3, 1, 1).max(dim=2).values

        matrix_clone = matrix.clone()
        matrix_clone[:, :, :, 0] = (matrix_clone[:, :, :, 0] - x_min) / (x_max - x_min + 1e-5)
        matrix_clone[:, :, :, 1] = (matrix_clone[:, :, :, 1] - y_min) / (y_max - y_min + 1e-5)
        normalized_matrix = matrix_clone.clone()

        xy_dist = matrix_clone[:, :, :, 0] - matrix_clone[:, :, :, 1] 
        score = matrix_clone[:, :, :, 0] * matrix_clone[:, :, :, 1]

        similarty = (torch.exp(-50 * xy_dist.pow(2).sum(dim=-1).unsqueeze(-1)) * score).sum(dim=-1) / score.sum(dim=-1) + 1e-6
        similarty[similarty.isnan()] = 0.0
        print(f"Similarty Vector: {similarty}")
        print(f"Normalized Matrix ranges from {normalized_matrix.min()} to {normalized_matrix.max()}")

        similarty = similarty.mean().item()
        return similarty

    def __cut_minimum_length(self,
                             guide_skeleton: torch.Tensor,
                             consumer_skeleton: torch.Tensor,
                             ) -> tuple:        
        guide_skeleton_shape, consumer_skeleton_shape = guide_skeleton.shape, consumer_skeleton.shape
        minimum_length = min(guide_skeleton_shape[1], consumer_skeleton_shape[1])
        guide_skeleton, consumer_skeleton = guide_skeleton[:, :minimum_length, :], consumer_skeleton[:, :minimum_length, :]
        return guide_skeleton, consumer_skeleton

    def __get_valid_incidences(self, 
                               skeleton: dict = None,
                               valid_incidences: np.ndarray = None,
                               execute_points: str = "NONE",
                               ) -> torch.Tensor:
        """Returns the valid incidences of the skeleton.
        The valid incidences are as follows:
            valid_incidences = [0] + list(range(5, 17))

        Args:
            skeleton (dict): The skeleton to get the valid incidences from.

        Returns:
            dict: The valid incidences of the skeleton."""
        # Default valid incidences
        if valid_incidences is None and execute_points == "NONE":
            logging.warning(f"[WARNING/GET_VALID_INCIDENCES] No valid incidences specified. Using default valid incidences.")
            valid_incidences = np.array([0]) + list(range(5, 17))
            valid_incidences = np.array(valid_incidences) 

        # Arm valid incidences
        elif valid_incidences is None and execute_points == "ARM":
            valid_incidences = np.array([0]) + [5, 6, 7, 8, 9, 10]   # Nose, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Wrist, Right Wrist
            valid_incidences = np.array(valid_incidences)

        # Shoulder valid incidences
        elif valid_incidences is None and execute_points == "SHOULDER":
            valid_incidences = np.array([0]) + [5, 6, 7, 8, 9, 10]  # Nose, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Wrist, Right Wrist
            valid_incidences = np.array(valid_incidences)

        # Knee valid incidences
        elif valid_incidences is None and execute_points == "KNEE":
            valid_incidences = np.array([0]) + [11, 12, 13, 14, 15, 16]  # Nose, Left Hip, Right Hip, Left Knee, Right Knee, Left Ankle, Right Ankle
            valid_incidences = np.array(valid_incidences)

        # Thighs valid incidences
        elif valid_incidences is None and execute_points == "THIGHS":
            valid_incidences = np.array([0]) + [11, 12, 13, 14]  # Nose, Left Hip, Right Hip, Left Knee, Right Knee
            valid_incidences = np.array(valid_incidences)

        key_match_incidences = []
        for idx, key in enumerate(skeleton.keys()):
            if idx in valid_incidences: key_match_incidences.append(key)

        valid_incidences = []
        for key in key_match_incidences:
            valid_incidences.append(skeleton[key])

        valid_incidences = torch.Tensor(valid_incidences)
        return valid_incidences