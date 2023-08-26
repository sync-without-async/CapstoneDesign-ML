# Reference:
# https://debuggercafe.com/human-pose-detection-using-pytorch-keypoint-rcnn/

import matplotlib.pyplot as plt
import numpy as np
import utils
import time
import cv2

from torchvision.models.detection import keypointrcnn_resnet50_fpn

import torchvision
import torch

device = torch.device('cpu')
model = keypointrcnn_resnet50_fpn(pretrained=True, num_keypoints=17)
model.to(device).eval()

consumer_video = cv2.VideoCapture("videos/guide/guide_1.mp4")
consumer_video_width = int(consumer_video.get(3))
consumer_video_height = int(consumer_video.get(4))

# saved_path = f"videos/guide/guide_1_keypoints.mp4"
# out = cv2.VideoWriter(
#     saved_path,
#     cv2.VideoWriter_fourcc(*'mp4v'),
#     20,
#     (consumer_video_width, consumer_video_height)
# )

frame_count = 0
total_fps = 0

while consumer_video.isOpened():
    ret, frame = consumer_video.read()
    if ret == True:
        image = np.array(frame, dtype=np.float32) / 255.0
        original_image = image.copy()

        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            prediction = model(image)
        end_time = time.time()

        keypoints = utils.draw_keypoints(prediction, original_image)

        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1
        
        wait_time = max(1, int(fps/4))
        cv2.imshow("Keypoints", keypoints)
        # out.write(keypoints)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break

consumer_video.release()
# out.release()
cv2.destroyAllWindows()

print(f"Average FPS: {total_fps/frame_count}")