# ReHab Machine Learning - Pose Estimation

This repository is the artificial intelligence repository for the ReHab project. Artificial intelligence is a crucial element in the project, as it provides essential services and methods for guiding users in performing exercises. Through artificial intelligence, we offer guidance videos and provide users with a way to perform exercises. We evaluate how well the user is doing by measuring similarity through feature extraction and cosine similarity using the videos provided by the user.

We utilize pre-trained models for our system. The baseline model employs Posenet, and this choice might change based on considerations such as the trade-off between communication overhead and computation overhead.

## Installation



## Our Model

The `torchvision.models` module in PyTorch provides various pre-trained and state-of-the-art model architectures. Since extracting human poses from images is crucial, we have used the Keypoint RCNN ResNet50 FPN-based model, which is capable of extracting keypoints. Understanding it as a structure comprising Keypoint R-CNN + ResNet50 FPN makes it easier to comprehend.

As mentioned in the [official documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn), the default weights are from a model trained on the COCO Dataset v1. This model has more parameters compared to legacy models, though GFLOPs are reduced, resulting in an improved performance model.
