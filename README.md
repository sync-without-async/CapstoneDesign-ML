# ReHab Machine Learning - Pose Estimation

This repository is the artificial intelligence repository for the ReHab project. Artificial intelligence is a crucial element in the project, as it provides essential services and methods for guiding users in performing exercises. Through artificial intelligence, we offer guidance videos and provide users with a way to perform exercises. We evaluate how well the user is doing by measuring similarity through feature extraction and cosine similarity using the videos provided by the user.

We utilize pre-trained models for our system. The baseline model employs Posenet, and this choice might change based on considerations such as the trade-off between communication overhead and computation overhead.

## Requirements

This code requires a set of essential modules to build an API server using FastAPI, run artificial intelligence processes with torch and torchvision, access databases using mysql_connector, transform and utilize uploaded videos using scikit-video, numpy, and openCV. It also utilizes the request and json modules for fetching files from Naver Cloud Object. Additionally, internal utility modules and methods exist, so `main.py` necessitates `utils.py`, `models.py`, and `connector.py`.

The summarized requirements are as follows:

- torch (>= 2.0.0)
- torchvision (>= 0.15.0)
- numpy (>= 1.23.5)
- skvideo (>= 1.1.11)
- cv2 (>= 4.8.0)
- fastapi (>= 0.100.0)
- polars (>= 0.17.7)
- mysql.connector (>= 8.1.0)

Please note that the `requirements.txt` hasn't been separately written due to the numerous modules used for personal experimentation and development within the current environment. Your understanding is appreciated.

## Quick Start

To set up the FastAPI server, it's essential to install [Uvicorn](https://www.uvicorn.org/) first. After installation, you can run the server using the following command in the directory containing `main.py`:

```bash
$ uvicorn main:app --host 127.0.0.1 --port 8080
INFO:     Started server process [60991]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

If you plan on making modifications and building the API server iteratively, you can use the following command:

```bash
$ uvicorn main:app --host 127.0.0.1 --port 8080 --reload
INFO:     Will watch for changes in these directories: ['path/to/ReHab-ML']
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
INFO:     Started reloader process [61155] using StatReload
INFO:     Started server process [61157]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

The `--reload` parameter automatically restarts the server whenever there's a change in the internal code and it's saved. However, if the server is in the process of preparing a response after a request, it might restart after responding, so keep that in mind.

## Our Model

The `torchvision.models` module in PyTorch provides various pre-trained and state-of-the-art model architectures. Since extracting human poses from images is crucial, we have used the Keypoint RCNN ResNet50 FPN-based model, which is capable of extracting keypoints. Understanding it as a structure comprising Keypoint R-CNN + ResNet50 FPN makes it easier to comprehend.

As mentioned in the [official documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html#torchvision.models.detection.keypointrcnn_resnet50_fpn), the default weights are from a model trained on the COCO Dataset v1. This model has more parameters compared to legacy models, though GFLOPs are reduced, resulting in an improved performance model.
