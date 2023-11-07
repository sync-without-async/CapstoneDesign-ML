# ReHab Machine Learning - Pose Estimation & Audio Summarization

This repository is the artificial intelligence repository for the ReHab project. Artificial intelligence is a crucial element in the project, as it provides essential services and methods for guiding users in performing exercises. Through artificial intelligence, we offer guidance videos and provide users with a way to perform exercises. We evaluate how well the user is doing by measuring similarity through feature extraction and cosine similarity using the videos provided by the user.

We utilize pre-trained models for our system. The baseline model employs Posenet, and this choice might change based on considerations such as the trade-off between communication overhead and computation overhead.

Furthermore, we have implemented the patient and doctor counseling feature through WebRTC. You can check out this functionality in the [Backend](https://github.com/sync-without-async/Rehab-BackEnd) and [Frontend](https://github.com/sync-without-async/Rehab-FrontEnd). We've also added an AI capability that summarizes the counseling content. The original repository can be found at [Rehab-Audio](https://github.com/sync-without-async/Rehab-Audio). The feature development is complete, and we have migrated it to this repository.

## Requirements

This code requires a set of essential modules to build an API server using FastAPI, run artificial intelligence processes with torch and torchvision, access databases using mysql_connector, transform and utilize uploaded videos using scikit-video, numpy, and openCV. It also utilizes the request and json modules for fetching files from Naver Cloud Object. Additionally, internal utility modules and methods exist, so `main.py` necessitates `utils.py`, `models.py`, `connector.py`, `summary.py` and `speech_to_text.py`.

The summarized requirements are as follows:

- torch (>= 2.0.0)
- torchvision (>= 0.15.0)
- numpy (>= 1.23.5)
- skvideo (>= 1.1.11)
- cv2 (>= 4.8.0)
- fastapi (>= 0.100.0)
- polars (>= 0.17.7)
- mysql.connector (>= 8.1.0)
- transformers (>= 1.89.1)
- openai (>= 0.28.1)

Please note that the `requirements.txt` hasn't been separately written due to the numerous modules used for personal experimentation and development within the current environment. Your understanding is appreciated.

The `requirements_jetson.txt` and `requirements_denoiser.txt` contain the Python module installation instructions for Nvidia Jetson Nano. These instructions are specific to Nvidia Jetson Nano and may involve different Python versions or module versions compared to a typical PC. These files are separated for use in creating a Docker image on Nvidia Jetson Nano via a Dockerfile. There is no need to install these modules manually; they are intended for use in the Docker image creation process.

## Nvidia Jetson Nano Installation
Our service is deployed on Nvidia Jetson Nano using Docker. To deploy on Nvidia Jetson Nano, you can create an image using a Dockerfile as follows:

```bash
sudo docker build -t <DOCKER_IMAGE_NAME>:<DOCKER_IMAGE_TAG> .
```

Once you've built the Docker image, you can create a container from it. Here's how to run the created container:

```bash
docker run -d --rm --runtime nvidia --network host <DOCKER_IMAGE_NAME>:<DOCKER_IMAGE_TAG>
```

- `-d` stands for daemon and is used to run the container in the background.
- `--rm` is used to automatically remove the container when it's stopped to save storage space.
- `--runtime` specifies the runtime to be used, and it needs to be set for CUDA support on Jetson Nano.
- `--network` is used to specify how the container should be networked. Setting it to `host` means the container shares the network with the host device.

When you run it as a daemon, only the Docker container's full ID will be printed, and there shouldn't be any issues. However, if you encounter any issues during image build or container run, please leave a description of the issue for further assistance.

## Quick Start

To set up the FastAPI server, it's essential to install [Uvicorn](https://www.uvicorn.org/) first. After installation, you can run the server using the following command in the directory containing `main.py`:

```bash
$ uvicorn main:app --host 0.0.0.0 --port 8080
INFO:     Started server process [60991]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

If you plan on making modifications and building the API server iteratively, you can use the following command:

```bash
$ uvicorn main:app --host 0.0.0.0 --port 8080 --reload
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
