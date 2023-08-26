from fastapi import FastAPI, Request, File, UploadFile
from tempfile import SpooledTemporaryFile
from pydantic import BaseModel

import skvideo.io as skvideo
import numpy as np
import cv2
import io

from models import SkeletonExtractor

DUMMY_VIDEO_FILE_NAME = "dummy.webm"

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='cpu')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/videoRegister")
async def registerVideo(video_file: UploadFile = File(...)):
    video = video_file.file.read()
    video = io.BytesIO(video)
    with open(DUMMY_VIDEO_FILE_NAME, "wb") as f: f.write(video.read())

    video_tensor = cv2.VideoCapture(DUMMY_VIDEO_FILE_NAME)
    skeletons = extractor.extract(video_tensor, score_threshold=0.5)

    if skeletons == "NO SKELETONS FOUND":
        return {"error": "No skeletons found"}

    return {"skeletons": skeletons}