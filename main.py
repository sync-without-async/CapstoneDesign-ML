from fastapi import FastAPI, File, UploadFile

from connector import database_connector, database_query

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
async def registerVideo(video_file: UploadFile = File(...), score_threshold: float = 0.8):
    video = video_file.file.read()
    video = io.BytesIO(video)
    with open(DUMMY_VIDEO_FILE_NAME, "wb") as f: f.write(video.read())

    video_tensor = cv2.VideoCapture(DUMMY_VIDEO_FILE_NAME)
    skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    if skeletons == "NO SKELETONS FOUND":
        return {"error": "No skeletons found"}

    return {"skeletons": skeletons}

@app.get("/getMetricsConsumer")
async def getMetricsConsumer(
    # video_file: UploadFile = File(...), 
    # score_threshold: float = 0.8
):
    # video = video_file.file.read()
    # video = io.BytesIO(video)
    # with open(DUMMY_VIDEO_FILE_NAME, "wb") as f: f.write(video.read())

    # video_tensor = cv2.VideoCapture(DUMMY_VIDEO_FILE_NAME)
    # skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    connector, cursor = database_connector(database_secret_path="secret_key.json")
    query = "SELECT * FROM metrics_consumer;"
    result = database_query(connector, cursor, query, verbose=True)

    return {
        "result": result
    }