from .models import SkeletonExtractor, DataPreprocessing, Metrics

from .connector import database_connector, database_query
from fastapi import FastAPI, File, UploadFile, Form

from typing import Annotated

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import *

import skvideo.io as skvideo
import requests
import logging
import json
import os

DUMMY_VIDEO_FILE_NAME = "dummy.webm"
EXTRACTOR_THRESHOLD = 0.85

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='cpu')
preprocessor = DataPreprocessing()
metrics = Metrics()

os.system("export PYTORCH_ENABLE_MPS_FALLBACK=1")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handling Error
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=500)

@app.post("/videoRegister")
async def registerVideo(
    video_file: UploadFile = File(...)
):
    """On this function, we will extract the skeleton from the video that the consumer wants to follow.
    And then, we will save the skeleton as a JSON file. The JSON file will be saved in the database, CRUD not work on our layer (AI layer).
    After saved the data on the database, we will return the skeleton and the video length to the consumer.

    Args:
        video_file (UploadFile, optional): The video file that the consumer wants to follow. Defaults to File(...).

    Returns:
        skeletons(json, dict) : The skeleton and the video length.
        video_length(int) : The video length."""
    print(f"[INFO/REGISTER] Video register request has been received.")
    print(f"[INFO/REGISTER] Extractor threshold: {EXTRACTOR_THRESHOLD}")

    video_tensor, video_heigth, video_width = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons, video_length = extractor.extract(video_tensor=video_tensor, score_threshold=EXTRACTOR_THRESHOLD, video_length=None)

    extracted_skeleton_json = {
        "skeletons": skeletons,
        "video_length": video_length,
        "video_heigth": video_heigth,
        "video_width": video_width
    }

    with open("extracted_skeleton.json", "w") as f:
        json.dump(extracted_skeleton_json, f)

    return extracted_skeleton_json

@app.post("/getMetricsConsumer")
async def getMetricsConsumer(
    vno: int = Form(), video_file: UploadFile = File(...)
):
    """On this function, we will calculate the metrics between the consumer's skeleton and the guide's skeleton.
    Guide's skeleton is the skeleton that is extracted from the video that the consumer wants to follow. And the consumer's skeleton is the skeleton that is extracted from the consumer's video.
    Standard skeleton is the guide's skeleton, and the skeleton that already exists in the database.

    Args:
        video_file (UploadFile, optional): The video file that the consumer wants to follow. Defaults to File(...).
        vno (int, optional): The video number that the consumer wants to follow. Defaults to 0.

    Returns:
        float or dobuble: The metrics between the consumer's skeleton and the guide's skeleton."""
    testing_flag = True
    print(f"[INFO/GETMETRICS] Video get metrics request has been received.")
    print(f"[INFO/GETMETRICS] VNO: {vno}")
    
    if not testing_flag:
        # Below code will be also used in the database query. 
        connector, cursor = database_connector(database_secret_path="secret_key.json")
        table_name = "video"
        query = f"SELECT * FROM {table_name};"
        result = database_query(connector, cursor, query, verbose=False)
        if result.shape[0] == 0:    return {"error": "No query found in database."}

        # Check if the video number is in the database. 
        vno_list = result[:, 0].tolist()
        if not vno in vno_list:     return {"error": "No video number found in database."}
        vno = vno_list.index(vno)

        json_url = result[vno, 7]
        response = requests.get(json_url)
        guide_skeleton = json.loads(response.text)['skeletons']

        # Below code will be also used in the database query.
        # JSON URL is the 8th column of the table. VNO is the user selected video number.
        # Extact consumer's skeleton.
        video_tensor, video_height, video_width = preprocessor.processing(video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)

        guide_video_height = json.loads(response.text)['video_heigth']
        guide_video_width = json.loads(response.text)['video_width']
        video_cut_point = result[vno, 8]
    
    else:
        with open("extracted_skeleton.json", "r") as f:
            guide_skeleton = json.load(f)
        guide_video_width, guide_video_height = guide_skeleton['video_width'], guide_skeleton['video_heigth']
        video_lenght = guide_skeleton['video_length']

        video_tensor, video_height, video_width = preprocessor.processing(video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
        video_cut_point = video_lenght

    print(f"[INFO/GETMETRICS] Testing flag: {testing_flag}")

    skeletons, frame_count = extractor.extract(video_tensor=video_tensor, score_threshold=EXTRACTOR_THRESHOLD, video_length=video_cut_point)

    # Check if the video cut point is in the database.
    if video_cut_point >= frame_count:  video_cut_point = frame_count
    logging.info(f"[INFO/GETMETRICS] Video cut point: {video_cut_point}")
    
    # Calculate metrics 
    score = metrics.score(
        y_true=guide_skeleton,
        true_video_height=guide_video_height,
        true_video_width=guide_video_width,
        true_cut_point=video_cut_point,
        y_pred=skeletons,
        pred_video_height=video_height,
        pred_video_width=video_width
    )

    logging.info(f"[INFO/GETMETRICS] Score Metrics: {score}")

    return {"metrics": score}
