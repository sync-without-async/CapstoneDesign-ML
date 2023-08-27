from sklearn.metrics import jaccard_score

from models import SkeletonExtractor, DataPreprocessing

from connector import database_connector, database_query
from fastapi import FastAPI, File, UploadFile

import numpy as np
import json
import os

DUMMY_VIDEO_FILE_NAME = "dummy.webm"

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='cpu')
preprocessor = DataPreprocessing()

@app.get("/videoRegister")
async def registerVideo(video_file: UploadFile = File(...), score_threshold: float = 0.8):
    video_tensor = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    # TODO: We will insert the skeletons to the database.
    with open("dummy_skeletons.json", "w") as f: json.dump(skeletons, f)

    os.remove(DUMMY_VIDEO_FILE_NAME)
    if skeletons == "NO SKELETONS FOUND":   return {"error": "No skeletons found"}

    return {"skeletons": skeletons}

@app.get("/getMetricsConsumer")
async def getMetricsConsumer(video_file: UploadFile = File(...), score_threshold: float = 0.8, vno: int = 0):
    video_tensor = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    os.remove(DUMMY_VIDEO_FILE_NAME)
    
    # Database currently, shuting down. So we will try it as simulation.
    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name = "program_video"
    query = f"SELECT * FROM {table_name} WHERE vno={vno};"
    result = database_query(connector, cursor, query, verbose=True)
    print(result)

    # TODO: We will return the result of the query.
    # As a dummy example, we will return the first row of the table. But now database has shutdown.
    with open("dummy_skeletons.json", "r") as f:
        guide_skeleton = json.load(f)

    # Below code will be also used in the database query.
    guide_skeleton_values = np.array(list(guide_skeleton.values())).flatten()
    consumer_skeleton_values = np.array(list(skeletons.values())).flatten()

    metrics = jaccard_score(guide_skeleton_values, consumer_skeleton_values, average='micro')

    return {"metrics": metrics}