from sklearn.metrics import jaccard_score

from models import SkeletonExtractor, DataPreprocessing

from connector import database_connector, database_query
from fastapi import FastAPI, File, UploadFile

import numpy as np
import json
import os

DUMMY_VIDEO_FILE_NAME = "dummy.webm"

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='mps')
preprocessor = DataPreprocessing()

@app.get("/videoRegister")
async def registerVideo(video_file: UploadFile = File(...), score_threshold: float = 0.5):
    video_tensor, video_length = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    print(f"video_length: {video_length}")

    # TODO: We will insert the skeletons to the database.
    with open("dummy_skeletons.json", "w") as f: json.dump(skeletons, f)

    os.remove(DUMMY_VIDEO_FILE_NAME)
    if skeletons == "NO SKELETONS FOUND":   return {"error": "No skeletons found"}

    return {
        "skeletons": skeletons,
        "video_length": video_length,
    }

@app.get("/getMetricsConsumer")
async def getMetricsConsumer(video_file: UploadFile = File(...), score_threshold: float = 0.5, vno: int = 0):
    # TODO: We will get the guide skeleton from the database.
    # Database currently, shuting down. So we will try it as simulation.
    # connector, cursor = database_connector(database_secret_path="secret_key.json")
    # table_name = "program_video"
    # query = f"SELECT * FROM {table_name} WHERE vno={vno};"
    # result = database_query(connector, cursor, query, verbose=True)

    # Currently, database has no data. So we will use dummy point.
    # cut_point = result[0][2]
    cut_point = 24 * 15 # 15 seconds

    video_tensor, video_length = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)[:cut_point]
    skeletons = extractor.extract(video_tensor, score_threshold=score_threshold)

    os.remove(DUMMY_VIDEO_FILE_NAME)

    # TODO: We will return the result of the query.
    # As a dummy example, we will return the first row of the table. But now database has shutdown.
    with open("dummy_skeletons.json", "r") as f:
        guide_skeleton = json.load(f)

    # Below code will be also used in the database query.
    guide_skeleton_values = np.array(list(guide_skeleton.values())).flatten()
    consumer_skeleton_values = np.array(list(skeletons.values())).flatten()

    metrics = jaccard_score(guide_skeleton_values, consumer_skeleton_values, average='micro')

    return {"metrics": metrics}