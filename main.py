from sklearn.metrics import jaccard_score

from models import SkeletonExtractor, DataPreprocessing, Metrics

from connector import database_connector, database_query
from fastapi import FastAPI, File, UploadFile

import matplotlib.pyplot as plt
import skvideo.io as skvideo
import numpy as np
import json
import os

DUMMY_VIDEO_FILE_NAME = "dummy.webm"
EXTRACTOR_THRESHOLD = 0.85

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='mps')
preprocessor = DataPreprocessing()
metrics = Metrics()

os.system("export PYTORCH_ENABLE_MPS_FALLBACK=1")

@app.get("/videoRegister")
async def registerVideo(video_file: UploadFile = File(...)):
    print(f"[INFO/REGISTER] Video register request has been received.")
    print(f"[INFO/REGISTER] Extractor threshold: {EXTRACTOR_THRESHOLD}")

    video_tensor = preprocessor.processing(video_file=video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons, video_length = extractor.extract(video_tensor, score_threshold=EXTRACTOR_THRESHOLD)

    print(f"[INFO/REGISTER] Video length: {video_length}")
    with open("dummy_skeletons.json", "w") as f: json.dump(skeletons, f)

    return {
        "skeletons": skeletons,
        "video_length": video_length,
    }

@app.get("/getMetricsConsumer")
async def getMetricsConsumer(video_file: UploadFile = File(...), vno: int = 0):
    # TODO: We will get the guide skeleton from the database.
    # Database currently, shuting down. So we will try it as simulation.
    # connector, cursor = database_connector(database_secret_path="secret_key.json")
    # table_name = "program_video"
    # query = f"SELECT * FROM {table_name} WHERE vno={vno};"
    # result = database_query(connector, cursor, query, verbose=True)

    # Currently, database has no data. So we will use dummy point.
    # cut_point = result[0][2]

    video_tensor = preprocessor.processing(video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
    skeletons, video_length = extractor.extract(video_tensor, score_threshold=EXTRACTOR_THRESHOLD)

    # TODO: We will return the result of the query.
    # As a dummy example, we will return the first row of the table. But now database has shutdown.
    with open("dummy_skeletons.json", "r") as f:
        guide_skeleton = json.load(f)

    # Below code will be also used in the database query.
    guide_skeleton_values = np.array(list(guide_skeleton.values())).flatten()
    consumer_skeleton_values = np.array(list(skeletons.values())).flatten()
    cut_index = np.min([len(guide_skeleton_values), len(consumer_skeleton_values)])

    score = metrics.score(guide_skeleton_values[:cut_index], consumer_skeleton_values[:cut_index])
    return {"metrics": score}
