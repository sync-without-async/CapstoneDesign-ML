from models import SkeletonExtractor, DataPreprocessing, Metrics, MMPoseStyleSimilarty

from connector import database_connector, database_query, database_select_using_pk, insert_summary_database
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError

from pydub import AudioSegment

import pandas as pd
import torch

import speech_to_text as stt
import denoising as den
import summary 

import requests
import logging
import json
import os

DUMMY_VIDEO_FILE_NAME = "dummy.webm"
EXTRACTOR_THRESHOLD = 0.85

app = FastAPI()
extractor = SkeletonExtractor(pretrained_bool=True, number_of_keypoints=17, device='cuda')
preprocessor = DataPreprocessing()
metrics = Metrics()
mmpose_similarity = MMPoseStyleSimilarty()

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

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=500)

@app.get("/")
async def root():
    return {"message": "rehab_ai_server_api_success"}

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
    skeletons, cropped_skeletons, video_length = extractor.extract(video_tensor=video_tensor, score_threshold=EXTRACTOR_THRESHOLD, video_length=None)

    extracted_skeleton_json = {
        "skeletons": skeletons,
        "cropped_skeletons": cropped_skeletons,
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
    testing_flag = False
    print(f"[INFO/GETMETRICS] Video get metrics request has been received.")
    print(f"[INFO/GETMETRICS] VNO: {vno}")
    
    if not testing_flag:
        # Below code will be also used in the database query. 
        connector, cursor = database_connector(database_secret_path="secret_key.json")
        table_name = "video"
        query = f"SELECT * FROM {table_name};"
        result = database_query(connector, cursor, query, verbose=False)

        print(f"[INFO/GETMETRICS] Database query: {query}")
        print(f"[INFO/GETMETRICS] Database result: {result}")

        if result.shape[0] == 0:    return {"error": "No query found in database."}

        # Check if the video number is in the database. 
        vno_list = result.iloc[:, 0].tolist()
        if not vno in vno_list:     return {"error": "No video number found in database."}
        vno = vno_list.index(vno)

        json_url = result.iloc[vno, 6]
        print(json_url)
        response = requests.get(json_url)
        guide_skeleton = json.loads(response.text)

        # Below code will be also used in the database query.
        # JSON URL is the 8th column of the table. VNO is the user selected video number.
        # Extact consumer's skeleton.
        video_tensor, video_height, video_width = preprocessor.processing(video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)

        guide_video_height = json.loads(response.text)['video_heigth']
        guide_video_width = json.loads(response.text)['video_width']
        video_cut_point = result.iloc[vno, 4]

        video_target = result.iloc[vno, 8]
    
    else:
        with open("extracted_skeleton.json", "r") as f:
            guide_skeleton = json.load(f)
        guide_video_width, guide_video_height = guide_skeleton['video_width'], guide_skeleton['video_heigth']
        video_lenght = guide_skeleton['video_length']
        video_target = "SHOULDER"

        video_tensor, video_height, video_width = preprocessor.processing(video_file, temp_video_file_path=DUMMY_VIDEO_FILE_NAME)
        video_cut_point = video_lenght

    print(f"[INFO/GETMETRICS] Testing flag: {testing_flag}")

    cropped_skeletons, skeletons, frame_count = extractor.extract(video_tensor=video_tensor, score_threshold=EXTRACTOR_THRESHOLD, video_length=video_cut_point)

    # Check if the video cut point is in the database.
    if video_cut_point >= frame_count:  video_cut_point = frame_count
    logging.info(f"[INFO/GETMETRICS] Video cut point: {video_cut_point}")

    print(guide_skeleton)
    print(skeletons)

    score = mmpose_similarity.score(
        guide_skeleton=guide_skeleton['skeletons'], 
        consumer_skeleton=skeletons,
        execrise_points=video_target,
    )

    # score = metrics.score(
    #     y_true=guide_skeleton['cropped_skeletons'],
    #     true_video_height=guide_video_height,
    #     true_video_width=guide_video_width,
    #     true_cut_point=video_cut_point,
    #     y_pred=cropped_skeletons,
    #     pred_video_height=video_height,
    #     pred_video_width=video_width,
    # )

    logging.info(f"[INFO/GETMETRICS] Score Metrics: {score}")

    return {"metrics": score}

@app.get("/getSummary")
async def getSummary(ano: int, 
                     background_tasks: BackgroundTasks = BackgroundTasks()
    ):
    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name = "audio"
    query = f"SELECT * FROM {table_name}"

    table = pd.DataFrame(database_query(connector, cursor, query, verbose=True), index=None)
    print(table) 
    result = database_select_using_pk(
        table=table,
        pk=ano,
        verbose=True
    )
    print(result)
    print(result.to_numpy().tolist())

    result = result.to_numpy().tolist()[0]

    try:
        doctor_audio_url, patient_audio_url = result[2], result[5]
        doctor_audio = requests.get(doctor_audio_url).content
        patient_audio = requests.get(patient_audio_url).content

        with open("doctor.wav", "+wb") as f:     f.write(doctor_audio)
        with open("patient.wav", "+wb") as f:    f.write(patient_audio)
        
        doctor_audio, doc_fs = den.load_audio("doctor.wav")
        patient_audio, pat_fs = den.load_audio("patient.wav")

        background_tasks.add_task(
            _do_summary,
            ano=ano,
            doctor_audio=doctor_audio,
            patient_audio=patient_audio,
            doc_fs=doc_fs,
            pat_fs=pat_fs,
        )

        return True

    except Exception as e:
        logging.error("[SUMMARY_MODULE] Error occured while getting audio from database.")
        logging.error(e)
        return False

async def _do_summary(
        ano: int, 
        doctor_audio: torch.Tensor,
        patient_audio: torch.Tensor,
        doc_fs: int,
        pat_fs: int,
    ):
    logging.info("[DSR_MODULE] Denoising audio...")
    doctor_audio, doc_sr = den.denoising(
        audio=doctor_audio,
        sample_rate=doc_fs,
        device="cpu",
        verbose=True
    )

    patient_audio, pat_sr = den.denoising(
        audio=patient_audio,
        sample_rate=pat_fs,
        device="cpu",
        verbose=True
    )

    logging.info("[DSR_MODULE] Transcribing audio...") 
    doctor_transcript = stt.speech_to_text(
        processor_pretrained_argument="kresnik/wav2vec2-large-xlsr-korean",
        audio=doctor_audio,
        audio_sample_rate=doc_sr,
        device="cpu",
        verbose=True
    )

    patient_transcript = stt.speech_to_text(
        processor_pretrained_argument="kresnik/wav2vec2-large-xlsr-korean",
        audio=patient_audio,
        audio_sample_rate=pat_sr,
        device="cpu",
        verbose=True
    )

    # TODO: Get summary
    summarized = summary.summarize(
        doctor_content=doctor_transcript,
        patient_content=patient_transcript,
        max_tokens=700,
        verbose=True,
    )

    connector, cursor = database_connector(database_secret_path="secret_key.json")
    table_name, table_column = "audio", "summary"

    _ = insert_summary_database(
        connector=connector,
        cursor=cursor,
        target_table_name=table_name,
        target_columns=table_column,
        target_values=summarized,
        target_room_number=ano,
        verbose=True,
    )

    logging.info("[SUMMARY_MODULE] Summary has been saved in the database.")
    logging.info("[SUMMARY_MODULE] Summary: ")
    logging.info(summarized)

def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)