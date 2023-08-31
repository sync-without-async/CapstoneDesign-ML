from fastapi.testclient import TestClient
from fastapi import FastAPI

from main import app

import logging

TEST_VIDEO_PATH = "videos/consumer/guide_1_1sec.mp4"
TEST_VIDEO_NAME = "guide_1_1sec.mp4"

client = TestClient(app)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def test_registerVideo():
    request = {
        "video_file": open(TEST_VIDEO_PATH, "rb")
    }

    response = client.get(
        "/videoRegister",
        params=request
    )

    assert response.status_code == 200
    assert response.json()["skeletons"] != None
    assert response.json()["video_length"] != None

def test_getMetricsConsumer():
    request = {
        "video_file": open(TEST_VIDEO_PATH, "rb"),
        "vno": 0
    }

    response = client.get(
        url="/getMetricsConsumer", 
        params=request
    )
    assert response.status_code == 200
    assert response.json()["metrics"] != None