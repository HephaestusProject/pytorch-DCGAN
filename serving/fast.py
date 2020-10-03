import config
import torch
from fastapi import FastAPI

from src.model import Generator

app = FastAPI()


def prediction():
    pass
    # model load
    # generate images
    # return images


@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/predict")
def predict():
    pred = prediction()
    return {"message": "Complete"}
