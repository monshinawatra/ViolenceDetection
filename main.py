from typing import Union
from fastapi import FastAPI
import cv2

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/frame/{path}{no_frame}")
def get_features_list(path: str, no_frame: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, no_frame)
    ret, frame = cap.read()
    if ret:
        return {"frame" : frame}
    return {"frame" : None}
