import cv2
import numpy as np
from keras.models import Model
from keras.applications import VGG16
from collections import deque
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Activation
from threading import Thread
image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
CLASSES = ['NonViolence', 'Violence', '']

def get_lstm_model():
    model = Sequential()
    model.add(LSTM(512, input_shape=(20, 4096)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


model_ = get_lstm_model()
model_.load_weights('model/vggLSTMv4_2/model_weightsv4_2.h5')


def get_frame(vid_path: str, frame: int):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def get_total_frame(vid_path: str):
    cap = cv2.VideoCapture(vid_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(total_frames)


def predict(pred_list: list):
    if len(pred_list) == 20:
        pred = model_.predict(np.expand_dims(pred_list, 0))
        class_index = np.argmax(pred[0])
        prob = pred[0, class_index]
    return prob, class_index

import time
def get_features_list(vid_path: str, 
                      start: int , 
                      stop: int):
    time_start = time.time()
    images = deque()
    cap = cv2.VideoCapture(vid_path)
    c = start
    for _ in range(start, stop):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float64)/255.
        prediction = image_model_transfer.predict(np.expand_dims(img, 0))
        images.append(prediction[0])

    cap.release()
    print("time used ", time.time() - time_start)
    return np.array(images)


def get_frames(vid_path: str, start: int, stop: int):
    frames = []
    count = start
    while count < stop:
        frames.append(get_frame(vid_path, count))
        count += 1
    return frames

def raw_predict(frames: list):
    raw = get_features_list(frames)
    return predict(raw)
    
