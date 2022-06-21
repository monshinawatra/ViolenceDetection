import cv2
import numpy as np
from keras.models import Model, load_model
from keras.applications import VGG16
from get_model import get_model

image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
CLASSES = ['NonViolence', 'Violence', '']
# model_ = load_model('model/vggLSTMv4/modelv4_5.h5')
model_ = get_model()
model_.load_weights('model/vggLSTMv4/w_model_weightsv4_5.h5')


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
        pred = model_.predict(np.expand_dims(pred_list, 0), verbose=0)
        class_index = np.argmax(pred[0])
        prob = pred[0, class_index]
    return prob, class_index


import time
def get_features_list(vid_path: str, 
                      start: int , 
                      stop: int):
    time_start = time.time()
    images = []
    cap = cv2.VideoCapture(vid_path)
    
    for i in range(start, stop):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float64)/255.
        prediction = image_model_transfer.predict(np.expand_dims(img, 0), verbose=0)
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