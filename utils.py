import cv2
import numpy as np
from keras.models import Model, load_model
from keras.applications import VGG16
from collections import deque


image_model = VGG16(include_top=True, weights='imagenet')
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)
model_ = load_model('model/vggLSTMv4_2/modelv4_2.h5')
CLASSES = ['NonViolence', 'Violence', '']


def get_frame(vid_path, frame):
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def get_total_frame(vid_path):
    cap = cv2.VideoCapture(vid_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(total_frames)


def predict(pred_list: list, cur_frame):

    # for frame in frames:
    #     img = cv2.resize(frame, (224, 224))
    #     img = img.astype(np.float32)/255.
    #     prediction = image_model_transfer.predict(np.expand_dims(img, 0))
    #     images.append(prediction[0])

    if len(pred_list) == 20:
        #seq = np.asarray(images).reshape(1, 20, 4096)
        pred = model_.predict(np.expand_dims(pred_list, 0))
        class_index = np.argmax(pred[0])
        prob = pred[0, class_index]

    #cv2.putText(cur_frame, f"{CLASSES[class_index]} {'{:.2f}'.format(prob * 100)}%", (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return prob, class_index


def get_features_list(frames):
    images = deque()
    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img = img.astype(np.float32)/255.
        prediction = image_model_transfer.predict(np.expand_dims(img, 0))
        images.append(prediction[0])
    return np.array(images)
