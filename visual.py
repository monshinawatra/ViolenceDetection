import cv2
import os
import random
import numpy as np
from keras.models import Model, load_model
from keras.applications import VGG16
from collections import deque
import pandas as pd

def get_resized_img(img, 
                    width: int=224, height: int=224, 
                    normalized: bool=True, predicted: bool=True, 
                    model=None):
    resized_img = cv2.resize(img, (width, height))
    
    if normalized: resized_img = resized_img.astype(np.float32)/255.
    if predicted: 
        prediction = model.predict(np.expand_dims(resized_img, 0))
        return prediction
    return resized_img


def pred_video(vid_path, image_model, lstm_model, write_csv: bool=True):
    cap = cv2.VideoCapture(vid_path)
    no_frame = 0
    images = deque()
    class_index = 2
    prob = 0.
    
    violence_frequency = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        images.append(get_resized_img(frame, model=image_model)[0])

        
        if len(images) == SEQUENCE_LENGHT:
            seq = np.asarray(images).reshape(1, 20, 4096)

            pred = lstm_model.predict(seq)
            class_index = np.argmax(pred[0])
            prob = pred[0, class_index]
            
            images.popleft() # Remove first frame
            violence_data = {'no_frame': no_frame, 'violence': prob}
            violence_frequency.append(violence_data)
        

        cv2.putText(frame, f"{CLASSES[class_index]} {'{:.2f}'.format(prob * 100)}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) and 0xff == ord('q'):
            break

        no_frame += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    if write_csv:
        df = pd.DataFrame(violence_frequency).set_index('no_frame')
        df.to_csv(f'output/data.csv')


if __name__ == "__main__":

    # Variables setting
    SEQUENCE_LENGHT = 20
    VIDEO_PATH = 'dataset/Test/Violence'
    CLASSES = ['NonViolence', 'Violence', '']
    


    # Load and setup model
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model = Model(inputs=image_model.input,
                        outputs=transfer_layer.output)
    model_ = load_model('model/vggLSTMv4_2/modelv4_2.h5')


    # Select video
    video_list = os.listdir(VIDEO_PATH)
    random_video = random.choice(video_list)
    selected_video = os.path.join(VIDEO_PATH, random_video)
    
    
    pred_video(selected_video, image_model, model_)