import cv2
import os
import random
import numpy as np
from keras.models import Model, load_model
from keras.applications import VGG16
from collections import deque
import pandas as pd
import argparse
from tqdm import tqdm
from get_model import get_model


def get_resized_img(img, 
                    width: int=224, height: int=224, 
                    normalized: bool=True, predicted: bool=True, 
                    model=None):
    resized_img = cv2.resize(img, (width, height))

    if normalized: resized_img = resized_img.astype(np.float32)/255.
    if predicted: 
        prediction = model.predict(np.expand_dims(resized_img, 0), verbose=0)
        return prediction
    return resized_img


def pred_video(vid_path, image_model, lstm_model):
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    images = deque()
    class_index = 2
    prob = 0.

    violence_frequency = []

    if args.write_vid != None:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        r = os.path.dirname(args.write_vid)
        if not os.path.exists(r):
            os.makedirs(r)

        out = cv2.VideoWriter(args.write_vid, 
                              cv2.VideoWriter_fourcc('M','J','P','G'), 
                              fps, (frame_width,frame_height))


    for no_frame in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        images.append(get_resized_img(frame, model=image_model)[0])


        if len(images) == SEQUENCE_LENGHT:
            seq = np.asarray(images).reshape(1, 20, 4096)

            pred = lstm_model.predict(seq, verbose=0)
            class_index = np.argmax(pred[0])
            prob = pred[0, class_index]

            images.popleft() # Remove first frame
            dict_prob = prob
            if class_index == 0:
                dict_prob = 1 - dict_prob
                
            violence_data = {'no_frame': no_frame, 'violence': dict_prob}
            violence_frequency.append(violence_data)
        
        cv2.putText(frame, f"{CLASSES[class_index]} {'{:.2f}'.format(prob * 100)}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.write_vid != None: out.write(frame)

    cap.release()
    if args.write_vid != None: out.release()
    cv2.destroyAllWindows()

    if args.write_csv != None:
        df = pd.DataFrame(violence_frequency).set_index('no_frame')

        r = os.path.dirname(args.write_csv)
        if not os.path.exists(r):
            os.makedirs(r)

        if os.path.exists(args.write_csv):
            os.remove(args.write_csv)

        df.to_csv(args.write_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='dataset/Test/Violence/NV_353.mp4', help='')
    parser.add_argument('--random', action='store_true')

    parser.add_argument('--write-csv', type=str, help='')
    parser.add_argument('--write-vid', type=str, help='')
    args = parser.parse_args()


    # Variables setting
    SEQUENCE_LENGHT = 20
    VIDEO_PATH = 'dataset/Test/Violence'
    CLASSES = ['NonViolence', 'Violence', '']


    # Load and setup model
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model = Model(inputs=image_model.input,
                        outputs=transfer_layer.output)
    #model_ = load_model('model/vggLSTMv4/modelv4_4.h5')

    model_ = get_model()
    model_.load_weights('model/vggLSTMv4/w_modelv4_4.h5')

    # Select video
    if args.random == True:
        video_list = os.listdir(args.dir)
        random_video = random.choice(video_list)
        selected_video = os.path.join(VIDEO_PATH, random_video)
    else:
        selected_video = args.dir


    os.system('cls||clear')

    print('Processing..')
    print('- directory', args.dir)
    if args.random: print('- random -> ', random_video) 
    if args.write_vid != None: 
        print('- write video -> ', args.write_vid)
        args.write_vid = args.write_vid.replace("'", '').replace('"', '')
    if args.write_csv != None: 
        print('- write csv -> ', args.write_csv)
        args.write_csv = args.write_csv.replace("'", '').replace('"', '')
    
    pred_video(selected_video, image_model, model_)