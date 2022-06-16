import streamlit as st
import cv2
import tempfile
import utilities as utl
from stqdm import stqdm
import numpy as np
import torch
import os
import h5py


def draw_bboxes(bboxes, cur_frame):
    people = [ (int(p['xmin']), int(p['ymin']),
            int(p['xmax']), int(p['ymax']))
            for p in
            [bboxes.pandas().xyxy[0].iloc[i, 0:4]
            for i in range(bboxes.pandas().xyxy[0].shape[0])]]
    
    c = tuple(int(st.session_state.box_color.replace('#', '')[i:i+2], 
                      16) for i in (0, 2, 4))

    for p in people:
        cv2.rectangle(cur_frame, 
                      (p[0], p[1]), (p[2], p[3]), 
                      (c[2], c[1], c[0]), 1)

    return cur_frame


def get_all_video_frames(vid_name: str):
    vid = vid_name
    prog = stqdm(range(utl.get_total_frame(vid)))
    prog.set_description_str("Getting all frames")

    frames_list = [utl.get_frame(vid, i) for i in prog]
    return frames_list


def get_frame(no_frame):
    with h5py.File('temp/data.h5', 'r') as f:
        frame_list = f['frames_list'][:]
        try:
            frame = frame_list[no_frame]
        except:
            st.session_state.frame = 20
            frame = frame_list[19]
    return frame



def probabiliy_label(prob: float, class_index: int):
    classes = {0: "NonViolence", 1: "Violence"}
    icons = {0: open('html/nonviolence_icon.txt', 'r').read(),
             1: open('html/violence_icon.txt', 'r').read()}
    
    first_icon = (icons[class_index], '{:.2f}'.format(prob * 100))
    second_icon = (icons[0], '{:.2f}'.format(100-(prob * 100))) \
        if class_index == 1 else (icons[1], '{:.2f}'.format(100-(prob * 100)))
    
    class_css = (f"""
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css" integrity="sha512-KfkfwYDsLkIlwQp6LFnl8zNdLGxu9YAA1QvwINks4PhcElQSvqcyVLLD9aMhXd13uQjoXtEKNosOWaZqXgel0g==" crossorigin="anonymous" referrerpolicy="no-referrer" />
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                <div class="container" style="padding: 0% 30%; text-align: center;">
                    <div class="row">
                        <div class="col-sm-6">
                            {first_icon[0]} {first_icon[1]}% </span>
                        </div>            
                        <div class="col-sm-6">
                            {second_icon[0]} {second_icon[1]}% </span>  
                        </div>
                    </div>
                </div>
                 """)
    progress_label.markdown(class_css, unsafe_allow_html=True)


def initial_video_frames():
    if video_file == None:
        return
    if st.session_state['video'] != video_file.name:
        st.session_state['video'] = video_file.name
        print('Change')
        create_framelist()
        
        
def update_frame():
    frame = get_frame(st.session_state.frame-1)
    temp_range = [st.session_state.frame-20, st.session_state.frame]
    
    with h5py.File('temp/data.h5', 'r') as file:
        temporal_frames = file['preds_list'][temp_range[0]: temp_range[1]]
    
    print('len temp', len(temporal_frames))

    prob, class_index = utl.predict(temporal_frames)
    probabiliy_label(prob, class_index)
    render_frame = frame.copy()
    
    if class_index == 1:
        results = detection_model(render_frame)
        render_frame = draw_bboxes(results, render_frame)
        # render_frame = np.squeeze(results.render())
    
    img_frame.image(cv2.cvtColor(render_frame, cv2.COLOR_BGR2RGB), use_column_width=True)


@st.cache(allow_output_mutation=True)
def get_detection_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def on_config_change():
    detection_config()
    update_frame()


def detection_config():
    detection_model.iou = st.session_state.iou
    detection_model.conf = st.session_state.conf
    detection_model.classes = [0]


def create_framelist():
    print('Create data')
    with h5py.File('temp/data.h5', 'w') as file:
        frames_list = get_all_video_frames(tffile.name)
        file.create_dataset('frames_list', 
                            data=frames_list) 
        file.create_dataset('preds_list', 
                            data=utl.get_features_list(frames_list)) 


def app_initial():
    if 'frame' not in st.session_state:
        st.session_state['frame'] = 20
    if 'conf' not in st.session_state:
        st.session_state['conf'] = 0.45
    if 'iou' not in st.session_state:
        st.session_state['iou'] = 0.45
    if 'box_color' not in st.session_state:
        st.session_state['box_color'] = '#47E242'

    if not video_file:
        tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file.read())
        
        
    if 'video' not in st.session_state:
        st.session_state['video'] = DEMO_VIDEO
        create_framelist()
        
    initial_video_frames()
    update_frame()
    detection_config()


def main():
    
    st.sidebar.markdown('---')
    st.sidebar.write('Detection')
    st.sidebar.slider('NMS IoU', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01, key='iou', on_change=on_config_change)
    st.sidebar.slider('Confidence', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01, key='conf', on_change=on_config_change)
    color = st.sidebar.color_picker('Bounding box color',
                                    value='#47E242',
                                    key='box_color',
                                    on_change=on_config_change)
    st.sidebar.markdown('---')
    st.sidebar.write('Video Setting (In develop...)')
    save_img = st.sidebar.checkbox('Save detection image')
    enabled_gpu = st.sidebar.checkbox('Enable GPU')
    st.sidebar.button('Extract Video')
    st.sidebar.markdown('---')


if __name__ == '__main__':
    st.sidebar.write(open('html/header.txt').read(), 
                     unsafe_allow_html=True)
    st.markdown(open('html/sidebar.txt').read(), unsafe_allow_html=True)
    video_file = st.sidebar.file_uploader('Upload Video', 
                                  type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    DEMO_VIDEO = 'dataset/Test/Violence/v_FTSN_-_5.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    video_label = st.markdown("""
                              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                              <center>
                              <h2>Video frame</h2>
                              </center>
                              """, unsafe_allow_html=True)
    progress_label = st.empty()
    _, img_frame, _ = st.columns([0.2, 1, 0.2])
    img_frame = st.empty()
    
    detection_model = get_detection_model()
    app_initial()

    frame_slider = st.slider('Frame', min_value=20,
                             max_value=int(utl.get_total_frame(tffile.name)), 
                             key='frame')

    try:
        main()
    except SystemExit:
        pass
    # check_size()