import streamlit as st
import cv2
import tempfile
import utilities as utl
import numpy as np
import os, psutil


def loading(func, args):
    loading_.markdown(open('html/loading.txt', 'r').read(), unsafe_allow_html=True)
    ret = func(*args)
    loading_.empty()
    return ret

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


def probabiliy_label(prob: float, class_index: int):
    icons = {0: open('html/nonviolence_icon.txt', 'r').read(),
             1: open('html/violence_icon.txt', 'r').read()}
    
    first_icon = (icons[class_index], '{:.2f}'.format(prob * 100))
    second_icon = (icons[0], '{:.2f}'.format(100-(prob * 100))) \
        if class_index == 1 else (icons[1], '{:.2f}'.format(100-(prob * 100)))
    
    class_css = (f"""
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css" integrity="sha512-KfkfwYDsLkIlwQp6LFnl8zNdLGxu9YAA1QvwINks4PhcElQSvqcyVLLD9aMhXd13uQjoXtEKNosOWaZqXgel0g==" crossorigin="anonymous" referrerpolicy="no-referrer" />
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                <div class="container" style="padding: 0% 25%; text-align: center;">
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

    
def get_frame(no_frame):
    total_frame = utl.get_total_frame(tffile.name)
    if no_frame < total_frame:
        return utl.get_frame(tffile.name, no_frame)
    else:
        print('But it out off range.. turnin to 20')
        st.session_state.no_frame = 20
        return utl.get_frame(tffile.name, 19)




def frame_change():
    
    no_frame = st.session_state.no_frame
    frame_range = (no_frame-20, no_frame)
    pred_range = st.session_state.range

    l = frame_range[0] - pred_range[0]
    st.session_state.range = frame_range
    if abs(l) < 20:
        
        _l = st.session_state.preds_list[l:] if l > 0 else \
            utl.get_features_list(tffile.name, frame_range[0], pred_range[0])
            
        
        _r = utl.get_features_list(tffile.name, pred_range[1], frame_range[1]) \
            if l > 0 else st.session_state.preds_list[: frame_range[1] - pred_range[0]]
            
        st.session_state.preds_list = np.array([*_l, *_r])
    elif abs(l) > 20:
        print('too long no data.. ')
        st.session_state.preds_list = \
            loading(get_preds_list, (frame_range[0], frame_range[1]))


def get_preds_list(start: int, stop: int):
    print('-get- range', start, stop)
    temporal = loading(utl.get_features_list, (tffile.name, start, stop)) 
    return temporal


def update_frame():
    print('update')
    frame = get_frame(st.session_state.no_frame - 1)
    img_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)


def video_initial():
    print('init')
    if not video_file:
        tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file.read())
        if video_file.name != st.session_state.video:
            print('video change create_data..')
            st.session_state.video = video_file.name
            st.session_state.no_frame = 20 
            st.session_state.range = [0, 20]
            st.session_state.preds_list = get_preds_list(0, 20)
            
    if 'video' not in st.session_state:
        print('create_data')
        st.session_state.video = DEMO_VIDEO
    if 'no_frame' not in st.session_state:
        st.session_state.no_frame = 20
    if 'range' not in st.session_state:
        st.session_state.range = [0, 20]
    if 'preds_list' not in st.session_state:
        st.session_state.preds_list = get_preds_list(0, 20)
    update_frame()
    
    
def main():
    st.sidebar.markdown('---')
    st.sidebar.write('Detection')
    st.sidebar.slider('NMS IoU', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01, key='iou')
    st.sidebar.slider('Confidence', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01, key='conf')
    color = st.sidebar.color_picker('Bounding box color',
                                    value='#47E242',
                                    key='box_color')
    st.sidebar.markdown('---')
    st.sidebar.write('Video Setting (In develop...)')
    save_img = st.sidebar.checkbox('Save detection image')
    enabled_gpu = st.sidebar.checkbox('Enable GPU')
    st.sidebar.button('Extract Video')
    st.sidebar.markdown('---')


def predict_img():
    frame_change()
    print('range before prob', st.session_state.range)
    prob, class_id = utl.predict(st.session_state.preds_list)
    probabiliy_label(prob, class_id)


if __name__ == '__main__':
    st.sidebar.write(open('html/header.txt').read(), 
                     unsafe_allow_html=True)
    st.markdown(open('html/sidebar.txt').read(), unsafe_allow_html=True)
    video_file = st.sidebar.file_uploader('Upload Video', 
                                  type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    
    DEMO_VIDEO = 'dataset/Test/Violence/v_FTSN_-_5.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    
    loading_ = st.empty()
    video_label = st.markdown("""
                              <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                              <center>
                              <h2>Video frame</h2>
                              </center>
                              """, unsafe_allow_html=True)
    
    _, progress_label, _ = st.columns([0.2, 1.0, 0.2])
    _, img_frame, _ = st.columns([0.2, 1.0, 0.2])

    video_initial()
    frame_slider = st.slider('Frame', min_value=20,
                             max_value=int(utl.get_total_frame(tffile.name)), 
                             key='no_frame')
    
    _, pred_button, _ = st.columns([0.8, 0.2, 0.8])
    with pred_button:
        st.button('Predict', on_click=predict_img)
    
    try:
        main()
    except SystemExit:
        pass
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    # check_size()