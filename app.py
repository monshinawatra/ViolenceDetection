import streamlit as st
import cv2
import tempfile
import utils as utl
import streamlit.components.v1 as components
from stqdm import stqdm


def update_frame():
    frame = get_frame(st.session_state.frame-1)

    temp_range = [st.session_state.frame-20, st.session_state.frame]
    temporal_frames = st.session_state.pred_list[temp_range[0]: temp_range[1]]

    prob, class_index = utl.predict(temporal_frames, frame)
    classes = {0: "NonViolence", 1: "Violence", 2: ""}

    progress_label.write("{0}: {1:.2f}%".format(
        classes[class_index], prob*100))
    progress.progress(float(prob))
    img_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def get_all_video_frames(vid_name):
    vid = vid_name

    prog = stqdm(range(utl.get_total_frame(vid)))
    prog.set_description_str("Getting all frames")

    frames_list = [utl.get_frame(vid, i) for i in prog]

    return frames_list


def initial_video_frames():
    if 'frames_list' not in st.session_state:
        st.session_state['frames_list'] = get_all_video_frames(
            tffile.name)

    if video_file == None:
        return

    if video_file.name != st.session_state.video:
        st.session_state.video = video_file.name
        loading = st.markdown(
            """
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">
                <center>
                <div class="fa-3x">
                    <i class="fa-solid fa-sync fa-spin"></i>
                </div>
                Wait a minute..
            """,
            unsafe_allow_html=True)
        st.session_state['frames_list'] = get_all_video_frames(
            tffile.name)
        st.session_state['pred_list'] = get_all_pred_list()
        loading.empty()


def get_all_pred_list():
    frame_list = st.session_state['frames_list']
    return utl.get_features_list(frame_list)


def get_frame(frame):
    if frame < len(st.session_state['frames_list']):
        return st.session_state['frames_list'][frame]
    else:
        st.session_state.frame = 20
        return st.session_state['frames_list'][19]


def app_initial():
    if 'video' not in st.session_state:
        st.session_state['video'] = DEMO_VIDEO

    if 'frame' not in st.session_state:
        st.session_state['frame'] = 20

    if not video_file:
        tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file.read())

    initial_video_frames()
    if 'pred_list' not in st.session_state:
        st.session_state['pred_list'] = get_all_pred_list()

    update_frame()


def main():
    css_header = \
        """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">
            <div class="fa-2x">
                <i class="fa-solid fa-cog"></i> &nbsp Setting
            </div>
        """
    header = st.sidebar.write(css_header, unsafe_allow_html=True)

    st.markdown(
        """
            <style>
                [data-testid='stSidebar'][aria-expanded="true"] > div:first-child{width: 400px;}
                [data-testid='stSidebar'][aria-expanded="false"] > div:first-child{width: 400px; margin-left: -400px;}
            </style>
        """, unsafe_allow_html=True
    )

    st.sidebar.markdown('---')
    st.sidebar.write('Detection')
    st.sidebar.slider('NMS IoU', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01)
    st.sidebar.slider('Confidence', min_value=0.01,
                      max_value=1.0, value=0.45, step=0.01)

    st.sidebar.write('Classification')
    st.sidebar.slider('Frame Skiping', min_value=0,
                      max_value=5, value=0)
    st.sidebar.markdown('---')

    classify_text = st.sidebar.checkbox('Classification Text', value=True)
    save_img = st.sidebar.checkbox('Save image detection')
    save_vid = st.sidebar.checkbox('Save video')
    enabled_gpu = st.sidebar.checkbox('Enable GPU')

    st.sidebar.markdown('---')


def media_button():
    css_media_player = \
        """
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">

            <center>
            <div class="container" >
                <div class="row" style="padding: 20px 150px;">
                    <div class="col">
                        <button type="button" class="btn btn-light">
                            <i class="fa-solid fa-backward"></i>
                        </button>
                    </div>

                    <div class="col">
                        <button type="button" class="btn btn-light">
                            <i class="fa-solid fa-play"></i>
                        </button>

                    </div>

                    <div class="col">
                        <button id="button" type="button" class="btn btn-light">
                            <i class="fa-solid fa-forward"></i>
                        </button>
                    </div>

                </div>
            </div>
        """
    components.html(css_media_player)


if __name__ == '__main__':
    video_file = st.file_uploader(
        'Upload Video', type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    DEMO_VIDEO = 'dataset/Test/Violence/v_FTSN_-_5.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    video_label = st.subheader('Video')
    progress_label = st.empty()
    progress = st.empty()

    img_frame = st.empty()
    app_initial()

    frame_slider = st.slider('Frame', min_value=20, max_value=int(
                             utl.get_total_frame(tffile.name)), key='frame', on_change=update_frame)

    try:
        main()
        media_button()
    except SystemExit:
        pass
