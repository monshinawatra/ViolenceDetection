# ViolenceDetection
Detect violence using video classification to recognize action in the input video.

- <a href='https://colab.research.google.com/drive/1v7OuiPpKz6FlPLOFtaDdyZfpKVgpmfm1?usp=sharing'> Notebook </a>
- <a href='https://medium.com/@monchinawat/%E0%B8%95%E0%B8%A3%E0%B8%A7%E0%B8%88%E0%B8%88%E0%B8%B1%E0%B8%9A%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%A3%E0%B8%B8%E0%B8%99%E0%B9%81%E0%B8%A3%E0%B8%87%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-video-classification-%E0%B8%89%E0%B8%9A%E0%B8%B1%E0%B8%9A%E0%B8%87%E0%B9%88%E0%B8%B2%E0%B8%A2-d2bbf894149f'> Medium </a>
## Install
```
pip install -r requirements.txt
```

## How to uses
```
python predict.py --dir ["dir"] --write-vid [output path].avi --write-csv [output path].csv
```

or you can try web app by using streamlit.
```
streamlit run app.py
```

Example of output
[![Watch the video](https://img.youtube.com/vi/ngz5GU7KlM4/maxresdefault.jpg)](https://youtu.be/ngz5GU7KlM4)
![plot](https://github.com/monshinawatra/ViolenceDetection/blob/main/preview/output.png?raw=true)
