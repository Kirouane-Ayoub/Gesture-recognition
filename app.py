import cv2
import numpy as np
from model import modle
import mediapipe as mp
import land_pip
import streamlit as st
import time
model = modle()
actions = np.array(['hello', 'thumbs up', 'thumbs down' , "Thanks"])
sequence = []
threshold = 0.8
mp_holistic = mp.solutions.holistic # Holistic model
land_vis = land_pip.Media_pipe_vis()


with st.sidebar:
    st.image("icon.png")
    detectf = st.selectbox("Detect from : " , ["File" , "webCam" , "URL"])
    save = st.radio("Do you want to save Results ? " , ["Yes" , "No"])


tab0 , tab1 = st.tabs(["HOME" , "DETECTION"])
with tab0 : 
    st.header("About This Project : ")
    st.image("handtracking_shot.jpg")
    st.write(""" 
        Gesture recognition is the process of identifying human hand gestures or movements and interpreting them as commands or actions
        for a computer system. In computer vision, gesture recognition has various applications such as virtual reality, gaming, sign language 
        recognition, and human-robot interaction. The system uses a camera or a sensor to capture the hand movements and then applies computer vision 
        algorithms to detect and recognize gestures. The machine learning models are trained on large datasets of annotated hand gestures to improve
        accuracy and performance. Gesture recognition systems are an important area of research in computer vision and have the potential to 
        revolutionize the way we interact with technology. A project on gesture recognition can involve building a real-time system that can
        recognize a set of predefined hand gestures and trigger corresponding actions.
        As an example, we will take (hello, thumbs up, thumbs down, and Thanks)
    """)

with tab1 :
    if detectf == "File" : 
        file_ = st.file_uploader("Upload your Video : " , type=["mp4" , "mkv", "webm"])
        if file_ : 
            source = file_.name
    elif detectf == "webCam" : 
        source = st.selectbox("Select Your Webcam Index : " , (1 , 2 , 3))
    elif detectf == "URL" : 
        source = st.text_input("Input Your URL here and click Entre : ")
    col1 , col2 , col3 = st.columns(3)
    with col1 : 
        st.write("Click To Start Detection : ")
        startb = st.button("Start") 
    with col3 :
        if save == "Yes" : 
            st.write("Double Click To Save Results : ")
        else : 
            st.write("Click To Stop")
        stop_saveb = st.button("Stop")
    # Set mediapipe model 
    if startb :
        cap = cv2.VideoCapture(source)
        if save == "Yes" :  
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
            out = cv2.VideoWriter(f'results/{str(time.asctime())}.mp4',
                                fourcc, 10, (w, h)) 
        else : 
            pass
        frame_window = st.image([])
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()
                # Make detections
                image, results = land_vis.mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                land_vis.draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = land_vis.extract_keypoints(results)
    
    
                sequence.append(keypoints)
                sequence = sequence[-15:]
            
                if len(sequence) == 15:

                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    cv2.putText(image, str(actions[np.argmax(res)]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (255, 0, 0), 2, cv2.LINE_AA)
                    
                img = cv2.cvtColor( image , cv2.COLOR_BGR2RGB)
                frame_window.image(img)
                try : 
                    out.write(image) 
                except : 
                    pass
        if stop_saveb : 
            cap.release()
            out.release()