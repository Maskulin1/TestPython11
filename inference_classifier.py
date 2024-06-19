import av
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pyttsx3

# Sidebar configuration
with st.sidebar:
    # Adding header
    st.header("BISINDO Recognition", divider='rainbow')
    # Adding logo
    st.image("bisindo.jpg")
    # Adding description
    st.write("Website Deteksi Gerakan Tangan ini memiliki kemampuan melakukan identifikasi alfabet BISINDO A-Z.")
    st.image("kataisyarat.jpg")
    st.write("Serta 6 kata antara lain:\n"
             "- Tolong\n"
             "- Semoga Beruntung\n"
             "- Sama-sama\n"
             "- Terima kasih\n"
             "- Keren\n"
             "- Halo")
    # Adding subheader
    st.subheader("by Reihan Septyawan")

# Load the trained model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Labels for prediction
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: 'SEMOGA BERUNTUNG ', 27: 'TOLONG ', 28: 'KEREN ', 29: 'HALO ',
               30: 'TERIMA KASIH ', 31: 'SAMA-SAMA '}

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_prediction = None
        self.prev_time = time.time()
        self.text = ""
        self.counter = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = image.shape
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Ensure that we have data for both hands (84 elements: 21 landmarks * 2 (x, y) * 2 hands)
            max_length = 84
            if len(data_aux) < max_length:
                data_aux.extend([0] * (max_length - len(data_aux)))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw rectangle and put text
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(image, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            current_time = time.time()
            if self.prev_prediction == predicted_character:
                self.counter += current_time - self.prev_time
                self.prev_time = current_time
                if self.counter >= 2:
                    self.text += predicted_character
                    self.counter = 0
            else:
                self.prev_prediction = predicted_character
                self.prev_time = current_time
                self.counter = 0

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Add custom CSS for background color
st.markdown(
    """
    <style>
    .streamlit-webrtc .row-widget.stButton button {
        background-color: white;
        color: black;
    }
    .predicted-character {
        white-space: pre-wrap;  /* This property ensures text wraps to the next line */
        word-wrap: break-word;  /* This property breaks long words to fit the container */
        font-size: 24px;
        font-family: Arial, sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hand Gesture Recognition")
st.write("This application recognizes hand gestures using a webcam feed.")

# Initialize session state for the text and counter
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'counter' not in st.session_state:
    st.session_state.counter = 0.0

# Display the webcam feed
webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})

# Create placeholders for the predicted character and counter
predicted_character_container = st.empty()
counter_container = st.empty()

# Reset button
if st.button("Reset"):
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.text = ""
        webrtc_ctx.video_transformer.counter = 0
        st.session_state.text = ""
        st.session_state.counter = 0.0

# Function to handle text-to-speech
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Text-to-speech button
if st.button("🔊 Speak"):
    if webrtc_ctx.video_transformer:
        speak(webrtc_ctx.video_transformer.text)

# Continuously update the containers with the predicted character and counter
while True:
    if webrtc_ctx.video_transformer:
        st.session_state.text = webrtc_ctx.video_transformer.text
        st.session_state.counter = webrtc_ctx.video_transformer.counter
        predicted_character_container.markdown(f"<div class='predicted-character'>Predicted Character: {st.session_state.text}</div>", unsafe_allow_html=True)
        counter_container.text(f"Counter: {st.session_state.counter:.2f} seconds")
    time.sleep(0.1)