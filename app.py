import os

import cv2
import numpy as np
from flask import Flask, render_template, request
import tensorflow
from keras.models import load_model
from werkzeug.utils import secure_filename
from gtts import gTTS
from flask import Flask, render_template, request, Response
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
MODEL_PATH = './model/TSR.h5'

# Ensure upload and static directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load the model once (avoids reloading on each prediction)
model = load_model(MODEL_PATH)

# Traffic sign classes
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Vehicle > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing vehicle > 3.5 tons'
}

def generate_speech(text):
    speech_filename = "prediction.mp3"
    speech_path = os.path.join(STATIC_FOLDER, speech_filename)
    tts = gTTS(text=f"The predicted traffic sign is {text}", lang='en')
    tts.save(speech_path)
    return f"/static/{speech_filename}"  # 🔁 Return relative path for browser
def image_processing(img_path):
    """Processes an image and predicts the traffic sign."""
    img = tensorflow.keras.utils.load_img(img_path, target_size=(30, 30))
    img_array = tensorflow.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class


def generate_speech(text):
    """Generates a speech file for the given text and saves it in the static folder."""
    speech_path = os.path.join(STATIC_FOLDER, "prediction.mp3")
    tts = gTTS(text=f"The predicted traffic sign is {text}", lang='en')
    tts.save(speech_path)
    return speech_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/more')
def more():
    return render_template('more.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(file_path)

    # Predict traffic sign
    predicted_class = image_processing(file_path)
    result_text = classes[predicted_class]

    # Generate speech output
    speech_file = generate_speech(result_text)

    # Remove the uploaded file after processing
    os.remove(file_path)

    return render_template('result.html', result=result_text, speech_file=speech_file)


# Capture webcam and process frames
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame for prediction
        img = cv2.resize(frame, (30, 30))
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = classes[predicted_class]

        # Overlay prediction on frame
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
