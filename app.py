import os
from flask import Flask, redirect, url_for, render_template, request, Response
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

model = YOLO("yolov8n-face.pt")

# Real Time Face detection on webcam
def real_time_Faces():
    while True:
        path = 'captured_image.jpg'
        frame = cv2.imread(path)
        result_frame = model.predict(source=path, show=False)
        boxes = result_frame[0].boxes
        boxIndx = []
        for box in boxes:
            boxIndx.append(box.data.tolist()[0])
        face_result_vid = np.array(boxIndx)

        for face_vid in face_result_vid:
            x1, y1, x2, y2, acc, _ = face_vid
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            imageVid = cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
            cv2.putText(imageVid, str(acc), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', imageVid)
        frame = buffer.tobytes()
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get image data from frontend
    image_data = request.files['image'].read()
    # Get the path to the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Specify the file path for saving the image
    image_path = os.path.join(current_directory, 'captured_image.jpg')

    # Save image data to a file
    with open(image_path, 'wb') as f:
        f.write(image_data)

    return 'Image uploaded successfully!'

@app.route('/video')
def video():
    return Response(real_time_Faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
