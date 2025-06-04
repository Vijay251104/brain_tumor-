from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = r"C:\Users\vijay\Downloads\modelres50.h5"
model = load_model(MODEL_PATH)

# Ensure upload folder exists
UPLOAD_FOLDER = r"D:\ml model\uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allow both GET and POST for home route
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'imagefile' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['imagefile']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    def model_predict(img_path, model):
        img = image.load_img(img_path, target_size=(200, 200))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255
        preds = model.predict(img)
        return np.argmax(preds, axis=1)

    pred = model_predict(file_path, model)
    os.remove(file_path)

    if pred is None:
        return jsonify({"error": "Prediction failed"}), 500

    labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    result = labels[pred[0]] if pred[0] < len(labels) else "Unknown"
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
