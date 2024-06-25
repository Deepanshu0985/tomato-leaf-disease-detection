from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import os
import cv2
import requests
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')


### prediction

img_size = 224

def preprocess_image(img, img_size):
    # Resize the image
    img = cv2.resize(img, (img_size, img_size))
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    img = gray.astype('float32') / 255.0
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=(0, -1))
    return img

@app.route('/leaf', methods=['GET', 'POST'])
def leaf():
    '''
    For rendering results on HTML GUI
    '''
    categories = None
    label_dict = None
    loaded_model = None

    if request.method == 'POST':

        file = request.files['image']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
        file.save(file_path)

       
        categories = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight']
 

        loaded_model = load_model("tomato.keras")
        label_dict = {i: category for i, category in enumerate(categories)}

         # Read the input image (grayscale)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize the image to img_size x img_size
        resized_img = cv2.resize(img_rgb, (img_size, img_size))

        # Normalize pixel values to range [0, 1]
        normalized_img = resized_img / 255.0

        # Reshape input for model prediction
        input_img = normalized_img.reshape(-1, img_size, img_size, 3)  # Reshape input for model prediction

        # Make a prediction
        prediction = loaded_model.predict(input_img)

        # Get the predicted class label
        predicted_class_index = np.argmax(prediction)
        predicted_label = label_dict[predicted_class_index]

        if predicted_label == "Tomato___Bacterial_spot":
            remedy = "Bacterial spot in tomatoes is caused by Xanthomonas species, resulting in dark, water-soaked spots on leaves and fruit. To manage this disease, use disease-free seeds and transplants. Apply copper-based bactericides as preventive measures. Remove and destroy infected plant debris, and avoid working in wet fields to prevent bacterial spread. Practice crop rotation and maintain good field hygiene."
        elif predicted_label == "Tomato___Early_blight":
            remedy = "Early blight in tomatoes, caused by Alternaria solani, manifests as concentric rings on leaves and stems. To control early blight, use resistant varieties and apply fungicides as needed. Remove and destroy infected plant debris, and practice crop rotation. Mulch around the base of plants to reduce soil splash and maintain proper spacing for air circulation."
        elif predicted_label == "Tomato___healthy":
            remedy = "Your tomato plant is healthy! Continue regular care routines such as consistent watering, balanced fertilization, and pest monitoring. Ensure the plant receives adequate sunlight and proper support for growth. Mulch around the base to retain moisture and suppress weeds. Regularly inspect for any early signs of disease or pests and address them promptly."
        elif predicted_label == "Tomato___Late_blight":
            remedy = "Late blight in tomatoes is caused by Phytophthora infestans, leading to large, water-soaked lesions on leaves and fruit. To manage late blight, use resistant varieties and apply fungicides preventatively, especially during cool, wet weather. Remove and destroy infected plant material and practice crop rotation. Avoid overhead watering and ensure good air circulation around plants."

        return render_template('remedy.html', predicted_label=predicted_label, remedy=remedy, img=file_path)

    return render_template('leaf.html')


if __name__ == "__main__":

    app.run(debug=True)
    