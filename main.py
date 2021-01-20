from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)
model_path='Acne Dataset\Acne_classifier.h5'
acne_model = load_model(model_path)
acne_model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
acne_model._make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(170, 170))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        number_class = np.argmax(preds)

        if number_class==0:
            return render_template('Cyst.html')
        elif number_class==1:
            return render_template('No Acne.html')
        elif number_class==2:
            return render_template('Nodule.html')
        elif number_class==3:
            return render_template('Pauple.html')
        elif number_class==4:
            return render_template('Pustule.html')

    return None


if __name__ == '__main__':
    app.run(debug=True)
