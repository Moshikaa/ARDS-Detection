from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__,template_folder='template')
MODEL_PATH = '/Applications/Moshi/Projects/ARDS detection system /model_1epoch.h5'
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
                 img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    img=np.expand_dims( img,axis=0 )
    preds = model.predict(img)
    return preds
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        pred_class = preds.argmax(axis=-1)          
        pred_class = str(pred_class)              
        if pred_class=='[1]':
            result ='This is a Normal case'
        else:
            result='This is a ARDS Case'
        return result
    return None
if __name__ == '__main__':
    app.run(port=5002, debug=True)