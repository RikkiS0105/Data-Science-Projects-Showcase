from flask import Flask, render_template, request
import tensorflow
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress NVCuda.dll file warning since no NVCuda Graphics Card enabled
import cv2
import numpy as np
from PIL import Image
from numpy import asarray

print('All Libraries imported successfully')

app = Flask(__name__)

model = keras.models.load_model('FMNIST_87.h5')
print('model loaded')

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result', methods = ['POST'])

def result():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        img = Image.open(f)
        img_array = asarray(img)
        resized_file = cv2.resize(img_array, (28, 28))
        grayscaled = cv2.cvtColor(resized_file, cv2.COLOR_BGR2GRAY)
        reshaped_file = grayscaled.reshape(1,28,28,1)
        scaled_file = reshaped_file/255
        final_file = scaled_file
        classes = ['T-Shirt/Top','Trouser' , 'Pullover' , 'Dress' , 'Coat' , 'Sandals' , 'Shirt' , 'Sneaker' , 'Bag' , 'Ankle Boots']
        result_pred = model.predict(final_file).argmax(axis=1)
        predicted_text = classes[result_pred[0]]

        
        return render_template("result.html", name = f.filename, predicted_text = predicted_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
