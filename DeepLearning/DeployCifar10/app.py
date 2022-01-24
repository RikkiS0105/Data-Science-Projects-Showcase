from flask import Flask, render_template, request
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Suppress NVCuda.dll file warning since no NVCuda Graphics Card enabled
import cv2
import numpy as np
from PIL import Image
from numpy import asarray

print('All Libraries imported successfully')

app = Flask(__name__)

model = keras.models.load_model('cifar10_71.h5')
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
        resized_file = cv2.resize(img_array, (32, 32))
        reshaped_file = resized_file.reshape(1,32,32,3)
        scaled_file = reshaped_file/255
        final_file = scaled_file
        classes = ['airplane','automobile' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck']
        result_pred = model.predict(final_file).argmax(axis=1)
        predicted_text = classes[result_pred[0]]

        
        return render_template("result.html", name = f.filename, predicted_text = predicted_text)


# # @app.route('/result', methods = ['POST'])

# # def result():
# #     preg = request.form.get('preg')
# #     plas = request.form.get('plas')
# #     pres = request.form.get('pres')
# #     skin = request.form.get('skin')
# #     test = request.form.get('test')
# #     mass = request.form.get('mass')
# #     pedi = request.form.get('pedi')
# #     age = request.form.get('age')
    
# #     pred = model.predict([[int(preg) ,int(plas) , int(pres) , int(skin) , int(test) , int(mass) , int(pedi) , int(age)]])

# #     if pred[0] == 1:
# #         output = 'diabetic'
# #     else:
# #         output = 'not diabetic'
    
# #     return render_template('result.html', predicted_text = f'You are {output}.')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
