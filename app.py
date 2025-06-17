from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.getcwd(), 'pneumonia_model.h5')
model = load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join('static', 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return "Pneumonia" if preds[0][0] > 0.5 else "Normal"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)

        result = model_predict(file_path, model)
        return render_template('index.html', result=result, image=file_path)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)

