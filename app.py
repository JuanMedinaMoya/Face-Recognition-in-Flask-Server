import flask

from flask import Flask, redirect, render_template, request
import faceRecognition as fc

import os

import magic


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getFaces', methods=['POST'])
def getFaces():
    image = request.files['Image']
    #my_type = (magic.from_file(image, mime=True))

    if 'image' in image.mimetype:
        image.save(os.path.join('', 'img.png'))

        fc.faceClassifier('img.png')

        os.remove('img.png')
        
    elif 'video' in image.mimetype:
        image.save(os.path.join('', 'video.mp4'))

        fc.faceClassifier('video.mp4')

        os.remove('video.mp4')

    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=4000)

from flask import Flask
app = Flask(__name__)
