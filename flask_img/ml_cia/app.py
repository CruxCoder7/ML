from flask import Flask, render_template, request, url_for
import os
from keras.models import load_model
from skimage.transform import resize
import imageio.v2 as imageio

app = Flask(__name__)

model = load_model(r"C:\Programming\ML-Tech-Lab\flask_img\ml_cia\model.h5")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    img = request.files["img"]
    img, _ = os.path.splitext(img.filename)

    img = url_for('static', filename=img+'.png')

    path = r'C:\Programming\ML-Tech-Lab\flask_img\ml_cia'
    image = imageio.imread(path+img, as_gray=True)

    # resizing and reshaping the image
    image = resize(image, (28, 28))
    image = image.reshape(1, 28, 28, 1)

    out = model.predict(image)
    out = out.argmax(axis=1)[0]
    return render_template("new.html", data=[img, out])


if __name__ == "__main__":
    app.run(debug=True)
