from flask import Flask, request
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



app = Flask(__name__)

model = tf.keras.models.load_model("letter_model.h5")


try:
    with open("class_names.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CLASS_NAMES)} classes.")
except FileNotFoundError:
    CLASS_NAMES = []

@app.route("/predict", methods=["POST"])
def predict():
    print("RECEIVED REQUEST!!")
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")  
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    plt.imshow(img)
    plt.title("Input Image")
    plt.savefig("input_img.png")
    plt.close()
    # Predict
    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    confidence = np.max(predictions)
    print(f"Predicted Letter: {predicted_class} with confidence {confidence:.2f}")


    return {"prediction": predicted_class}

if __name__ == "__main__":
    app.run(host='192.168.1.2', port=5000)
