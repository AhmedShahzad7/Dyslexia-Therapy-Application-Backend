from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google import genai

from apis.letter_handler import g_handler_direction
from apis.letter_handler import g_handler_letter

img_height=64
img_width=64

model = tf.keras.models.load_model('models/letter_model.h5')


#CLASS NAMES EXCEPTION HANDLING
try:
    with open("models/class_names.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CLASS_NAMES)} classes.")
except FileNotFoundError:
    CLASS_NAMES = []



#PREDICTING LETTERS
def letter_predict(img):
    img = img.resize((img_height, img_width
    ))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    plt.imshow(img)
    plt.title("Input Image")
    plt.savefig("models/input_img.png")
    plt.close()
    # Predict
    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    confidence = np.max(predictions)
    
    #HIDE GEMINI CODE ##
    print(g_handler_letter(img)) 
    
    return ({"prediction": {predicted_class,confidence}})



#PREDICTING DIRECTIONS
def direction_predict(img):
    img = img.resize((img_height, img_width
    ))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    plt.imshow(img)
    plt.title("Direction Image")
    plt.savefig("models/direction_img.png")
    plt.close()
    # Predict
    return g_handler_direction(img)
 