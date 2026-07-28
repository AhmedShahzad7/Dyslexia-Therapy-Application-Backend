from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google import genai
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from apis.letter_handler import g_handler_direction
from apis.letter_handler import g_handler_letter

img_height=64
img_width=64

model = tf.keras.models.load_model('models/letter_model.h5')



try:
    with open("models/class_names.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CLASS_NAMES)} classes.")
except FileNotFoundError:
    CLASS_NAMES = []




def letter_predict(img):
    img = img.resize((img_height, img_width
    ))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    plt.imshow(img)
    plt.title("Input Image")
    plt.savefig("models/input_img.png")
    plt.close()
 
    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    confidence = np.max(predictions)
    


    predictions_indices = model.predict(img_array)[0]
    top_5_indices = np.argsort(predictions_indices)[-5:][::-1]

    top_5_results = []
    for i in top_5_indices:
        top_5_results.append({
            "label": CLASS_NAMES[i],
            "confidence": float(predictions_indices[i])
        })
    
    print(top_5_results)
    
    return ({"prediction": {predicted_class,confidence}})




def direction_predict(img):
    img = img.resize((img_height, img_width
    ))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    plt.imshow(img)
    plt.title("Direction Image")
    plt.savefig("models/direction_img.png")
    plt.close()

    return g_handler_direction(img)
 

def predict_handwriting(image):
   
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

  
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text