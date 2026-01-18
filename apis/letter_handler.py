from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from google import genai


client = genai.Client(api_key="AIzaSyBT0alA9j4koF6ihAplZLLrCxryYfUTo4A")
def g_handler_direction(img):
    prompt = (
        "You will be our Arrow Direction identifier you will detect which hand-drawn arrow  is shown in the image and it should only be 1 word(only english word if you see something else put -) answer from you for example if the image shows UP arrow then you will say Up if its Left then you will say Left"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, img]
    )
    gemini_analysis = response.text
    return gemini_analysis


def g_handler_letter(img):

    prompt = (
       "You will be our handwriting identifier you will detect which letter is shown in the image and it should only be 1 letter(only english letters if you see something else put -) answer from you for example if the image shows A then you will say A_caps if its a then you will say a"
     )
    response = client.models.generate_content(
       model="gemini-2.5-flash",
        contents=[prompt, img]
    )
    gemini_analysis = response.text

    return gemini_analysis    
