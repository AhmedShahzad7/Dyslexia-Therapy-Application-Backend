from flask import Flask, request
from PIL import Image,ImageDraw,ImageFont
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from config.firebase import get_db
from firebase_admin import firestore
from google.cloud.firestore_v1 import ArrayUnion


#IMPORTING FUNCTIONS FROM FOLDER
from config.firebase import initialize_firebase
from apis.model_api import letter_predict
from apis.model_api import direction_predict
from apis.model_api import predict_handwriting
from apis.letter_handler import g_handler_letter
from apis.firestorequery_handler import store_direction_error


app = Flask(__name__)

#INITIALIZING FIREBASE
initialize_firebase()


#LETTER API REQUEST

@app.route("/predict_letter", methods=["POST"])
def predict_letter():
    print("RECEIVED REQUEST!!")
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")  
    prediction=letter_predict(img)
    print(prediction)


#DIRECTOIN API REQUEST ALEVEL 1 QUESTION 1
@app.route("/predict_direction", methods=["POST"])
def predict_direction():
    user_id = request.form.get('user_id')
    file = request.files["file"]

    if user_id:
        print(f"\n(DEBUG) Assessment Level 1 Question 1 RECEIVED REQUEST: {user_id}!!\n")
        
        img = Image.open(file.stream).convert("RGB")  
        direction=direction_predict(img)
        print(direction)
        update_success = store_direction_error(user_id, direction,"1")
        return direction
    return "error"

#DIRECTION MCQ ALEVEL 1 QUESTION 2
@app.route("/predict_direction_mcq", methods=["POST"])
def predict_direction_mcq():
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number')
    arrow_selected = request.form.get('arrow_selected')
    
    print(user_id,question_number,arrow_selected)
    if user_id:
        print(f"\n(DEBUG) Assessment Level 1 Question {question_number} RECEIVED REQUEST!! USERID: {user_id},QUESTION NUMBER: {question_number}, USER_INPUT: {arrow_selected}\n")
        update_success = store_direction_error(user_id, arrow_selected,question_number)
        print(update_success)
    return "error"
# DIRECTION MATCHING (ALEVEL 1 QUESTION 4)
@app.route("/predict_q4", methods=["POST"])
def predict_q4():
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number', "4") 
    
    # This receives: "Completed", "Correct Match: UP", or "Error: ..."
    arrow_selected = request.form.get('arrow_selected') 
    
    if user_id:
        
        print(f"\n(DEBUG) Assessment Level 1 Question {question_number} RECEIVED REQUEST!! USERID: {user_id},QUESTION NUMBER: {question_number}, USER_INPUT: {arrow_selected}\n")
     
        # Store the status in Firebase
        update_success = store_direction_error(user_id, arrow_selected, question_number)
        
        if update_success:
            return "Success"
        else:
            return "Database Error"

    return "error: missing user_id"
# DIRECTION MATCHING (ALEVEL 1 QUESTION 5)
@app.route("/predict_q5", methods=["POST"])
def predict_q5():
    user_id = request.form.get('user_id')
    question_number = "5"
    arrow_selected = request.form.get('arrow_selected') 
    if user_id:
        print(f"\n(DEBUG) Q5 LEFT FOOT RECEIVED: User={user_id}, Status={arrow_selected}\n")
        
        update_success = store_direction_error(user_id, arrow_selected, question_number)
        
        if update_success:
            return "Success"
        else:
            return "Database Error"
    return "error"

#A-LEVEL 2 QUESTION#6
@app.route("/predict_q6",methods=['POST'])
def predict_q6():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    letter=request.form.get('letter_selected')
    if user_id:
        print("The user(id): ",user_id)
        print("The Question Number: ",question_number)
        print("Letter Selected: ",letter)
        if not letter:
            return "Letter not Provided"
        img = Image.new('RGB', (120, 120), color=(255, 255, 255)) 
        draw = ImageDraw.Draw(img)

        # 2. Load the font
        try:
            # Use a large font size to fill the 120x120 space
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()

        # 3. Center the letter on the 120x120 canvas
        # Get dimensions of the letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Calculate coordinates to put the letter in the middle
        x = (120 - w) / 2
        y = (120 - h) / 2 - 10 # Slight offset for visual centering
        
        # 4. Draw the letter in black (0)
        draw.text((x, y), letter, fill=(0, 0, 0), font=font)

        #save the image
        # image_name = f"input_{letter}.png"
        # img.save(image_name)
        result=letter_predict(img)
        predicted_letter=""
        for x in result['prediction']:
            if isinstance(x, str):
                predicted_letter= x
        print(predicted_letter)
       
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
            .document(user_id) \
            .collection('Level_2') \
            .document(str(question_number))

        doc = doc_ref.get()
        if doc.exists:
            if doc.to_dict().get('Answer') == 'Incorrect':
                return "incorrect"
            elif doc.to_dict().get('Answer') == 'Correct' and predicted_letter!='p':
                doc_ref.update({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    'Error': 'p'
                })
                return "incorrect"
            elif doc.to_dict().get('Answer') == 'Correct' and predicted_letter=='p':
                return "correct"

        if predicted_letter == 'p' and letter == 'p':
            doc_ref.set({
                'Question Number': question_number,
                'Answer': 'Correct'
            })
            return "correct"
        else:
            doc_ref.set({
                'Question Number': question_number,
                'Answer': 'Incorrect',
                'Error': 'p'
            })
            return "incorrect"

#A-LEVEL 2 QUESTION#7
@app.route("/predict_q7",methods=['POST'])
def predict_q7():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    expected_Letter=request.form.get('expected_Letter')
    file = request.files["file"]
    if user_id:
        print("The User(id): ",user_id)
        print("The Question Number: ",question_number)
        print("Expected Letter: ",expected_Letter)
        img = Image.open(file.stream).convert("RGB") 
        #for debugging
        #img.save("test.png")
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
            .document(user_id) \
            .collection('Level_2') \
            .document(str(question_number))
        if expected_Letter == 'q':
            prediction=g_handler_letter(img)
            if prediction not in ['q','Q']:
                doc_ref.set({
                'Question Number': question_number,
                'Answer': 'Incorrect',
                },merge=True)
                doc_ref.update({
                    'Error': ArrayUnion(['q'])
                    })
                print("Error q")
            else:
                print("Correct q")
        else:
            pred=letter_predict(img)
            model_predict=""
            for x in pred['prediction']:
                if isinstance(x, str):
                    model_predict= x
            # print(model_predict)
            if expected_Letter == 'p':
                if model_predict not in ['p','P_caps']:
                    doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    },merge=True)
                    doc_ref.update({
                    'Error': ArrayUnion(['p'])
                    })
                    print("Error p")
                else:
                    print("Correct p")
            elif expected_Letter == 'b':
                if model_predict not in ['b', 'B_caps']:
                    doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    },merge=True)
                    doc_ref.update({
                    'Error': ArrayUnion(['b'])
                    })
                    print("Error b")
                else:
                    print("Correct b")
            elif expected_Letter == 'd':
                if model_predict not in ['d','D_caps']:
                    doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    },merge=True)
                    doc_ref.update({
                    'Error': ArrayUnion(['d'])
                    })
                    print("Error d")
                else:
                    print("Correct d")


if __name__ == "__main__":
    app.run(host='192.168.0.14', port=5000)

