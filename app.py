import os
import json
import re
import jellyfish
from flask import Flask, request,jsonify
from PIL import Image,ImageDraw,ImageFont
import io
import whisperx
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from config.firebase import get_db
from firebase_admin import firestore
from google.cloud.firestore_v1 import ArrayUnion
from werkzeug.utils import secure_filename
import traceback
import string

#IMPORTING FUNCTIONS FROM FOLDER
from config.firebase import initialize_firebase
from apis.model_api import letter_predict
from apis.model_api import direction_predict
from apis.model_api import predict_handwriting
from apis.letter_handler import g_handler_letter
from apis.firestorequery_handler import *
import requests


app = Flask(__name__)
#WHISPER X
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_audio') 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = "cpu" 
compute_type = "int8" 
print("Loading WhisperX model...")
model = whisperx.load_model("base", device, compute_type=compute_type)
print("Model loaded successfully!")



#INITIALIZING FIREBASE
initialize_firebase()


#LETTER API REQUEST
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # 2. Check if an audio file was sent in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # 3. Load the saved audio and transcribe it
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=8) 
        
        # 4. Clean up the temporary file
        os.remove(filepath)
        
        # 5. Return the transcribed text segments as JSON
        return jsonify({"transcription": result["segments"]}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#A-Level Question 12
@app.route('/transcribe_and_score', methods=['POST'])
def transcribe_and_score():
    if 'audio' not in request.files or 'target_word' not in request.form:
        return jsonify({"error": "Missing audio file or target_word"}), 400
    user_id = request.form.get('user_id')
    target_word = request.form['target_word'].lower().strip()
    file = request.files['audio']
    question_number = request.form.get('question_number')
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # 1. Transcribe the audio
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=8) 
        os.remove(filepath)
        
        # 2. Extract and clean the text (remove punctuation and make lowercase)
        raw_text = " ".join([segment["text"] for segment in result["segments"]]).strip().lower()
        transcribed_clean = re.sub(r'[^\w\s]', '', raw_text)
        
        # If Whisper picked up multiple words (e.g., "the board"), grab the last one
        words_spoken = transcribed_clean.split()
        word_to_compare = words_spoken[-1] if words_spoken else ""

        # 3. Calculate Phonetic Similarity
        # Jaro-Winkler prioritizes words that start with the same sounds/letters
        similarity_score = jellyfish.jaro_winkler_similarity(target_word, word_to_compare)
        
        # 4. Setting the threshold! 
        is_correct = bool(similarity_score >= 0.65)
        
        # Generate the raw phonetic codes just so you can see them in your logs!
        target_metaphone = jellyfish.metaphone(target_word)
        transcribed_metaphone = jellyfish.metaphone(word_to_compare)
        
        print(f"Target: {target_word} ({target_metaphone}) | AI Heard: {word_to_compare} ({transcribed_metaphone}) | Score: {similarity_score}")
        detected_errors=[]
        if is_correct:
            print(f"(DEBUG) User {user_id} correctly pronounced {target_word}.")
        elif user_id:
            detected_errors.append(target_word)
            store_voice_error(user_id,target_word,word_to_compare,question_number, detected_errors)
      
            
                                  
        
        
        
        return jsonify({
            "target_word": target_word,
            "transcribed_word": word_to_compare,
            "similarity_score": similarity_score,
            "is_correct": is_correct
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500




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

        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()

        # Center the letter on the 120x120 canvas
        # Get dimensions of the letter
        bbox = draw.textbbox((0, 0), letter, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Calculate coordinates to put the letter in the middle
        x = (120 - w) / 2
        y = (120 - h) / 2 - 10 
        
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

#A-Level 2 Question#8
@app.route("/predict_q8",methods=['POST'])
def predict_q8():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    expected_Letter=request.form.get('expected_Letter')
    Box_Index=request.form.get('Box_Index')
    file = request.files["file"]
    
    if user_id:
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
            .document(user_id) \
            .collection('Level_2') \
            .document(str(question_number))
        print("The User(id): ",user_id)
        print("The Question Number: ",question_number)
        #Box n
        if Box_Index=="1":
            img = Image.open(file.stream).convert("RGB") 
            print("Box Index",Box_Index)
            print("Expected Letter: ",expected_Letter)
            model_predict_1=g_handler_letter(img)
            # model_predict_1=""
            # for x in pred['prediction']:
            #     if isinstance(x, str):
            #         model_predict_1= x
            # print("Model Prediction: ",model_predict_1)
            print("Model Prediction: ",model_predict_1)
            if model_predict_1 != 'n':
                doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    'Error': ArrayUnion(['n'])
                    },merge=True)
                # doc_ref.update({
                #     'Error': ArrayUnion(['n'])
                #     })
                print("Error n")
                return "incorrect"
            print("correct")
            return "correct"
        #Box u
        elif Box_Index=="2":
            img = Image.open(file.stream).convert("RGB") 
            print("Box Index",Box_Index)
            print("Expected Letter: ",expected_Letter)
            print("Box Index",Box_Index)
            pred=letter_predict(img)
            model_predict_2=""
            for x in pred['prediction']:
                if isinstance(x, str):
                    model_predict_2= x
            print("Model Prediction: ",model_predict_2)
            if model_predict_2 not in ['u','U_caps']:
                doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    'Error': ArrayUnion(['u'])
                    },merge=True)
                # doc_ref.update({
                #     'Error': ArrayUnion(['u'])
                #     })
                print("Error u")
                return "incorrect"
            print("correct")
            return "correct"
        #Box s
        elif Box_Index=="3":
            img = Image.open(file.stream).convert("RGB") 
            print("Box Index",Box_Index)
            print("Expected Letter: ",expected_Letter)
            print("Box Index",Box_Index)
            pred=letter_predict(img)
            model_predict_3=""
            for x in pred['prediction']:
                if isinstance(x, str):
                    model_predict_3= x
            print("Model Prediction: ",model_predict_3)
            if model_predict_3 not in ['s','S_caps','5']:
                doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    'Error': ArrayUnion(['s'])
                    },merge=True)
                # doc_ref.update({
                #     'Error': ArrayUnion(['s'])
                #     })
                print("Error s")
                return "incorrect"
            print("correct")
            return "correct"
        #Box z
        elif Box_Index=="4":
            img = Image.open(file.stream).convert("RGB") 
            print("Box Index",Box_Index)
            print("Expected Letter: ",expected_Letter)
            print("Box Index",Box_Index)
            pred=letter_predict(img)
            model_predict_4=""
            for x in pred['prediction']:
                if isinstance(x, str):
                    model_predict_4= x
            print("Model Prediction: ",model_predict_4)
            if model_predict_4 not in ['z','Z_caps','2']:
                doc_ref.set({
                    'Question Number': question_number,
                    'Answer': 'Incorrect',
                    'Error': ArrayUnion(['z'])
                    },merge=True)
                # doc_ref.update({
                #     'Error': ArrayUnion(['z'])
                #     })
                print("Error z")
                return "incorrect"
            print("correct")
            return "correct"
        else:
            return "incorrect"
  #A-Level 2 Question#9
@app.route('/transcribe_phoneme', methods=['POST'])
def transcribe_phoneme():
    if 'audio' not in request.files or 'target_sound' not in request.form:
        return jsonify({"error": "Missing audio file or target_sound"}), 400
        
    target_word = request.form['target_sound'].lower().strip()
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number', '9')
    
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        audio = whisperx.load_audio(filepath)
        

        result = model.transcribe(audio, batch_size=8, language="en") 
        os.remove(filepath)
        
        raw_text = " ".join([segment["text"] for segment in result["segments"]]).strip().lower()
        transcribed_clean = re.sub(r'[^\w\s]', '', raw_text)
        
        # Get the word the AI heard
        words_spoken = transcribed_clean.split()
        heard_word = words_spoken[0] if words_spoken else ""
        
   
        similarity_score = jellyfish.jaro_winkler_similarity(target_word, heard_word)
        
    
        is_correct = bool(similarity_score >= 0.65)
    
        target_metaphone = jellyfish.metaphone(target_word)
        transcribed_metaphone = jellyfish.metaphone(heard_word)
        
        print(f"Target Word: '{target_word}' ({target_metaphone}) | AI Heard: '{heard_word}' ({transcribed_metaphone}) | Score: {similarity_score}")

   
        if not is_correct and user_id:
            db = get_db()
            doc_ref = db.collection('Assessment_Test') \
                        .document(user_id) \
                        .collection('Level_2') \
                        .document(str(question_number))
             
         

            error_log = [target_word]           
            
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Incorrect',
            }, merge=True)

    
            doc_ref.update({
                'Error': firestore.ArrayUnion(error_log)
            })

        return jsonify({
            "target_sound": target_word,
            "heard_sound": heard_word,
            "similarity_score": similarity_score,
            "is_correct": is_correct
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500    



#A-Level 2 Question#10
@app.route("/predict_q10",methods=['POST'])
def predict_q10():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    reverse_letter=request.form.get('reversed_selected')
    if user_id:
        print("The user(id): ",user_id)
        print("The Question Number: ",question_number)
        print("Reversed Letter: ",reverse_letter)
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
            .document(user_id) \
            .collection('Level_2') \
            .document(str(question_number))
        if reverse_letter in ['ᗺ','Ↄ']:
            return "correct"
        else:
            doc_ref.set({
                'Question Number': question_number,
                'Answer': 'Incorrect',
                },merge=True)
            doc_ref.update({
                'Error': ArrayUnion(['B','C'])
                })
            return "incorrect"







#A-LEVEL 3 QUESTION#11 14
@app.route("/check_answers_q11",methods=['POST'])
def check_answers_q11():
    user_id = request.form.get('user_id')
    answers_str = request.form.get('answers_list')
    question_number=request.form.get('question_number')


    print(f"\n(DEBUG) Assessment Level 3 Question 11 RECEIVED REQUEST: {user_id} and {answers_str} and {question_number}!!\n")
    
    if answers_str and user_id:
        answers_list = json.loads(answers_str)
        store_mcq_error(user_id,answers_list,question_number)
        

    return "error"


#A-LEVEL 3 QUESTION#15
@app.route("/predict_handwriting_batch",methods=['POST'])
def predict_handwriting_batch():
    user_id = request.form.get('user_id')
    target_word = request.form.get('target_word')       
    question_number = request.form.get('question_number') 
    uploaded_files = request.files.getlist("images")
    print(f"\n(DEBUG) Batch Prediction for User: {user_id}, Word: {target_word}, Files: {len(uploaded_files)}\n")
    if user_id and target_word and uploaded_files:
        try:
            db = get_db()
            doc_ref = db.collection('Assessment_Test') \
                        .document(user_id) \
                        .collection('Level_3') \
                        .document(str(question_number))
            
            detected_errors = []
            for i, file in enumerate(uploaded_files):
                if i >= len(target_word): 
                    break

                expected_char = target_word[i]
                
                # Convert image for model
                img = Image.open(file.stream).convert("RGB")
                if expected_char.lower() in ['a', 'i', 'g', 'r']:
                    pred_response = g_handler_letter(img)
                else:
                    pred_response = letter_predict(img)
                    
                model_predict = pred_response
                if 'prediction' in pred_response:
                    for x in pred_response['prediction']:
                        if isinstance(x, str):
                            model_predict = x
                
                print(f"Letter {i}: Expected '{expected_char}' vs Predicted '{model_predict}'")

                
                # Check if the model's prediction matches the expected letter.
                # We use .lower() to allow 'F' == 'f'

                if model_predict.lower() != expected_char.lower():
                    if (expected_char.lower()=='o' and (str(model_predict) =='O_caps' or str(model_predict)==0) or
                        expected_char.lower()=='b' and (str(model_predict) =='B_caps')):
                        continue
                    else:
                        error_msg =expected_char
                        detected_errors.append(error_msg)

            # Update Database based on results
            if detected_errors:
                print(f"Errors Found: {detected_errors}")
                
                # Mark Question as Incorrect
                doc_ref.set({
                    'Question Number': int(question_number),
                    'Answer': 'Incorrect',
                }, merge=True)

                # Add specific errors to the array
                doc_ref.update({
                    'Error': firestore.ArrayUnion(detected_errors)
                })
                
                return "Checked with Errors", 200
                

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return f"Server Error: {e}", 500

    return "Missing Data", 400



@app.route('/question16', methods=['POST'])
def question16():
    if 'audio' not in request.files or 'target_sound' not in request.form:
        return jsonify({"error": "Missing audio file or target_sound"}), 400
        
    target_sentence = request.form['target_sound'].lower().strip()
    target_words = re.sub(r'[^\w\s]', '', target_sentence).split()
    
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number', '16')
    
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        
        #  Transcribe the full sentence
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=8, language="en") 
        os.remove(filepath)
        
        raw_text = " ".join([segment["text"] for segment in result["segments"]]).strip().lower()
        print(f"Target: {target_sentence}")
        print(f"AI Heard: {raw_text}")
        heard_text_clean = re.sub(r'[^\w\s]', '', raw_text)
        heard_words = heard_text_clean.split()
        
        # Identify Incorrect Words
        # We compare word-by-word. 
        detected_errors = []
        
        # We iterate through the target words
        for i, target_w in enumerate(target_words):
            # Check if the AI actually heard a word at this position
            if i < len(heard_words):
                heard_w = heard_words[i]
                similarity = jellyfish.jaro_winkler_similarity(target_w, heard_w)
                print(f"Comparing [{target_w}] vs [{heard_w}] -> Similarity: {similarity:.2f}")
                
                if similarity < 0.65: 
                    detected_errors.append(target_w)
            else:
                # User skipped the word or AI didn't catch it
                detected_errors.append(target_w)

        # 3. Final Result Logic
        is_fully_correct = len(detected_errors) == 0


        # 4. Database Update
        if not is_fully_correct and user_id:
            db = get_db()
            doc_ref = db.collection('Assessment_Test') \
                        .document(user_id) \
                        .collection('Level_4') \
                        .document(str(question_number))
            
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Incorrect',
                'Full Sentence': target_sentence
            }, merge=True)

            # Store only the specific words that were wrong
            doc_ref.update({
                'Error': firestore.ArrayUnion(detected_errors)
            })

        return jsonify({
            "target_sentence": target_sentence,
            "transcribed_text": raw_text,
            "wrong_words": detected_errors,
            "is_correct": is_fully_correct
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    


#level 4 question 18 
@app.route("/check_answers_q18", methods=['POST'])
def check_answers_q18():
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number')
    answers_list_raw = request.form.get('answers_list') 
    
    if not user_id or not answers_list_raw:
        return "Missing Data", 400

    try:
        selected_words = json.loads(answers_list_raw)
        
        print(f"\n--- Question {question_number} Selection ---")
        print(f"User ID: {user_id}")
        print(f"Full Selection: {selected_words}")
        print("------------------------------------------\n")
        
        detected_errors = [word for word in selected_words if word.lower() != "bog"]
        
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
                    .document(user_id) \
                    .collection('Level_4') \
                    .document(str(question_number))

        if detected_errors:
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Incorrect',
                'Timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            # Store ONLY the errors in the Firebase array
            doc_ref.update({
                'Error': firestore.ArrayUnion(detected_errors)
            })
            return "Errors stored to Firebase; Full list printed to terminal", 200
        else:
            # If everything was correct, mark it so
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Correct',
                'Timestamp': firestore.SERVER_TIMESTAMP
            }, merge=True)
            return "Correct selection; Full list printed to terminal", 200

    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500
#level 4 Question 17 and 19





@app.route("/predict_handwriting_sentence", methods=['POST'])
def predict_handwriting_sentence():
    user_id = request.form.get('user_id')
    target_sentence = request.form.get('target_sentence')     
    question_number = request.form.get('question_number') 
    uploaded_files = request.files.getlist("images")

    print(f"\n(DEBUG) Sentence Prediction for User: {user_id}, Sentence: {target_sentence}, Files: {len(uploaded_files)}\n")

    if user_id and target_sentence and uploaded_files:
        try:
            db = get_db()
            doc_ref = db.collection('Assessment_Test') \
                        .document(user_id) \
                        .collection('Level_4') \
                        .document(str(question_number))
            
        
            expected_chars = [char for char in target_sentence if char.isalnum()]
            
            detected_errors = []

            for i, file in enumerate(uploaded_files):
                if i >= len(expected_chars): 
                    break

                expected_char = expected_chars[i]
                
                img = Image.open(file.stream).convert("RGB")
                if expected_char.lower() in ['a', 'i', 'g', 'r','w']:
                    pred_response = g_handler_letter(img)
                else:
                    pred_response = letter_predict(img)
                model_predict=pred_response 
                
                model_predict = ""
                if isinstance(pred_response, dict) and 'prediction' in pred_response:
                    for x in pred_response['prediction']:
                        if isinstance(x, str):
                            model_predict = x
                else:
                    model_predict = str(pred_response)
                
                print(f"Index {i}: Expected '{expected_char}' vs Predicted '{model_predict}'")

                if model_predict.lower() != expected_char.lower():
                    if model_predict.lower() != expected_char.lower():
                        if ((expected_char.lower()=='o' and (str(model_predict) =='O_caps' or str(model_predict)=='0')) or (expected_char.lower()=='s' and (str(model_predict) =='S_caps' or str(model_predict)=='5')) or (expected_char.lower()=='w' and str(model_predict) =='W_caps') or
                            (expected_char=='T' and str(model_predict)=='T_caps') or 
                            (expected_char=='b' and str(model_predict)=='B_caps') or 
                            (expected_char=='H' and str(model_predict)=='H_caps') or
                            (expected_char=='g' and str(model_predict)=='9')or
                            (expected_char=='m' and str(model_predict)=='n')):
                            continue
                        # Record the error
                        else:
                            error_msg =expected_char
                            detected_errors.append(error_msg)

            # --- DATABASE UPDATE ---
            if detected_errors:
                print(f"Errors Found in Sentence: {detected_errors}")
                
                # Mark as Incorrect and store specific character errors
                doc_ref.set({
                    'Question Number': int(question_number),
                    'Answer': 'Incorrect',
                }, merge=True)

                doc_ref.update({
                    'Error': firestore.ArrayUnion(list(set(detected_errors)))
                })
                print("UPDATED FIRESTORE")
                return "Sentence Checked with Errors", 200
            else:
                # Mark as Correct if all characters matched
                doc_ref.set({
                    'Question Number': int(question_number),
                    'Answer': 'Correct',
                     }, merge=True)
                print("UPDATED FIRESTORE")
                return "Sentence Correct", 200

        except Exception as e:
            print("ERROR FIRESTORE")
            print(f"Error in sentence prediction: {e}")
            return f"Server Error: {e}", 500

    return "Missing Data", 400

#question19 
@app.route('/transcribe_and_score1', methods=['POST'])
def transcribe_and_score1():
    if 'audio' not in request.files or 'target_word' not in request.form:
        return jsonify({"error": "Missing audio file or target_word"}), 400
    
    user_id = request.form.get('user_id')
    target_sentence = request.form['target_word'].lower().strip()
    # Clean target sentence for word splitting
    target_words = re.sub(r'[^\w\s]', '', target_sentence).split()
    
    question_number = request.form.get('question_number', '19')
    file = request.files['audio']
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # 1. Transcribe the full sentence
        audio = whisperx.load_audio(filepath)
        # Using batch_size=8 for faster sentence processing
        result = model.transcribe(audio, batch_size=8, language="en") 
        os.remove(filepath)
        
        # 2. Extract and clean the AI's transcription
        raw_text = " ".join([segment["text"] for segment in result["segments"]]).strip().lower()
        print(f"--- Q{question_number} Assessment ---")
        print(f"Target: {target_sentence}")
        print(f"AI Heard: {raw_text}")
        
        heard_text_clean = re.sub(r'[^\w\s]', '', raw_text)
        heard_words = heard_text_clean.split()
        
        detected_errors = []
        
        # Iterate through target words to maintain strict order
        for i, target_w in enumerate(target_words):
            if i < len(heard_words):
                heard_w = heard_words[i]
                similarity = jellyfish.jaro_winkler_similarity(target_w, heard_w)
                print(f"Pos {i}: [{target_w}] vs [{heard_w}] -> Sim: {similarity:.2f}")
                
                # If similarity is too low (< 0.65), it's a specific word error
                if similarity < 0.65:
                    detected_errors.append(target_w)
            else:
                detected_errors.append(target_w)

        # 4. Final Result Logic
        is_fully_correct = len(detected_errors) == 0

        # 5. Database Update (Firestore)
        if user_id:
            try:
                db = get_db()
                level ="Level_4"
                doc_ref = db.collection('Assessment_Test') \
                            .document(user_id) \
                            .collection(level) \
                            .document(str(question_number))
                
                if not is_fully_correct:
                    # Store as Incorrect with specific word errors
                    doc_ref.set({
                        'Question Number': int(question_number),
                        'Answer': 'Incorrect',
                        'Full Sentence': target_sentence,
                        'Transcribed': raw_text
                    }, merge=True)
                    
                    # Store only the specific words that were wrong
                    doc_ref.update({
                        'Error': firestore.ArrayUnion(detected_errors)
                    })
                    print(f"(DEBUG) Incorrect stored for Q{question_number}. Errors: {detected_errors}")
                else:
                    # Store as Correct
                    doc_ref.set({
                        'Question Number': int(question_number),
                        'Answer': 'Correct',
                        'Full Sentence': target_sentence
                    }, merge=True)
                    print(f"(DEBUG) Correct stored for Q{question_number}")

            except Exception as db_error:
                print(f"Firestore Update Failed: {db_error}")
        
        return jsonify({
            "target_sentence": target_sentence,
            "transcribed_text": raw_text,
            "wrong_words": detected_errors,
            "is_correct": is_fully_correct
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"General Audio Route Error: {e}")
        return jsonify({"error": str(e)}), 500
    
    
@app.route("/get_common_errors/<user_id>", methods=['GET'])
def get_common_errors(user_id):
    try:
        db = get_db()
        analytics_payload = []
        
        collection_scopes = [
            ('Assessment_Test', 'Initial Assessment'),
            ('Level', 'Therapy Practice'),
            ('Quiz', 'Active Quizzes')
        ]
        
        levels = ["Level_1", "Level_2", "Level_3", "Level_4"]

        # 1. Iterate through distinct learning modes
        for db_collection, ui_category in collection_scopes:
            for level_name in levels:
                sub_ref = db.collection(db_collection).document(user_id).collection(level_name)
                
                if db_collection == 'Assessment_Test':
                    docs = sub_ref.where("Answer", "==", "Incorrect").stream()
                else:
                    docs = sub_ref.stream()

                for doc in docs:
                    data = doc.to_dict() or {}
                    
                    raw_errors = data.get('Error', [])
                    if not isinstance(raw_errors, list):
                        raw_errors = [str(raw_errors)] if raw_errors else []
                        
                    clean_concepts = [str(e).strip() for e in raw_errors if str(e).strip()]
                    
                    if not clean_concepts:
                        continue

                    q_num = data.get('Question Number', doc.id)
                    domain_indicator = ""
                    if level_name == "Level_1": domain_indicator = "Spatial Orientation"
                    elif level_name == "Level_2": domain_indicator = "Letter Recognition"
                    elif level_name == "Level_3": domain_indicator = "Phonetic rhyming"
                    elif level_name == "Level_4": domain_indicator = "Sentence Context"

                    level_title = f"{level_name.replace('_', ' ')}: {domain_indicator} (Q{q_num})"

                    analytics_payload.append({
                        "source_category": ui_category,
                        "level_title": level_title,
                        "error_concepts": clean_concepts
                    })

        return jsonify(analytics_payload), 200

    except Exception as e:
        print(f"(CRITICAL) Common Error sync faulted for {user_id}:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal analytics compilation engine failed"}), 500


from typecast import Typecast
from typecast.models import TTSRequest
import random

#  Initialize the Typecast client
# REPLACE THIS WITH YOUR REAL API KEY FROM THE DASHBOARD
typecast_client = Typecast(api_key="__pltGcPSWdfiN4gPgxUwh2tFx4Efzw4wWCex4s3yCDQ8") 
CHARACTER_ID = "tc_645b39b760386589fd851133" # Your Doraemon-like character

CARTOON_VOICE_MAP = {
    "doraemon":"tc_645b39b760386589fd851133",
    "mickey": "tc_67db753311833db994c4fed7", # Your current Doraemon/Mickey voice
    "pooh": "tc_67db753311833db994c4fed7",       # TODO: Add your Typecast ID here
    "tom": "tc_660e5c11eef728e75f95f520",         # TODO: Add your Typecast ID here
    "duffy": "tc_replace_with_duffy_id"      # TODO: Add your Typecast ID here
}

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)



def generate_typecast_audio(text, filename, cartoon_name="doraemon"):
    try:
        direction_map = {
            r'\bne\b': "North East", r'\bnw\b': "North West",
            r'\bse\b': "South East", r'\bsw\b': "South West"
        }

        spoken_text = text
        for pattern, full_form in direction_map.items():
            spoken_text = re.sub(pattern, full_form, spoken_text, flags=re.IGNORECASE)

        # Look up the specific voice ID, default to Mickey if not found
        voice_id = CARTOON_VOICE_MAP.get(cartoon_name.lower(), CARTOON_VOICE_MAP["mickey"])

        response = typecast_client.text_to_speech(TTSRequest(
            text=spoken_text,
            model="ssfm-v30",
            voice_id=voice_id 
        ))

        filepath = os.path.join(AUDIO_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(response.audio_data)

        audio_url = f"http://{request.host}/static/audio/{filename}"
        return audio_url

    except Exception as e:
        print(f"(ERROR) Failed to generate Typecast audio: {e}")

def get_or_generate_audio(text, base_filename, cartoon_name="doraemon"):
    name, ext = os.path.splitext(base_filename)
    dynamic_filename = f"{name}_{cartoon_name.lower()}{ext}"
    
    filepath = os.path.join(AUDIO_DIR, dynamic_filename)
    if os.path.exists(filepath):
        return f"http://{request.host}/static/audio/{dynamic_filename}"
        
    return generate_typecast_audio(text, dynamic_filename, cartoon_name)


def ensure_word_audio_exists(word, cartoon_name="doraemon"):
    clean_word = word.strip().lower()
    filename = f"cached_word_v2_{clean_word}.wav"
    return get_or_generate_audio(word.strip().capitalize(), filename, cartoon_name)

def ensure_standalone_word_audio(word):
    """
    Guarantees the existence of hardcoded frontend UI audio files.
    Bypasses dynamic character suffixes so the Android app can find them via its hardcoded URL.
    Checks local storage first to prevent wasting Typecast API quota.
    """
    clean_word = word.strip().lower()
    filename = f"cached_word_v2_{clean_word}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"(INFO) Standalone word missing for UI: {word}. Generating once...")
        # Force generation using default voice, saving to the hardcoded filename
        generate_typecast_audio(word.strip().upper(), filename, "doraemon")

SLOT_CONFIGS = {
    1: {"type": "DRAWING", "prefix": "Draw the arrow"},
    2: {"type": "MCQ",     "prefix": "Click the"},     # Strictly MCQ
    3: {"type": "MCQ", "prefix": "Click the direction of given arrow ?"},
    4: {"type": "MCQ", "prefix":"Match the arrow to the correct word"},
}

def get_user_cartoon_preference(user_id):
    """
    Fetches the user's preferred cartoon helper from Firestore.
    Defaults to 'mickey' if the document or field does not exist.
    """
    try:
        db = get_db()
        user_doc = db.collection('cartoon_selection').document(user_id).get()
        if user_doc.exists:
            data = user_doc.to_dict() or {}
            return data.get('cartoon', 'mickey').lower().strip()
    except Exception as e:
        print(f"Error fetching cartoon preference for {user_id}: {e}")
    
    return 'mickey'



@app.route('/init_level_session', methods=['GET'])
def init_level_session():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        db = get_db()
        session_questions = []
        MASTERY_THRESHOLD = 3 

        # Fetch the user's dynamic cartoon selection once at session start
        user_cartoon = get_user_cartoon_preference(user_id)

        
        all_directions = ["Up", "Down", "Left", "Right", "NE", "NW", "SE", "SW"]
        for direction in all_directions:
            ensure_standalone_word_audio(direction)
        
        meta_ref = db.collection('Level').document(user_id).collection('Level_1').document('meta_status')
        meta_doc = meta_ref.get()
        meta_data = meta_doc.to_dict() or {}
        
        is_graduated = meta_data.get('graduated', False)
        has_imported_assessment = meta_data.get('Assessment_Imported', False)

        active_error_pool = []
        imported_any_new = False
        
        for q_num in range(1, 5): 
            doc_id = str(q_num)
            active_ref = db.collection('Level').document(user_id).collection('Level_1').document(doc_id)
            active_doc = active_ref.get()
            target_word = None

            if active_doc.exists:
                data = active_doc.to_dict() or {}
                current_success = data.get('success_count', 0)
                if current_success < MASTERY_THRESHOLD:
                    target_word = data.get("Error", ["Up"])[0]
            elif not has_imported_assessment:
                assess_ref = db.collection('Assessment_Test').document(user_id).collection('Level_1').document(doc_id)
                assess_doc = assess_ref.get()
                
                if assess_doc.exists:
                    assess_data = assess_doc.to_dict() or {}
                    raw_errors = assess_data.get("Error", [])
                    if assess_data.get("Answer") == "Incorrect" and raw_errors:
                        target_word = raw_errors[0] if isinstance(raw_errors, list) else str(raw_errors)
                        
                        active_ref.set({
                            'Question Number': q_num,
                            'Error': [target_word.strip().capitalize()],
                            'success_count': 0,
                        })
                        imported_any_new = True

            if target_word:
                active_error_pool.append({
                    "original_db_slot": q_num,
                    "word": target_word.strip().capitalize()
                })

        if imported_any_new:
            meta_ref.set({'Assessment_Imported': True}, merge=True)
            has_imported_assessment = True

        if not active_error_pool and not is_graduated:
            meta_ref.set({
                'graduated': True,
                'unlocked_level_2': True,
                'Assessment_Imported': True
            }, merge=True)
            
            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon, # Inject selection here
                "total_questions": 0,
                "maintenance_mode": False,
                "questions": []
            }), 200

        if is_graduated and not active_error_pool:
            directions = ["Up", "Down", "Left", "Right", "NE", "NW", "SE", "SW"]
            rand_dir = random.choice(directions)
            q_type = random.choice(["DRAWING", "MCQ_GRID", "MCQ_IDENTIFY", "MCQ_MATCH"])
            clean_word = rand_dir.lower()

            if q_type == "DRAWING":
                instruction_text = f"Draw the arrow {rand_dir}"
                audio_filename = f"cached_draw_{clean_word}.wav"
                mapped_type = "DRAWING"
                mapped_slot = 99 
            elif q_type == "MCQ_GRID":
                instruction_text = f"Click the {rand_dir} Arrow"
                audio_filename = f"cached_click_{clean_word}.wav"
                mapped_type = "MCQ"
                mapped_slot = 2  
            elif q_type == "MCQ_IDENTIFY":
                rand_dir = random.choice(["Up", "Down", "Left", "Right"])
                clean_word = rand_dir.lower()
                instruction_text = "Click the direction of the given arrow"
                audio_filename = f"cached_identify_{clean_word}.wav"
                mapped_type = "MCQ"
                mapped_slot = 3  
            else: # MCQ_MATCH
                rand_dir = random.choice(["Up", "Down", "Left", "Right"])
                clean_word = rand_dir.lower()
                instruction_text = "Match the arrow to the correct word"
                audio_filename = f"cached_match_v2_{clean_word}.wav"
                mapped_type = "MCQ"
                mapped_slot = 4

            # ---> FIX: Passed user_cartoon here so maintenance mode sounds right <---
            audio_url = get_or_generate_audio(instruction_text, audio_filename, user_cartoon)

            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon,
                "total_questions": 1,
                "maintenance_mode": True,
                "questions": [{
                    "db_question_number": mapped_slot, 
                    "question_type": mapped_type,
                    "ui_slot_assigned": mapped_slot,
                    "target_word": rand_dir,
                    "instruction_text": instruction_text,
                    "audio_url": audio_url
                }]
            }), 200

        random.shuffle(active_error_pool)
        available_ui_slots = sorted(list(SLOT_CONFIGS.keys()))
        
        for index, error_obj in enumerate(active_error_pool):
            if index >= len(available_ui_slots):
                break
                
            ui_slot_num = available_ui_slots[index]
            config = SLOT_CONFIGS[ui_slot_num]
            target_word = error_obj["word"]
            clean_word = target_word.lower()

            if config["type"] == "MCQ":
                if ui_slot_num == 3:
                    instruction_text = "Click the direction of the given arrow"
                    audio_filename = f"cached_identify_v2_{clean_word}.wav"
                elif ui_slot_num == 4:
                    instruction_text = "Match the arrow to the correct word"
                    audio_filename = f"cached_match_v2_{clean_word}.wav"
                else:
                    instruction_text = f"{config['prefix']} {target_word} Arrow"
                    audio_filename = f"cached_click_v2_{clean_word}.wav"
            else:
                instruction_text = f"{config['prefix']} {target_word}"
                audio_filename = f"cached_draw_v2_{clean_word}.wav" 

            audio_url = get_or_generate_audio(instruction_text, audio_filename, user_cartoon)

            session_questions.append({
                "db_question_number": error_obj["original_db_slot"],  
                "question_type": config["type"], 
                "ui_slot_assigned": ui_slot_num, 
                "target_word": target_word,
                "instruction_text": instruction_text,
                "audio_url": audio_url
            })

        return jsonify({
            "status": "success",
            "cartoon_selection": user_cartoon, 
            "total_questions": len(session_questions),
            "maintenance_mode": False,
            "questions": session_questions
        }), 200

    except Exception as e:
        print(f"(ERROR) /init_level_session failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to initialize therapy session"}), 500

@app.route('/predict_therapy_direction', methods=['POST'])
def predict_therapy_direction():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id')
    target_word = request.form.get('target_word')
    question_number = request.form.get('question_number', '1') 

    if not user_id or not target_word:
        return jsonify({"error": "Missing critical therapy data"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")  
        model_prediction = direction_predict(img)
        is_correct = (model_prediction.strip().lower() == target_word.strip().lower())
        
        _, mastery_status = update_therapy_progress(
            user_id=user_id,
            target_word=target_word,
            is_correct=is_correct,
            question_number=question_number,
            level_name="Level_1"
        )

        return jsonify({
            "status": "success", 
            "correct": is_correct, 
            "detected": model_prediction,
            "target": target_word,
        }), 200

    except Exception as e:
        print(f"(ERROR) /predict_therapy_direction failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500



# --- GET ROUTE FOR Q2 DYNAMIC MCQ ---
@app.route('/verify_therapy_mcq', methods=['POST'])
def verify_therapy_mcq():
    user_id = request.form.get('user_id')
    target_word = request.form.get('target_word')
    arrow_selected = request.form.get('arrow_selected')
    
    question_number = request.form.get('question_number', '2')

    if not all([user_id, target_word, arrow_selected]):
        return jsonify({"error": "Missing critical data"}), 400

    try:
        is_correct = (arrow_selected.strip().lower() == target_word.strip().lower())
        
        _, mastery_status = update_therapy_progress(
            user_id=user_id,
            target_word=target_word,
            is_correct=is_correct,
            question_number=question_number,
            level_name="Level_1"
        )

        return jsonify({
            "status": "success", 
            "correct": is_correct, 
            "selected": arrow_selected,
            "target": target_word,
            "mastery_status": mastery_status  
        }), 200

    except Exception as e:
        print(f"(ERROR) /verify_therapy_mcq failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500




# --- POST ROUTE TO VERIFY Q3 SELECTION ---
@app.route('/verify_therapy_q3', methods=['POST'])
def verify_therapy_q3():
    user_id = request.form.get('user_id')
    target_word = request.form.get('target_word')
    arrow_selected = request.form.get('arrow_selected')
    question_number = request.form.get('question_number', '3')

    if not all([user_id, target_word, arrow_selected]):
        return jsonify({"error": "Missing critical data"}), 400

    try:
        is_correct = (arrow_selected.strip().lower() == target_word.strip().lower())
        
        _, mastery_status = update_therapy_progress(
            user_id=user_id,
            target_word=target_word,
            is_correct=is_correct,
            question_number=question_number,
            level_name="Level_1"
        )

        return jsonify({
            "status": "success", 
            "correct": is_correct, 
            "selected": arrow_selected,
            "target": target_word,
            "mastery_status": mastery_status  
        }), 200

    except Exception as e:
        print(f"(ERROR) /verify_therapy_q3 failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500


# --- POST ROUTE TO VERIFY Q4 SELECTION ---
@app.route('/verify_therapy_q4', methods=['POST'])
def verify_therapy_q4():
    user_id = request.form.get('user_id')
    target_word = request.form.get('target_word')
    arrow_selected = request.form.get('arrow_selected')
    question_number = request.form.get('question_number', '4')

    if not all([user_id, target_word, arrow_selected]):
        return jsonify({"error": "Missing matching parameters"}), 400

    try:
        is_correct = (arrow_selected.strip().lower() == target_word.strip().lower())
        
        _, mastery_status = update_therapy_progress(
            user_id=user_id,
            target_word=target_word,
            is_correct=is_correct,
            question_number=question_number,
            level_name="Level_1"
        )

        return jsonify({
            "status": "success", 
            "correct": is_correct, 
            "validated_selection": arrow_selected.strip().capitalize(),
            "target": target_word.strip().capitalize(),
            "mastery_status": mastery_status  
        }), 200

    except Exception as e:
        print(f"(ERROR) /verify_therapy_q4 failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

#QUIZ1
@app.route('/generate_quiz1', methods=['GET'])
def generate_quiz1():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    try:
        quiz_questions = []
        
        cardinal_dirs = ["Up", "Down", "Left", "Right"]
        combined_dirs = ["NE", "NW", "SE", "SW","Up","Down","Left","Right"]
        
        for direction in (cardinal_dirs + combined_dirs):
            ensure_word_audio_exists(direction)

        target_slots = [1, 2, 3, 4]
        
        random.shuffle(cardinal_dirs)
        random.shuffle(combined_dirs)

        for index, slot_num in enumerate(target_slots):
            if slot_num == 1:
                target_word = cardinal_dirs[0] 
            elif slot_num==3:
                target_word = combined_dirs[index - 1] 
                
            clean_word = target_word.lower()

            if slot_num == 1:
                q_type = "DRAWING"
                instruction_text = f"Draw the arrow {target_word}"
                audio_filename = f"cached_quiz_draw_{clean_word}.wav"
            elif slot_num == 2:
                q_type = "MCQ"
                instruction_text = f"Click the {target_word} Arrow"
                audio_filename = f"cached_quiz_click_{clean_word}.wav"
            elif slot_num == 3:
                q_type = "MCQ"
                instruction_text = "Click the direction of the given arrow"
                audio_filename = f"cached_identify_v2_down.wav"
            else: 
                q_type = "MCQ"
                instruction_text = "Match the arrow to the correct word"
                audio_filename = f"cached_match_v2_left.wav"

            audio_url = get_or_generate_audio(instruction_text, audio_filename)

            quiz_questions.append({
                "db_question_number": slot_num,  
                "question_type": q_type,
                "ui_slot_assigned": slot_num,
                "target_word": target_word,
                "instruction_text": instruction_text,
                "audio_url": audio_url
            })

        return jsonify({
            "status": "success",
            "total_questions": len(quiz_questions),
            "quiz_mode": True,
            "questions": quiz_questions
        }), 200

    except Exception as e:
        print(f"(ERROR) /generate_quiz1 failed runtime compilation:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to compile randomized testing parameters"}), 500



@app.route('/evaluate_quiz1_batch', methods=['POST'])
def evaluate_quiz1_batch():
    try:
        user_id = request.form.get('user_id')
        metadata_str = request.form.get('metadata')

        if not user_id or not metadata_str:
            return jsonify({"error": "Missing payload routing configurations"}), 400

        db = get_db()
        metadata_list = json.loads(metadata_str)
        
        total_questions = len(metadata_list)
        correct_count = 0
        failed_concepts = []

        # Iterate over submitted sequence entries
        for meta in metadata_list:
            q_index = meta['question_index']
            target_word = meta['target_word']
            q_type = meta['question_type']
            db_slot = meta['db_question_number']

            is_correct = False

            if q_type == "DRAWING":
                file_key = f"file_{q_index}"
                if file_key in request.files:
                    image_file = request.files[file_key]
                    
                    img = Image.open(image_file.stream).convert("RGB")
                    
                    prediction = direction_predict(img) 
                    if prediction.strip().lower() == target_word.strip().lower():
                        is_correct = True
            elif q_type == "MCQ":
                file_key = f"file_{q_index}"
                if file_key in request.files:
                    selection_bytes = request.files[file_key].read()
                    selected_arrow = selection_bytes.decode('utf-8').strip()
                    
                    if selected_arrow.lower() == target_word.strip().lower():
                        is_correct = True
            else:
                is_correct = True

            if is_correct:
                correct_count += 1
            else:
                failed_concepts.append({
                    "slot": db_slot,
                    "concept": target_word.strip().capitalize()
                })

        # Calculate final percentage distribution
        score_ratio = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        is_eligible_for_level_2 = score_ratio >= 75.0  # Pass threshold mapping set to 75%

        # 1. WRITE TO QUIZ TRACKING TABLE
        quiz_ref = db.collection('Quiz').document(user_id).collection('Quiz 1').document('Score')
        quiz_ref.set({
            'score': correct_count,
            'total_questions': total_questions,
            'percentage': score_ratio,
            'passed': is_eligible_for_level_2,
            'timestamp': firestore.SERVER_TIMESTAMP
        }, merge=True)

        # 2. WRITE TO LEVEL 2 ACCESS PERMISSION NODE
        meta_ref = db.collection('Level').document(user_id).collection('Level_1').document('meta_status')
        meta_ref.set({
            'unlocked_level_2': is_eligible_for_level_2
        }, merge=True)

        # 3. WRITE IDENTIFIED ERRORS BACK TO LEVEL 1 PRACTICE POOLS
        for err_item in failed_concepts:
            slot_id = str(err_item['slot'])
            active_practice_ref = db.collection('Level').document(user_id).collection('Level_1').document(slot_id)
            
            active_practice_ref.set({
                'Question Number': err_item['slot'],
                'Error': [err_item['concept']],
                'success_count': 0  
            }, merge=True)

        return jsonify({
            "status": "success",
            "final_score": correct_count,
            "passed": is_eligible_for_level_2,
            "errors_logged": len(failed_concepts)
        }), 200

    except Exception as e:
        print(f"(CRITICAL) Batch processing faulted:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to compile evaluations batch"}), 500



#----------------LEVEL 4------------------
#----------------------------------------

import pandas as pd
from werkzeug.utils import secure_filename

import os
import random
import re
import traceback
import pandas as pd

def get_l4_targets_from_csv(error_list, max_count=3, mode="VOICE"):
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        csv_path = os.path.join(base_dir, 'models', 'DyslexiaDataSet.csv')
        
        if not os.path.exists(csv_path):
            csv_path = 'models/DyslexiaDataSet.csv'

        df = pd.read_csv(csv_path, header=None, names=['raw_text'])
        df = df.dropna()
        
        df['raw_text'] = df['raw_text'].astype(str).apply(lambda x: " ".join(x.split()))
        
        valid_errors = [str(w).strip().lower() for w in error_list if str(w).strip()]
        if not valid_errors:
            valid_errors = ["bat", "cat", "rat"]

        results = []
        used_sentences = set() 
        
        for i in range(max_count):
            target_word = valid_errors[i % len(valid_errors)]
            
            def count_words(text):
                return len(text.split())
                
            def count_boxes(text):
                return sum(1 for c in text if c.isalnum())
            
            pattern = r'\b' + re.escape(target_word) + r'\b'
            matches = df[df['raw_text'].str.lower().str.contains(pattern, regex=True)]
            
            if matches.empty:
                matches = df[df['raw_text'].str.lower().str.contains(re.escape(target_word), regex=True)]
            
            if mode == "GRID_SELECT":
                valid_matches = matches
            elif mode == "VOICE":
                valid_matches = matches[matches['raw_text'].apply(count_words) >= 3]
            else: 
                valid_matches = matches[
                    (matches['raw_text'].apply(count_words) >= 2) & 
                    (matches['raw_text'].apply(count_boxes) <= 16)
                ]
            
            if valid_matches.empty:
                print(f"(WARN) '{target_word}' missing exact match. Executing Jaro-Winkler multi-sample scan...")
                fallback_candidates = []
                
                for raw_sentence in df['raw_text'].values:
                    if mode == "VOICE" and count_words(raw_sentence) < 3:
                        continue
                    if mode in ["WRITING", "COMBO"] and (count_words(raw_sentence) < 2 or count_boxes(raw_sentence) > 16):
                        continue
                        
                    tokens = re.sub(r'[^\w\s]', '', raw_sentence).lower().split()
                    for token in tokens:
                        score = jellyfish.jaro_winkler_similarity(target_word, token)
                        if score >= 0.70:
                            fallback_candidates.append(raw_sentence)
                            break 
                
                if fallback_candidates:
                    valid_matches = pd.DataFrame(fallback_candidates, columns=['raw_text'])
                else:
                    valid_matches = df 

            available = valid_matches[~valid_matches['raw_text'].isin(used_sentences)]
            
            if available.empty:
                print(f"(INFO) Exhausted unique contexts for '{target_word}'. Halting extraction at {len(results)} items.")
                break 

            sampled_sentence = available.sample(n=1)['raw_text'].values[0]
            used_sentences.add(sampled_sentence)
            
            if mode == "GRID_SELECT":
                all_words = " ".join(df['raw_text'].values).split()
                distractors = random.sample(all_words, 23)
                grid_pool = distractors + [target_word]
                random.shuffle(grid_pool)
                final_string = " ".join(grid_pool)
            else:
                clean_text = re.sub(r'[^\w\s]', '', sampled_sentence)
                final_string = " ".join(clean_text.split())

            results.append({
                "word": target_word.capitalize(),
                "sentence": final_string
            })
            
        if not results:
            results.append({"word": valid_errors[0].capitalize(), "sentence": f"Practice the word {valid_errors[0]}"})
            
        return results

    except Exception as e:
        print(f"(CRITICAL ERROR) Pandas CSV Engine failed: {str(e)}")
        traceback.print_exc()
        return [{"word": "Fallback", "sentence": "System loaded a default string"}]


# --- 3. LEVEL 4 INITIALIZATION ROUTE (STRICT SEQUENTIAL MULTIMODAL ORDER) ---
@app.route('/init_level4_session', methods=['GET'])
def init_level4_session():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try: 
        db = get_db()
        session_questions = []
        MASTERY_THRESHOLD = 3 

        # 1. Fetch User Character Preference
        user_cartoon = get_user_cartoon_preference(user_id)

        meta_ref = db.collection('Level').document(user_id).collection('Level_4').document('meta_status')
        meta_doc = meta_ref.get()
        is_graduated = meta_doc.to_dict().get('graduated', False) if meta_doc.exists else False

        slot_allocations = {
            16: {"ui_slot": 1, "type": "VOICE", "instruction": "Read the sentence out loud"},
            17: {"ui_slot": 2, "type": "WRITING", "instruction": "Rewrite the sentence below"},
            18: {"ui_slot": 3, "type": "GRID_SELECT", "instruction": "Find and click the target word in the grid."},
            19: {"ui_slot": 4, "type": "COMBO", "instruction": "Read the word aloud, then type it out."}
        }

        active_error_pool = []

        if not is_graduated:
            for q_num in range(16, 20):
                doc_id = str(q_num)
                active_ref = db.collection('Level').document(user_id).collection('Level_4').document(doc_id)
                active_doc = active_ref.get()
                error_list_payload = []

                if active_doc.exists:
                    data = active_doc.to_dict() or {}
                    raw_err = data.get("Error", [])
                    scores_map = data.get("scores_tracker", {})
                    
                    for item in raw_err if isinstance(raw_err, list) else [raw_err]:
                        clean_str = str(item).strip().lower()
                        if clean_str and scores_map.get(clean_str, 0) < MASTERY_THRESHOLD:
                            error_list_payload.append(clean_str)
                else:
                    assess_ref = db.collection('Assessment_Test').document(user_id).collection('Level_4').document(doc_id)
                    assess_doc = assess_ref.get()
                    if assess_doc.exists:
                        assess_data = assess_doc.to_dict() or {}
                        raw_err = assess_data.get("Error", []) if assess_data else []
                        error_list_payload = [str(w).strip().lower() for w in (raw_err if isinstance(raw_err, list) else [raw_err]) if str(w).strip()]
                        
                        if error_list_payload:
                            init_scores = {w: 0 for w in error_list_payload}
                            active_ref.set({
                                'Question Number': q_num,
                                'Error': error_list_payload,
                                'scores_tracker': init_scores,
                                'success_count': 0,
                            })

                if error_list_payload:
                    active_error_pool.append({
                        "original_db_slot": q_num,
                        "errors": error_list_payload
                    })

            if len(active_error_pool) == 0:
                meta_ref.set({'graduated': True}, merge=True)
                is_graduated = True

        if is_graduated:
            mini_q_data = get_l4_targets_from_csv(dummy_errors, max_count=3, mode="VOICE")
            instruction = "Read the sentence out loud"
            # Pass user_cartoon to get the correct voice
            audio_url = get_or_generate_audio(instruction, "l4_maint_sentence.wav", user_cartoon)
            
            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon,
                "total_questions": 1,
                "maintenance_mode": True,
                "questions": [{
                    "db_question_number": 99,
                    "question_type": "VOICE",
                    "ui_slot_assigned": 1,
                    "mini_questions": mini_q_data, 
                    "instruction_text": instruction,
                    "audio_url": audio_url
                }]
            }), 200

        for error_obj in active_error_pool:
            db_num = error_obj["original_db_slot"]
            current_error_array = error_obj["errors"]
            config = slot_allocations[db_num]
            
            ui_slot = config["ui_slot"]
            q_type = config["type"]
            instruction = config["instruction"]
            
            mini_q_data = get_l4_targets_from_csv(current_error_array, max_count=3, mode=q_type)
             
            clean_audio_name = f"l4_slot{ui_slot}.wav"
            audio_url = get_or_generate_audio(instruction, clean_audio_name, user_cartoon)

            session_questions.append({
                "db_question_number": db_num, 
                "question_type": q_type,
                "ui_slot_assigned": ui_slot,  
                "mini_questions": mini_q_data, 
                "instruction_text": instruction,
                "audio_url": audio_url
            })

        return jsonify({
            "status": "success",
            "cartoon_selection": user_cartoon,
            "total_questions": len(session_questions),
            "maintenance_mode": False,
            "questions": session_questions
        }), 200

    except Exception as e:
        print(f"(ERROR) /init_level4_session failed:\n{traceback.format_exc()}")
        return jsonify({"error": "Failed to initialize Level 4 session"}), 500
@app.route('/verify_l4_q1_voice', methods=['POST'])
def verify_l4_q1_voice():
    """
    Receives spoken audio payloads, transcribes the string using WhisperX, 
    and applies Jaro-Winkler phonetic token evaluation against the active target key.
    """
    if 'audio' not in request.files or 'target_sentence' not in request.form:
        return jsonify({"error": "Missing audio or target_sentence payload"}), 400
        
    user_id = request.form.get('user_id')
    target_sentence = request.form['target_sentence'].lower().strip()
    target_word = request.form.get('target_word', 'bat').lower().strip()
    question_number = request.form.get('question_number', '1')
    
    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=8, language="en") 
        os.remove(filepath)
        
        raw_text = " ".join([segment["text"] for segment in result["segments"]]).strip().lower()
        clean_transcribed = re.sub(r'[^\w\s]', '', raw_text)
        
        print("\n" + "="*50)
        print(f"(DEBUG WHISPERX) Target Key     : '{target_word}'")
        print(f"(DEBUG WHISPERX) Mined Sentence : '{target_sentence}'")
        print(f"(DEBUG WHISPERX) Spoken Audio   : '{clean_transcribed}'")
        print("="*50 + "\n")
        
        transcribed_tokens = clean_transcribed.split()
        best_token_score = 0.0
        matched_token = ""
        
        # Jaro-Winkler handles slight mispronunciations or accent shifts gracefully
        for token in transcribed_tokens:
            score = jellyfish.jaro_winkler_similarity(target_word, token)
            if score > best_token_score:
                best_token_score = score
                matched_token = token
                
        print(f"Similarity sentence for {target_word} is {best_token_score}")
        is_correct = bool(best_token_score >= 0.65)
        
        _, mastery = update_therapy_progress_l4(user_id, target_word, is_correct, question_number)
            
        return jsonify({
            "status": "success",
            "target_sentence": target_sentence,
            "transcribed_sentence": clean_transcribed,
            "target_word": target_word,
            "matched_token": matched_token,
            "similarity_score": best_token_score,
            "correct": is_correct,
            "mastery_status": mastery
        }), 200
        
    except Exception as e:
        print(f"(CRITICAL ERROR) /verify_l4_q1_voice failed: {str(e)}")
        return jsonify({"error": str(e)}), 500




# --- LEVEL 4 DYNAMIC WRITING VERIFICATION ROUTE (AUTOMATIC CAPS NORMALIZATION) ---
@app.route('/verify_l4_q2_writing', methods=['POST'])
def verify_l4_q2_writing():
    user_id = request.form.get('user_id')
    target_sentence = request.form.get('target_sentence', '').lower().strip()
    target_word = request.form.get('target_word', 'bat').lower().strip()
    question_number = request.form.get('question_number', '17')
    
    if not target_sentence:
        return jsonify({"error": "Missing target_sentence payload"}), 400
        
    uploaded_files = request.files.getlist("images")
            
    print(f"\n(DEBUG L4 WRITING) User: {user_id} | Target Sentence: '{target_sentence}' | Active Word: '{target_word}' | Char Streams: {len(uploaded_files)}")

    if not uploaded_files:
        return jsonify({"error": "No drawing payloads extracted from field 'images'"}), 400

    try:
        expected_chars = [c for c in target_sentence if c.isalnum()]
        assembled_chars = []
        
        for i, file_storage in enumerate(uploaded_files):
            if i >= len(expected_chars):
                break
                
            expected_char = expected_chars[i]
            img = Image.open(file_storage.stream).convert("RGB")
            
            if expected_char.lower() in ['a', 'i', 'g', 'r', 'w']:
                pred_response = g_handler_letter(img)
            else:
                pred_response = letter_predict(img)
                
            model_predict = ""
            if isinstance(pred_response, dict) and 'prediction' in pred_response:
                for x in pred_response['prediction']:
                    if isinstance(x, str):
                        model_predict = x
            else:
                model_predict = str(pred_response)
                
            if model_predict.lower().endswith("_caps"):
                model_predict = model_predict.split("_")[0]

            clean_pred = model_predict.lower()
            clean_exp = expected_char.lower()
            
            if clean_pred != clean_exp:
                if ((clean_exp == 'o' and clean_pred == '0') or 
                    (clean_exp == 's' and clean_pred == '5') or 
                    (clean_exp == 'g' and clean_pred == '9')):
                    clean_pred = clean_exp 
                    
            assembled_chars.append(clean_pred if clean_pred else "?")
            print(f"  -> Slot {i}: Expected '{expected_char}' vs Raw AI '{model_predict}' -> Mapped '{clean_pred}'")

        raw_recognized_text = "".join(assembled_chars)
        clean_recognized = re.sub(r'[^\w\s]', '', raw_recognized_text)
        
        print("\n" + "="*50)
        print(f"(DEBUG L4 OCR) Target Word Key : '{target_word}'")
        print(f"(DEBUG L4 OCR) Full String Heard : '{clean_recognized}'")
        print("="*50 + "\n")
        
        best_token_score = 0.0
        matched_token = ""
        
        transcribed_tokens = clean_recognized.split() if " " in clean_recognized else [clean_recognized]
        
        for token in transcribed_tokens:
            score = jellyfish.jaro_winkler_similarity(target_word, token)
            if score > best_token_score:
                best_token_score = score
                matched_token = token
        print(f"Similarity sentence for {target_word} is {best_token_score}")      
        is_correct = bool(best_token_score >= 0.65)
        
        _, mastery = update_therapy_progress_l4(user_id, target_word, is_correct, question_number)
        
    
        return jsonify({
            "status": "success",
            "target_sentence": target_sentence,
            "transcribed_sentence": clean_recognized,
            "target_word": target_word,
            "matched_token": matched_token,
            "similarity_score": best_token_score,
            "correct": is_correct,
            "mastery_status": mastery
        }), 200

    except Exception as e:
        print(f"(CRITICAL ERROR) /verify_l4_q2_writing execution failed: {str(e)}")
        return jsonify({"error": f"Internal execution failure: {str(e)}"}), 500


# --- LEVEL 4 ISOLATED VERIFICATION ENDPOINT (GRID VALIDATION) ---
@app.route('/verify_l4_q3_grid', methods=['POST'])
def verify_l4_q3_grid():
    user_id = request.form.get('user_id')
    target_sentence = request.form.get('target_sentence', '').lower().strip()
    target_word = request.form.get('target_word', 'bat').lower().strip()
    question_number = request.form.get('question_number', '18')
    raw_answers_json = request.form.get('answers_list', '[]')
    
    if not target_sentence:
        return jsonify({"error": "Missing target_sentence payload"}), 400
        
    try:
        selected_answers = json.loads(raw_answers_json)
        clean_answers = [re.sub(r'[^\w\s]', '', str(w)).strip().lower() for w in selected_answers if str(w).strip()]
        
        print("\n" + "="*50)
        print(f"(DEBUG L4 GRID) Target Word Key : '{target_word}'")
        print(f"(DEBUG L4 GRID) Selected Tokens : {clean_answers}")
        print("="*50 + "\n")
        
        grid_pool_words = [w.strip().lower() for w in target_sentence.split()]
        total_targets_in_grid = grid_pool_words.count(target_word)
        
        correct_clicks = 0
        incorrect_clicks = 0
        
        for answer_token in clean_answers:
            if answer_token == target_word:
                correct_clicks += 1
            else:
                incorrect_clicks += 1
        
        is_correct = bool((correct_clicks == total_targets_in_grid) and (incorrect_clicks == 0))
        
        _, mastery = update_therapy_progress_l4(user_id, target_word, is_correct, question_number)
        

        return jsonify({
            "status": "success",
            "target_sentence": target_sentence,
            "selected_answers": clean_answers,
            "target_word": target_word,
            "correct": is_correct,
            "mastery_status": mastery
        }), 200

    except Exception as e:
        print(f"(CRITICAL ERROR) /verify_l4_q3_grid execution failed: {str(e)}")
        return jsonify({"error": f"Internal execution failure: {str(e)}"}), 500






#---LEVEL 2--------------------

# CARTOON SELECTION
@app.route("/select_cartoon", methods=["POST"])
def select_cartoon():
    user_id = request.form.get('user_id')
    cartoon_name = request.form.get('cartoon_name')
    
    if user_id and cartoon_name:
        print(f"\n(DEBUG) Cartoon Selection Received: User={user_id}, Cartoon={cartoon_name}\n")
        
        success = store_cartoon_selection(user_id, cartoon_name)
        
        if not success:
            return "Database Error"
    try: 
        db = get_db()
        level2_ref = db.collection('Assessment_Test') \
                    .document(user_id) \
                    .collection('Level_2')

        docs = level2_ref.stream()

        all_errors = []
        for doc in docs:

            doc_data = doc.to_dict()

            if doc_data.get("Answer") == "Incorrect":

                errors = doc_data.get("Error", [])

                if isinstance(errors, list):
                    all_errors.extend(errors)
        letters = []

        for word in all_errors:

            clean_word = re.sub(r'[^a-zA-Z]', '', word)

            for ch in clean_word:
                letters.append(ch.lower())

        # Remove duplicates
        unique_letters = list(dict.fromkeys(letters))
        level2_ref = db.collection('Level') \
            .document(user_id) \
            .collection('Level 2')

        for letter in unique_letters:
            level2_ref.document(letter).set({
                "letter": letter,
                "count": 0
            })
        print("User id: ", user_id)
        print("User Errors: ", unique_letters)
        random_letters = random.sample(unique_letters, min(3, len(unique_letters)))
        # All lowercase alphabets
        all_letters = list(string.ascii_lowercase)

        # Find letters NOT already in random_letters
        remaining_letters = [ch for ch in all_letters if ch not in random_letters]
        extra_letters = random.sample(remaining_letters, 2)
        random_letters.extend(extra_letters)
        level2_ref = db.collection('Quiz') \
            .document(user_id) \
            .collection('Quiz 2')
        for letter in random_letters:
            level2_ref.document(letter).set({
                "letter": letter
            })
        


    except Exception as e:
        return jsonify({"error": str(e)}), 500
            
    return "Error: Missing data"



@app.route('/TherapyLevel2', methods=['POST'])
def TherapyLevel2():
    print("Backend reached!")
    user_id = request.form.get('user_id')
    print(f"User ID received: {user_id}")
    if not user_id:
        return jsonify({"error": "User ID missing"}), 400
    
    db = get_db()

    # 1. Fetch user's cartoon preference securely
    user_cartoon = get_user_cartoon_preference(user_id)

    # 2. Extract active Level 2 Target Pool
    level2_ref = db.collection('Level') \
        .document(user_id) \
        .collection('Level 2')

    docs = level2_ref.stream()
    letters_array = []

    for doc in docs:
        letters_array.append(doc.id)
    
    print(letters_array)
    
    return jsonify({
        "status": "success",
        "cartoon_selection": user_cartoon,
        "letters": letters_array
    }), 200

def decrement_therapylevel2_count(user_id,expected_Letter):
    db = get_db()
    letter_ref = db.collection('Level') \
    .document(user_id) \
    .collection('Level 2') \
    .document(expected_Letter)

    expected_doc = letter_ref.get()

    if expected_doc.exists:

        expected_data = expected_doc.to_dict()

        current_count = expected_data.get("count", 0)

        print(f"{expected_Letter} count:", current_count)

        #Decrement the count if current_count is not 0
        if current_count > 0:

            letter_ref.update({
            "count": 0
            })

            print(f"{expected_Letter} count decremented")

def increment_therapylevel2_count(user_id,expected_Letter):
    db = get_db()
    letter_ref = db.collection('Level') \
    .document(user_id) \
    .collection('Level 2') \
    .document(expected_Letter)

    letter_ref.update({
    "count": firestore.Increment(1)
    })

    print(f"Count incremented for {expected_Letter}")
    updated_doc = letter_ref.get()

    if updated_doc.exists:

        updated_data = updated_doc.to_dict()
        current_count = updated_data.get("count", 0)
        print(f"{expected_Letter} count:", current_count)

        # Delete document if count reaches 6
        if current_count >= 6:
            letter_ref.delete()

            print(f"{expected_Letter} deleted from Level 2")



@app.route("/predict_therapy_level2",methods=['POST'])
def predict_therapy_level2():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    expected_Letter=request.form.get('expected_Letter')
    file = request.files["file"]
    db = get_db()
    if user_id:
        print("The User(id): ",user_id)
        print("The Question Number: ",question_number)
        print("Expected Letter: ",expected_Letter)
        img = Image.open(file.stream).convert("RGB")
        if expected_Letter == 'q':
            prediction=g_handler_letter(img)
            if prediction not in ['q','Q']:

                decrement_therapylevel2_count(user_id,expected_Letter)
                print("Error q")
            else:
                increment_therapylevel2_count(user_id,expected_Letter)
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
                    decrement_therapylevel2_count(user_id,expected_Letter)
                    
                    print("Error p")
                else:
                    increment_therapylevel2_count(user_id,expected_Letter)
                    print("Correct p")
            elif expected_Letter == 'b':
                if model_predict not in ['b', 'B_caps']:
                    decrement_therapylevel2_count(user_id,expected_Letter)
                    
                    print("Error b")
                else:
                    increment_therapylevel2_count(user_id,expected_Letter)
                    print("Correct b")
            elif expected_Letter == 'd':
                if model_predict not in ['d','D_caps']:
                    decrement_therapylevel2_count(user_id,expected_Letter)
                    
                    print("Error d")
                else:
                    increment_therapylevel2_count(user_id,expected_Letter)
                    print("Correct d")
            else:
                if expected_Letter in ['a','i','n','u','y']:
                    prediction=g_handler_letter(img)
                    if prediction != expected_Letter:
                        decrement_therapylevel2_count(user_id,expected_Letter)
                        print(f"Error {expected_Letter}")
                    else:
                        increment_therapylevel2_count(user_id,expected_Letter)
                        print(f"Correct {expected_Letter}")
                else :
                    if expected_Letter == 'c':
                        if model_predict not in ['c','C_caps']:
                            decrement_therapylevel2_count(user_id,expected_Letter)
                            print("Error c")
                        else:
                            increment_therapylevel2_count(user_id,expected_Letter)
                            print("Correct c")
                    elif expected_Letter == 'l':
                        if model_predict not in ['l','L_caps']:
                            decrement_therapylevel2_count(user_id,expected_Letter)
                            print("Error l")
                        else:
                            increment_therapylevel2_count(user_id,expected_Letter)
                            print("Correct l")
                    elif expected_Letter == 'o':
                        if model_predict not in ['o','O_caps',0]:
                            decrement_therapylevel2_count(user_id,expected_Letter)
                            print("Error o")
                        else:
                            increment_therapylevel2_count(user_id,expected_Letter)
                            print("Correct o")
                    elif expected_Letter == 's':
                        if model_predict not in ['s','S_caps']:
                            decrement_therapylevel2_count(user_id,expected_Letter)
                            print("Error s")
                        else:
                            increment_therapylevel2_count(user_id,expected_Letter)
                            print("Correct s")
                    else:
               

                        if model_predict != expected_Letter:
                            decrement_therapylevel2_count(user_id,expected_Letter)
                            
                            print(f"Error {expected_Letter}")
                        else:
                            increment_therapylevel2_count(user_id,expected_Letter)

                            print(f"Correct {expected_Letter}")



@app.route('/Quiz2Questions', methods=['POST'])
def Quiz2Questions():
    print("Backend reached!")
    user_id = request.form.get('user_id')
    print(f"User ID received: {user_id}")
    if not user_id:
        return jsonify({"error": "User ID missing"}), 400
    
    db = get_db()

    level2_ref = db.collection('Quiz') \
        .document(user_id) \
        .collection('Quiz 2')

    docs = level2_ref.stream()

    letters_array = []

    for doc in docs:
        letters_array.append(doc.id)
    
    print (letters_array)
    
    return jsonify(letters_array)




def delete_quiz2_document(user_id,expected_Letter):
    db = get_db()
    letter_ref = db.collection('Quiz') \
    .document(user_id) \
    .collection('Quiz 2') \
    .document(expected_Letter)
    updated_doc = letter_ref.get()
    print("Document exists:", updated_doc.exists)

    if updated_doc.exists:
        letter_ref.delete()
        print(f"Document deleted from Quiz 2 of letter {expected_Letter}")



#Quiz#2
@app.route("/predict_quiz2",methods=['POST'])
def predict_quiz2():
    user_id=request.form.get('user_id')
    question_number=request.form.get('question_number')
    expected_Letter=request.form.get('expected_Letter')
    file = request.files["file"]
    db = get_db()
    if user_id:
        print("The User(id): ",user_id)
        print("The Question Number: ",question_number)
        print("Expected Letter: ",expected_Letter)
        img = Image.open(file.stream).convert("RGB")
        if expected_Letter == 'q':
            prediction=g_handler_letter(img)
            if prediction not in ['q','Q']:

                print("Error q")
            else:
                delete_quiz2_document(user_id,expected_Letter)
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
                    
                    print("Error p")
                else:
                    delete_quiz2_document(user_id,expected_Letter)
                    print("Correct p")
            elif expected_Letter == 'b':
                if model_predict not in ['b', 'B_caps']:
                    
                    print("Error b")
                else:
                    delete_quiz2_document(user_id,expected_Letter)
                    print("Correct b")
            elif expected_Letter == 'd':
                if model_predict not in ['d','D_caps']:
                    
                    print("Error d")
                else:
                    delete_quiz2_document(user_id,expected_Letter)
                    print("Correct d")
            else:
                if expected_Letter in ['a','i','n','u','y']:
                    prediction=g_handler_letter(img)
                    if prediction != expected_Letter:
                        print(f"Error {expected_Letter}")
                    else:
                        delete_quiz2_document(user_id,expected_Letter)
                        print(f"Correct {expected_Letter}")
                else :
                    if expected_Letter == 'c':
                        if model_predict not in ['c','C_caps']:
                            print("Error c")
                        else:
                            delete_quiz2_document(user_id,expected_Letter)
                            print("Correct c")
                    elif expected_Letter == 'l':
                        if model_predict not in ['l','L_caps']:
                            print("Error l")
                        else:
                            delete_quiz2_document(user_id,expected_Letter)
                            print("Correct l")
                    elif expected_Letter == 'o':
                        if model_predict not in ['o','O_caps',0]:
                            print("Error o")
                        else:
                            delete_quiz2_document(user_id,expected_Letter)
                            print("Correct o")
                    elif expected_Letter == 's':
                        if model_predict not in ['s','S_caps']:
                            print("Error s")
                        else:
                            delete_quiz2_document(user_id,expected_Letter)
                            print("Correct s")
                    else:
               

                        if model_predict != expected_Letter:
                            
                            print(f"Error {expected_Letter}")
                        else:
                            delete_quiz2_document(user_id,expected_Letter)

                            print(f"Correct {expected_Letter}")






#----LEVEL 3-----------------------

_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'DyslexiaDataSet.csv')
def _load_csv_words():
    """
    Loads DyslexiaDataSet.csv and returns a cleaned list of unique lowercase
    alpha-only words (length 2–8).  Called once at module load.
    """
    try:
        df = pd.read_csv(_CSV_PATH, encoding='latin-1')
        col = df.columns[0]         
        raw = df[col].tolist()
        seen = set()
        cleaned = []
        for w in raw:
            if not isinstance(w, str):
                continue
            w = w.strip().lower()
            if w.isalpha() and 2 <= len(w) <= 8 and w not in seen:
                seen.add(w)
                cleaned.append(w)
        print(f"(CSV) Loaded {len(cleaned)} unique words from DyslexiaDataSet.csv")
        return cleaned
    except Exception as e:
        print(f"(CSV ERROR) Could not load DyslexiaDataSet.csv: {e}")
        return []
 
CSV_WORDS = _load_csv_words()
 

def find_similar_words_from_csv(error_word, count=4, required_length=None):
    """
    Returns a deduplicated list of `count` CSV words most similar to error_word.
    The error_word is always element [0] so it is always practised first.
    """
    error_word = str(error_word).strip().lower()
 
    if not CSV_WORDS:
        return [error_word] * count
 
    target_len = required_length if required_length else len(error_word)
    rhyme_key  = error_word[-2:] if len(error_word) >= 2 else error_word[-1:]
 
    p1, p2, p3, p4 = [], [], [], []
    for w in CSV_WORDS:
        if w == error_word:
            continue 
        ends_same  = w.endswith(rhyme_key)
        len_same   = len(w) == target_len
        start_same = w[0] == error_word[0]
 
        if ends_same and len_same:
            p1.append(w)
        elif ends_same:
            p2.append(w)
        elif start_same and len_same:
            p3.append(w)
        elif len_same:
            p4.append(w)
 
    random.shuffle(p1)
    random.shuffle(p2)
    random.shuffle(p3)
    random.shuffle(p4)
 
    result = [error_word]
    for pool in [p1, p2, p3, p4, CSV_WORDS]:
        for w in pool:
            if w not in result:
                result.append(w)
            if len(result) >= count:
                break
        if len(result) >= count:
            break
 
    return result[:count]
 

_CONFUSION_MAP = {
    'b': 'd', 'd': 'b', 'p': 'q', 'q': 'p',
    'n': 'u', 'u': 'n', 'm': 'w', 'w': 'm',
}
 
def build_distractor_options_for_q11(target_word):
    """
    Returns a 3-element list: [target_word, distractor1, distractor2]
    All distractors come from the CSV or are plausible reversals.
    """
    distractors = []
 
    for i, ch in enumerate(target_word):
        if ch in _CONFUSION_MAP:
            variant = target_word[:i] + _CONFUSION_MAP[ch] + target_word[i+1:]
            if variant in CSV_WORDS and variant not in distractors and variant != target_word:
                distractors.append(variant)
 
    rhyme_key = target_word[-2:] if len(target_word) >= 2 else target_word[-1:]
    rhyme_pool = [
        w for w in CSV_WORDS
        if w.endswith(rhyme_key) and w != target_word and w not in distractors
    ]
    random.shuffle(rhyme_pool)
    distractors.extend(rhyme_pool)
 
    if len(distractors) < 2:
        length_pool = [
            w for w in CSV_WORDS
            if len(w) == len(target_word) and w != target_word and w not in distractors
        ]
        random.shuffle(length_pool)
        distractors.extend(length_pool)
 
    options = [target_word] + distractors[:2]
    random.shuffle(options)
    return options
 

@app.route('/get_personalized_question', methods=['GET'])
def get_personalized_question():
    user_id = request.args.get('user_id')
    q_num   = int(request.args.get('question_number', '11'))
 
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
 
    try:
        db = get_db()
        
        user_cartoon = get_user_cartoon_preference(user_id)
 
        level_ref = (
            db.collection('Level')
              .document(user_id)
              .collection('Level_3')
              .document(str(q_num))
        )
        assessment_ref = (
            db.collection('Assessment_Test')
              .document(user_id)
              .collection('Level_3')
              .document(str(q_num))
        )
 
        level_doc  = level_ref.get()
        assess_doc = assessment_ref.get()
 
        errors = []
        source  = "default"
 
        if level_doc.exists:
            lvl_errors = level_doc.to_dict().get("Error", [])
            if lvl_errors:
                errors = lvl_errors
                source = "Level" 
 
        if not errors and assess_doc.exists:
            asmnt_errors = assess_doc.to_dict().get("Error", [])
            if asmnt_errors:
                errors = asmnt_errors
                source = "Assessment_Test"
 
        error_word = errors[0].strip().lower() if errors else "cat"
        print(f"(Q{q_num}) User={user_id} | Source={source} | Error word='{error_word}'")
 
        response_data    = []
        instruction_text = ""
 
        if q_num == 11:
            instruction_text = "Circle the option that matches with the word."
            therapy_words = find_similar_words_from_csv(error_word, count=4)
            for tw in therapy_words:
                options = build_distractor_options_for_q11(tw)
                response_data.append({"target": tw, "options": options})
 
        elif q_num == 12:
            instruction_text = "Read the following words out loud."
            response_data = find_similar_words_from_csv(error_word, count=5)
 
        elif q_num == 13:
            instruction_text = "Circle the words that rhyme the same."
            response_data = find_similar_words_from_csv(error_word, count=12)
 
        elif q_num == 14:
            instruction_text = f'Circle all "{error_word}".'
            distractors = find_similar_words_from_csv(error_word, count=9)[1:]
            grid = [error_word] * 4 + distractors[:8]
            random.shuffle(grid)
            response_data = grid[:12]
 
        elif q_num == 15:
            instruction_text = "Write the word below in the boxes."
            response_data = find_similar_words_from_csv(
                error_word, count=4, required_length=len(error_word)
            )
 
        return jsonify({
            "status":           "success",
            "cartoon_selection": user_cartoon, 
            "question_number":  q_num,
            "instruction_text": instruction_text,
            "target_word":      error_word,
            "data":             response_data,
            "audio_url":        None
        }), 200
 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def process_therapy_submission(user_id, q_num, target_word, is_correct):
    db = get_db()
    level_ref = (
        db.collection('Level')
          .document(user_id)
          .collection('Level_3')
          .document(str(q_num))
    )

    doc = level_ref.get()
    existing = doc.to_dict() if doc.exists else {}
    existing_errors = existing.get("Error", [])

    if not is_correct:
        current_threshold = existing.get("Threshold_Count", 0) + 1

        if target_word not in existing_errors:
            existing_errors.append(target_word)

        level_ref.set({
            'Question Number':  int(q_num),
            'Error':            existing_errors,
            'Threshold_Count':  current_threshold
        }, merge=True)

        print(f"(THERAPY) User={user_id} Q{q_num}: WRONG — '{target_word}' stored in Level. "
              f"Threshold={current_threshold}")
    else:
        if target_word in existing_errors:
            existing_errors.remove(target_word)
        
        if not existing_errors:
            level_ref.delete()
            print(f"(THERAPY) User={user_id} Q{q_num}: CORRECT — All words mastered, doc deleted from Level.")
        else:
            level_ref.update({
                'Error': existing_errors
            })
            print(f"(THERAPY) User={user_id} Q{q_num}: CORRECT — '{target_word}' removed from Level list.")
            

@app.route('/check_answers_therapy', methods=['POST'])
def check_answers_therapy():
    try:
        user_id        = request.form.get('user_id')
        q_num          = request.form.get('question_number')
        target_word    = request.form.get('target_word', 'unknown')
        is_correct_str = request.form.get('is_correct', 'false')
        is_correct     = is_correct_str.lower() == 'true'
 
        if not user_id or not q_num:
            return jsonify({"status": "error", "message": "Missing user_id or question_number"}), 400
 
        process_therapy_submission(user_id, q_num, target_word, is_correct)
        return jsonify({"status": "success"}), 200
 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
 
 

@app.route('/transcribe_and_score_therapy', methods=['POST'])
def transcribe_and_score_therapy():
    if 'audio' not in request.files or 'target_word' not in request.form:
        return jsonify({"error": "Missing audio file or target_word"}), 400
 
    user_id     = request.form.get('user_id')
    target_word = request.form['target_word'].lower().strip()
    q_num       = request.form.get('question_number', '12')
 
    file     = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
 
    try:
        audio  = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=8, language="en")
        os.remove(filepath)
 
        raw_text          = " ".join([s["text"] for s in result["segments"]]).strip().lower()
        transcribed_clean = re.sub(r'[^\w\s]', '', raw_text)
        words_spoken      = transcribed_clean.split()
        word_to_compare   = words_spoken[-1] if words_spoken else ""
 
        # 3. Jaro-Winkler similarity
        similarity  = jellyfish.jaro_winkler_similarity(target_word, word_to_compare)
        is_correct  = bool(similarity >= 0.65)
 
        print(f"(THERAPY Q{q_num}) Target='{target_word}' | Heard='{word_to_compare}' "
              f"| Score={similarity:.2f} | Correct={is_correct}")
 
        # 4. Update Level_Schema
        if user_id:
            process_therapy_submission(user_id, q_num, target_word, is_correct)
 
        return jsonify({
            "status":           "success",
            "target_word":      target_word,
            "transcribed_word": word_to_compare,
            "similarity_score": similarity,
            "is_correct":       is_correct
        }), 200
 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
 

@app.route("/predict_handwriting_batch_therapy", methods=['POST'])
def predict_handwriting_batch_therapy():
    try:
        user_id        = request.form.get('user_id')
        target_word    = request.form.get('target_word', '')
        q_num          = request.form.get('question_number', '15')
        uploaded_files = request.files.getlist("images")
 
        print(f"(THERAPY Q{q_num}) Handwriting: user={user_id} word='{target_word}' "
              f"files={len(uploaded_files)}")
 
        if not user_id or not target_word or not uploaded_files:
            return jsonify({"status": "error", "message": "Missing data"}), 400
 
        detected_errors = []
 
        for i, file in enumerate(uploaded_files):
            if i >= len(target_word):
                break
 
            expected_char = target_word[i]
            img = Image.open(file.stream).convert("RGB")
 
            if expected_char.lower() in ['a', 'i', 'g', 'r']:
                pred_response = g_handler_letter(img)
            else:
                pred_response = letter_predict(img)
 
            model_predict = str(pred_response)
            if isinstance(pred_response, dict) and 'prediction' in pred_response:
                for x in pred_response['prediction']:
                    if isinstance(x, str):
                        model_predict = x
 
            print(f"  [{i}] expected='{expected_char}' predicted='{model_predict}'")
 
            if model_predict.lower() != expected_char.lower():
                is_exception = (
                    (expected_char.lower() == 'o' and model_predict in ('O_caps', '0'))    or
                    (expected_char.lower() == 's' and model_predict in ('S_caps', '5'))    or
                    (expected_char.lower() == 'w' and model_predict == 'W_caps')           or
                    (expected_char        == 'T'  and model_predict == 'T_caps')           or
                    (expected_char        == 'b'  and model_predict == 'B_caps')           or
                    (expected_char        == 'H'  and model_predict == 'H_caps')           or
                    (expected_char        == 'g'  and model_predict == '9')
                )
                if not is_exception:
                    detected_errors.append(expected_char)
 
        is_correct = len(detected_errors) == 0
        process_therapy_submission(user_id, q_num, target_word, is_correct)
 
        return jsonify({
            "status":    "success",
            "is_correct": is_correct,
            "errors":    detected_errors
        }), 200
 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500




import traceback, random
from flask import request, jsonify
from config.firebase import get_db
from firebase_admin import firestore



@app.route('/get_personalized_question_quiz3', methods=['GET'])
def get_personalized_question_quiz3():
    user_id = request.args.get('user_id')
    q_num   = str(request.args.get('question_number', '1')).strip() 

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        db = get_db()
        error_words = []
        
        user_cartoon = get_user_cartoon_preference(user_id)

        try:
            docs = (db.collection('Level')
                      .document(user_id)
                      .collection('Level_3')
                      .stream())
            for doc in docs:
                for w in (doc.to_dict() or {}).get('Error', []):
                    w = str(w).strip().lower()
                    if w and 3 <= len(w) <= 5 and w not in error_words:
                        error_words.append(w)
        except Exception as e:
            print(f"Level_Schema read error: {e}")

        try:
            docs = (db.collection('Assessment_Test')
                      .document(user_id)
                      .collection('Level_3')
                      .stream())
            for doc in docs:
                data = doc.to_dict() or {}
                if data.get('Mastered'):
                    continue
                for w in data.get('Error', []):
                    w = str(w).strip().lower()
                    if w and 3 <= len(w) <= 5 and w not in error_words:
                        error_words.append(w)
        except Exception as e:
            print(f"Assessment_Test read error: {e}")

        if not error_words:
            pool = [w for w in CSV_WORDS if 3 <= len(w) <= 5]
            random.shuffle(pool)
            error_words = pool[:10]

        random.shuffle(error_words)
        target_word = error_words[0] if error_words else "cat"

        print(f"(QUIZ3 Q{q_num}) User={user_id} | Target='{target_word}'")

        if q_num == "1":
            options = build_distractor_options_for_q11(target_word)
            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon, 
                "audio_url":   None,
                "target_word": target_word,
                "data": [{"target": target_word, "options": options}]
            }), 200

        elif q_num == "2":
            read_words = find_similar_words_from_csv(target_word, count=4)
            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon, 
                "audio_url":   None,
                "target_word": target_word,
                "data":        read_words 
            }), 200

        elif q_num == "3":
            rhyme_words = find_similar_words_from_csv(target_word, count=12)
            random.shuffle(rhyme_words)
            return jsonify({
                "status": "success",
                "cartoon_selection": user_cartoon, 
                "audio_url":   None,
                "target_word": target_word,
                "data":        rhyme_words 
            }), 200

        else:
            return jsonify({"error": f"Invalid question_number '{q_num}'. Use 1, 2, or 3."}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/submit_quiz_answer', methods=['POST'])
def submit_quiz_answer():
    user_id    = request.form.get('user_id')
    target     = request.form.get('target_word', '').strip().lower()
    q_num_str  = request.form.get('question_number', '0')
    is_correct = request.form.get('is_correct', 'false').lower() == 'true'

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        db = get_db()
        quiz_col_ref = (db.collection('Quiz')
                          .document(user_id)
                          .collection('Quiz 3'))

        quiz_col_ref.add({
            "word":            target,
            "question_number": int(q_num_str) if q_num_str.isdigit() else 0,
            "is_correct":      is_correct,
            "timestamp":       firestore.SERVER_TIMESTAMP
        })

        
        all_docs = list(quiz_col_ref.stream())

      
        from collections import defaultdict
        latest_per_q = {}  

        for doc in all_docs:
            data = doc.to_dict() or {}
            qn   = data.get('question_number', 0)
            if qn not in (1, 2, 3):        
                continue
            ts = data.get('timestamp')      

            if qn not in latest_per_q:
                latest_per_q[qn] = (ts, data.get('is_correct', False))
            else:
                prev_ts, _ = latest_per_q[qn]
                if ts is not None and (prev_ts is None or ts > prev_ts):
                    latest_per_q[qn] = (ts, data.get('is_correct', False))

        total_questions  = 3   # Q1, Q2, Q3
        correct_total    = sum(1 for (_, correct) in latest_per_q.values() if correct)
        reaches_75_percent = correct_total >= 2

        # 3. Update score summary
        quiz_col_ref.document('score_summary').set({
            "total_correct":   correct_total,
            "total_questions": total_questions,
            "percentage":      round((correct_total / total_questions) * 100, 1),
            "passed_75":       reaches_75_percent,
            "last_updated":    firestore.SERVER_TIMESTAMP
        }, merge=True)

        # 4. Mark completion in Assessment_Test so Level 4 unlocks
        if reaches_75_percent:
            (db.collection('Assessment_Test')
               .document(user_id)
               .collection('Quiz_3')
               .document('completion')
               .set({
                   'completed':  True,
                   'score':      f"{correct_total}/{total_questions}",
                   'flag_75':    True,
                   'timestamp':  firestore.SERVER_TIMESTAMP
               }, merge=True))

        return jsonify({
            "status":        "success",
            "total_correct": correct_total,
            "quiz3_passed":  reaches_75_percent
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


import traceback
from flask import jsonify

@app.route("/api/scores/<user_id>", methods=['GET'])
def get_user_score(user_id):
    try:
        db = get_db()

        user_profile_doc = db.collection('users').document(user_id).get()
        if not user_profile_doc.exists:
            return jsonify({"status": "error", "message": "User profile not found."}), 404

        user_data = user_profile_doc.to_dict() or {}
        has_completed = user_data.get('hasCompletedAssessment', None)
        if has_completed is None:
            has_completed = user_data.get('hasCompleteAssesment', True)

        if not has_completed:
            return jsonify({
                "status": "success", 
                "hasCompletedAssessment": False,
                "assessment_unlocked_index": 0,
                "Level2_Empty": True,
                "data": []
            }), 200

        assessment_ref = db.collection('Assessment_Test').document(user_id)
        level_root_ref = db.collection('Level').document(user_id)
        quiz_root_ref  = db.collection('Quiz').document(user_id)
        
        scores_summary = []
        assessment_scores = {}
        level2_is_empty = True

        level_totals = {'Level_1': 5, 'Level_2': 5, 'Level_3': 5, 'Level_4': 4}
        
        for level_name, total_questions in level_totals.items():
            docs = list(assessment_ref.collection(level_name).stream())
            errors_made = len(docs)
            
            if level_name == 'Level_2' and errors_made > 0:
                level2_is_empty = False

            correct = max(0, total_questions - errors_made)
            assessment_scores[level_name] = correct
            scores_summary.append({
                "level": level_name,
                "score": f"{correct}/{total_questions}"
            })

        current_max_index = 0

        # Pass Level 1 (>=3) -> Instantly unlock Quiz 1 (Index 1)
        if assessment_scores.get('Level_1', 0) >= 3:
            current_max_index = 1
            
            # Pass Level 2 (>=3) -> Instantly unlock Quiz 2 (Index 3)
            if assessment_scores.get('Level_2', 0) >= 3:
                current_max_index = 3
                
                # Pass Level 3 (>=3) -> Instantly unlock Quiz 3 (Index 5)
                if assessment_scores.get('Level_3', 0) >= 3:
                    current_max_index = 5

        # ── 4. DYNAMIC STAGE STATE EVALUATION ──
        
        # Stage 1: Level 1 Mastery (Therapy fallback)
        lvl1_graduated = False
        l1_meta = level_root_ref.collection('Level_1').document('meta_status').get()
        if l1_meta.exists:
            m_data = l1_meta.to_dict() or {}
            lvl1_graduated = m_data.get('graduated', False) or m_data.get('unlocked_level_2', False)

        # Stage 2: Quiz 1 Passed
        quiz1_passed = False
        q1_doc = quiz_root_ref.collection('Quiz 1').document('Score').get()
        if q1_doc.exists:
            q1_data = q1_doc.to_dict() or {}
            quiz1_passed = q1_data.get('passed', False) or (q1_data.get('percentage', 0) >= 75.0) or (q1_data.get('score', 0) >= 3)

        # Stage 3: Level 2 Mastery (Checks both spaced and underscored collections)
        lvl2_graduated = False
        l2_meta = level_root_ref.collection('Level_2').document('meta_status').get()
        if not l2_meta.exists:
            l2_meta = level_root_ref.collection('Level 2').document('meta_status').get()
            
        if l2_meta.exists and l2_meta.to_dict().get('graduated', False):
            lvl2_graduated = True
        else:
            # Dynamic fallback: If Quiz 1 passed, verify if active error targets have been depleted
            l2_docs_spaced = list(level_root_ref.collection('Level 2').stream())
            l2_docs_under  = list(level_root_ref.collection('Level_2').stream())
            active_l2_targets = [d for d in (l2_docs_spaced + l2_docs_under) if d.id != 'meta_status']
            
            if quiz1_passed and len(active_l2_targets) == 0:
                lvl2_graduated = True

        # Stage 4: Quiz 2 Passed
        quiz2_passed = False
        q2_summary = quiz_root_ref.collection('Quiz 2').document('score_summary').get()
        if q2_summary.exists:
            q2_data = q2_summary.to_dict() or {}
            quiz2_passed = q2_data.get('passed', False) or (q2_data.get('score', 0) >= 3)
        elif lvl2_graduated:
            # Verification endpoints delete active target documents inside Quiz 2 upon completion
            q2_docs = list(quiz_root_ref.collection('Quiz 2').stream())
            active_q2_targets = [d for d in q2_docs if d.id != 'score_summary']
            if len(active_q2_targets) == 0:
                quiz2_passed = True
            
        scores_summary.append({
            "level": "Quiz_2",
            "score": f"{'1' if quiz2_passed else '0'}/1"
        })

        # Stage 5: Level 3 Mastery
        lvl3_graduated = False
        l3_meta = level_root_ref.collection('Level_3').document('meta_status').get()
        if l3_meta.exists and l3_meta.to_dict().get('graduated', False):
            lvl3_graduated = True
        else:
            # Dynamic fallback: Level 3 documents get deleted upon completion
            l3_docs = list(level_root_ref.collection('Level_3').stream())
            active_l3_targets = [d for d in l3_docs if d.id != 'meta_status']
            if quiz2_passed and len(active_l3_targets) == 0:
                lvl3_graduated = True

        # Stage 6: Quiz 3 Passed
        quiz3_passed = False
        q3_doc = quiz_root_ref.collection('Quiz 3').document('score_summary').get()
        if q3_doc.exists:
            q3_data = q3_doc.to_dict() or {}
            quiz3_passed = q3_data.get('passed_75', False) or (q3_data.get('total_correct', 0) >= 2)
            
        scores_summary.append({
            "level": "Quiz_3",
            "score": f"{'1' if quiz3_passed else '0'}/1",
            "score_summary": {"passed_75": quiz3_passed}
        })

        # ── 5. LAYER DYNAMIC GAMEPLAY OVERRIDES ──
        
        # Apply Level 1 therapy graduation fallback if assessment failed
        if lvl1_graduated and current_max_index < 1:
            current_max_index = 1  

        # Passing Quiz 1 unlocks Level 2
        if quiz1_passed and current_max_index < 2:
            current_max_index = 2  
            
        # Completing Level 2 therapy folder unlocks Quiz 2
        if lvl2_graduated and current_max_index < 3:
            current_max_index = 3  

        # Passing Quiz 2 independently unlocks Level 3
        if quiz2_passed and current_max_index < 4:
            current_max_index = 4  
            
        # Completing Level 3 therapy folder unlocks Quiz 3
        if lvl3_graduated and current_max_index < 5:
            current_max_index = 5  

        # Passing Quiz 3 independently unlocks Level 4
        if quiz3_passed and current_max_index < 6:
            current_max_index = 6

        return jsonify({
            "status": "success",
            "hasCompletedAssessment": True,
            "assessment_unlocked_index": current_max_index,
            "Level2_Empty": level2_is_empty,
            "data": scores_summary
        }), 200

    except Exception as e:
        print(f"Scores Gating API Error:\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": "Internal evaluation engine faulted"}), 500
@app.route('/api/user_progress/<user_id>', methods=['GET'])
def get_user_progress(user_id):
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        db = get_db()
        
        TOTAL_LEVELS = 4.0
        TOTAL_QUIZZES = 3.0
        
        levels_attempted = 0
        quizzes_attempted = 0

        # Level 1 check
        l1_doc = db.collection('Level').document(user_id).collection('Level_1').document('meta_status').get()
        if l1_doc.exists or len(list(db.collection('Level').document(user_id).collection('Level_1').limit(1).stream())) > 0:
            levels_attempted += 1

        # Level 2 check
        l2_docs = list(db.collection('Level').document(user_id).collection('Level 2').limit(1).stream())
        if len(l2_docs) > 0:
            levels_attempted += 1

        # Level 3 check
        l3_docs = list(db.collection('Level').document(user_id).collection('Level_3').limit(1).stream())
        if len(l3_docs) > 0:
            levels_attempted += 1

        # Level 4 check
        l4_doc = db.collection('Level').document(user_id).collection('Level_4').document('meta_status').get()
        if l4_doc.exists or len(list(db.collection('Level').document(user_id).collection('Level_4').limit(1).stream())) > 0:
            levels_attempted += 1

        # Quiz 1 check
        q1_doc = db.collection('Quiz').document(user_id).collection('Quiz 1').document('Score').get()
        if q1_doc.exists:
            quizzes_attempted += 1

        # Quiz 2 check
        q2_docs = list(db.collection('Quiz').document(user_id).collection('Quiz 2').limit(1).stream())
        if len(q2_docs) > 0:
            quizzes_attempted += 1

        # Quiz 3 check
        q3_docs = list(db.collection('Quiz').document(user_id).collection('Quiz 3').limit(1).stream())
        if len(q3_docs) > 0:
            quizzes_attempted += 1

        level_progress_float = levels_attempted / TOTAL_LEVELS
        quiz_progress_float = quizzes_attempted / TOTAL_QUIZZES
        
        overall_percentage = int(((level_progress_float * 0.6) + (quiz_progress_float * 0.4)) * 100)

        return jsonify({
            "status": "success",
            "overall_progress_percentage": f"{overall_percentage}%",
            "levels": {
                "attempted": levels_attempted,
                "total": int(TOTAL_LEVELS),
                "progress_float": round(level_progress_float, 2),
                "progress_text": f"{int(level_progress_float * 100)}%"
            },
            "quizzes": {
                "attempted": quizzes_attempted,
                "total": int(TOTAL_QUIZZES),
                "progress_float": round(quiz_progress_float, 2),
                "progress_text": f"{int(quiz_progress_float * 100)}%"
            }
        }), 200

    except Exception as e:
        print(f"(CRITICAL) Progress API faulted:\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error calculating progress"}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001,threaded=True)

