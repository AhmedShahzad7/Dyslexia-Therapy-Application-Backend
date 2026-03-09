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

#IMPORTING FUNCTIONS FROM FOLDER
from config.firebase import initialize_firebase
from apis.model_api import letter_predict
from apis.model_api import direction_predict
from apis.model_api import predict_handwriting
from apis.letter_handler import g_handler_letter
from apis.firestorequery_handler import store_direction_error
from apis.firestorequery_handler import store_mcq_error
from apis.firestorequery_handler import store_cartoon_selection
from apis.firestorequery_handler import store_voice_error



app = Flask(__name__)
#WHISPER X
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_audio') # <-- Now it's an absolute path!
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
        
        # 4. Set your threshold! 0.80 is usually the sweet spot for minor AI misinterpretations.
        is_correct = bool(similarity_score >= 0.65)
        
        # Optional: Generate the raw phonetic codes just so you can see them in your logs!
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
            if model_predict_3 not in ['s','S_caps',5]:
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
            if model_predict_4 not in ['z','Z_caps']:
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
                pred_response = g_handler_letter(img) 
                model_predict = pred_response
                if 'prediction' in pred_response:
                    for x in pred_response['prediction']:
                        if isinstance(x, str):
                            model_predict = x
                
                print(f"Letter {i}: Expected '{expected_char}' vs Predicted '{model_predict}'")

                # --- COMPARISON LOGIC ---
                # Check if the model's prediction matches the expected letter.
                # We use .lower() to allow 'F' == 'f'
                if model_predict.lower() != expected_char.lower():
                    # Record the error
                    error_msg =expected_char
                    detected_errors.append(error_msg)

            # 4. Update Database based on results
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

# CARTOON SELECTION
@app.route("/select_cartoon", methods=["POST"])
def select_cartoon():
    user_id = request.form.get('user_id')
    cartoon_name = request.form.get('cartoon_name')
    
    if user_id and cartoon_name:
        print(f"\n(DEBUG) Cartoon Selection Received: User={user_id}, Cartoon={cartoon_name}\n")
        
        success = store_cartoon_selection(user_id, cartoon_name)
        
        if success:
            return "Success"
        else:
            return "Database Error"
            
    return "Error: Missing data"


#level 4 question 18 
@app.route("/check_answers_q18", methods=['POST'])
def check_answers_q18():
    user_id = request.form.get('user_id')
    question_number = request.form.get('question_number')
    answers_list_raw = request.form.get('answers_list') 
    
    if not user_id or not answers_list_raw:
        return "Missing Data", 400

    try:
        # 1. Parse the JSON string from the frontend
        selected_words = json.loads(answers_list_raw)
        
        # 2. PRINT full selection to VS Code Terminal
        print(f"\n--- Question {question_number} Selection ---")
        print(f"User ID: {user_id}")
        print(f"Full Selection: {selected_words}")
        print("------------------------------------------\n")
        
        # 3. Filter: Identify only words that are NOT "bog"
        detected_errors = [word for word in selected_words if word.lower() != "bog"]
        
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
                    .document(user_id) \
                    .collection('Level_4') \
                    .document(str(question_number))

        # 4. Prepare Firebase Payload (No 'User Selection' field)
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
    # For Level 4, we receive a full sentence (e.g., "The big dog")
    target_sentence = request.form.get('target_sentence')     
    question_number = request.form.get('question_number') 
    uploaded_files = request.files.getlist("images")

    print(f"\n(DEBUG) Sentence Prediction for User: {user_id}, Sentence: {target_sentence}, Files: {len(uploaded_files)}\n")

    if user_id and target_sentence and uploaded_files:
        try:
            db = get_db()
            # Updated collection path to 'Level_4' as per Question 17 requirements
            doc_ref = db.collection('Assessment_Test') \
                        .document(user_id) \
                        .collection('Level_4') \
                        .document(str(question_number))
            
        
            expected_chars = [char for char in target_sentence if char.isalnum()]
            
            detected_errors = []

            for i, file in enumerate(uploaded_files):
                # Guard against more files than expected characters
                if i >= len(expected_chars): 
                    break

                expected_char = expected_chars[i]
                
                # Process image for the AI model
                img = Image.open(file.stream).convert("RGB")
                pred_response = g_handler_letter(img)
                model_predict=pred_response 
                
                model_predict = ""
                # Extract the character string from the model's response dictionary
                if isinstance(pred_response, dict) and 'prediction' in pred_response:
                    for x in pred_response['prediction']:
                        if isinstance(x, str):
                            model_predict = x
                else:
                    model_predict = str(pred_response)
                
                print(f"Index {i}: Expected '{expected_char}' vs Predicted '{model_predict}'")

                # --- COMPARISON LOGIC ---
                # Match character by character (case-insensitive)
                if model_predict.lower() != expected_char.lower():
                    detected_errors.append(expected_char)

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


if __name__ == "__main__":
    app.run(host='192.168.1.11', port=5000,threaded=True)

