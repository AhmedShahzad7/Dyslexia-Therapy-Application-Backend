from config.firebase import get_db
from firebase_admin import firestore



def store_direction_error(user_id, direction_predicted, question_number):
    db = get_db()
    detected_errors = []
    
    error_key = None
    q_str = str(question_number) 
    
    if q_str == "1" and direction_predicted != "Up":
        error_key = "Up"
    elif q_str == "2" and direction_predicted != "Left":
        error_key = "Left"
    elif q_str == "3" and direction_predicted != "Down":
        error_key = "Down"
    elif q_str == "5" and direction_predicted != "Completed":
        error_key = "Left"
        

    elif q_str == "4":
        if "Completed" in direction_predicted or "Correct Match" in direction_predicted:
            return True # 
            
        
        if "Arrow(" in direction_predicted:
            start = direction_predicted.find("Arrow(") + 6
            end = direction_predicted.find(")", start)
            if start > 5 and end > start:
                error_key = direction_predicted[start:end].capitalize()
        
  
        elif direction_predicted.capitalize() in ["Up", "Down", "Left", "Right"]:
            error_key = direction_predicted.capitalize()

    if not error_key:
        return True

    doc_ref = db.collection('Assessment_Test') \
                .document(user_id) \
                .collection('Level_1') \
                .document(q_str)
    detected_errors.append(error_key)
    payload = {
        'Answer': 'Incorrect',
        "Error": detected_errors,
        'Question Number': int(question_number),
    }

    doc_ref.set(payload, merge=True)
    print(f"Firestore updated for User {user_id}: error.Direction.{error_key} = True")
    
    return True




#ALEVEL 3 Q11 Q13
def store_mcq_error(user_id,answer_list,question_number):
    db=firestore.client()
    doc_ref = db.collection('Assessment_Test') \
                .document(user_id) \
                .collection('Level_3') \
                .document(str(question_number))
    detected_errors = []
    
    #QUESTION 11
    if question_number=="11":
        first_answer = answer_list[0]
        second_answer=answer_list[1]
        print(first_answer,second_answer)
        if first_answer!="ben":
            detected_errors.append("ben")
        if second_answer!="pen":
            detected_errors.append("pen")    
        if detected_errors:
            print(f"Firestore updated for User {user_id} with errors: {detected_errors}")
            doc_ref.set({
            'Question Number': int(question_number),
            'Answer': 'Incorrect',
            }, merge=True)
            doc_ref.update({
            'Error': firestore.ArrayUnion(detected_errors)
            
            })
            #QUESTION 13
    elif question_number=="13":
        correct_set = {"cap", "lap", "map", "nap", "tap"}
        user_set=set(answer_list)
  
        missing_words = correct_set - user_set
        for word in missing_words:
            detected_errors.append(word)

        wrong_selections = user_set - correct_set
        for word in wrong_selections:
            detected_errors.append(word)

        if detected_errors:
            print(f"Firestore updated for User {user_id} with errors: {detected_errors}")
            
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Incorrect',
            }, merge=True)

            doc_ref.update({
                'Error': firestore.ArrayUnion(detected_errors)
            })
            #QUESTION 14
    elif question_number=="14":
        target_word = "was"
        count = 4
        for word in answer_list:
            if word != target_word:
                if word == "saw":
                    # "saw" is a specific reversal error common in dyslexia THIS WILL BE HANDLED LATER
                    detected_errors.append(word)
                else:
                    detected_errors.append(word)
        found_count = answer_list.count(target_word)
        if found_count < count:
            missing_count = count - found_count #MISSED WILL BE HANDLED LATERR
            detected_errors.append(word)
        if detected_errors:
            print(f"Firestore updated for User {user_id} with errors: {detected_errors}")
            
            doc_ref.set({
                'Question Number': int(question_number),
                'Answer': 'Incorrect',
            }, merge=True)

            doc_ref.update({
                'Error': firestore.ArrayUnion(detected_errors)
            })
                
    
    
def store_cartoon_selection(user_id, cartoon_name):
    db = get_db()
    
    # Reference the 'cartoon_selection' collection
    doc_ref = db.collection('cartoon_selection').document(user_id)
    
    try:
        user_ref = db.collection('users').document(user_id)
        user_ref.set({
            "hasCompletedAssessment": True,
        }, merge=True)
        # Save the selection 
        doc_ref.set({
            "cartoon": cartoon_name.lower() 
        })
        print(f"Cartoon '{cartoon_name}' saved for user {user_id}")
        return True
    except Exception as e:
        print(f"Error saving cartoon: {e}")
        return False
    


#ALEVEL 3 Q12
def store_voice_error(user_id,targetname,error,question_number,detected_errors):
        print(f"Firestore updated for User {user_id} with errors: {detected_errors}")
        db = get_db()
        doc_ref = db.collection('Assessment_Test') \
                    .document(user_id) \
                    .collection('Level_3') \
                    .document(str(question_number))
        
        # 1. Ensure the document exists and is marked as Incorrect
        doc_ref.set({
            'Question Number': int(question_number),
            'Answer': 'Incorrect',
        }, merge=True)

        # 2. Push the specific error into the Error array
        doc_ref.update({
            'Error': firestore.ArrayUnion(detected_errors)
        })


#question19
def store_voice_error1(user_id, targetname, error, question_number, detected_errors):
    try:
        db = get_db()
        level = "Level_4" if int(question_number) >= 17 else "Level_3"
        
        doc_ref = db.collection('Assessment_Test').document(user_id).collection(level).document(str(question_number))
        
        # 1. Mark the question status
        doc_ref.set({
            'Question Number': int(question_number),
            'Answer': 'Incorrect',
        }, merge=True)

        doc_ref.update({
            'Error': firestore.ArrayUnion(detected_errors)
        })
        
        print(f"(SUCCESS) Firestore updated for User {user_id} with errors: {detected_errors}")
        return True
    except Exception as e:
        print(f"Firestore update failed: {e}")
        return False
