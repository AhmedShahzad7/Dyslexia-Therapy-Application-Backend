from config.firebase import get_db
from google.cloud.firestore_v1 import Increment
from firebase_admin import firestore



def store_direction_error(user_id, direction_predicted, question_number):
    db = get_db()
    detected_errors = []
    
    error_key = None
    q_str = str(question_number) 
    
    # --- UPDATED DYNAMIC LOGIC FOR QUESTION 1 ---
    if q_str == "1":
        # In app.py, we evaluate correct/incorrect before calling this function.
        # We pass the 'target_word' directly into the 'direction_predicted' parameter.
        # So we just store whatever word they failed!
        error_key = direction_predicted.capitalize()

    # --- OLD STATIC LOGIC FOR OTHER QUESTIONS ---
    elif q_str == "2" and direction_predicted != "Left":
        error_key = "Left"
    elif q_str == "3" and direction_predicted != "Down":
        error_key = "Down"
    elif q_str == "5" and direction_predicted != "Completed":
        error_key = "Left"
        
    elif q_str == "4":
        if "Completed" in direction_predicted or "Correct Match" in direction_predicted:
            return True 
            
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
    print(f"Firestore updated for User {user_id}: error.Direction.{error_key} = True in Document {q_str}")
    
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


#========================================= PERSONALIZED Levels================================

#LEVEL 1 

def update_therapy_progress(user_id, target_word, is_correct, question_number, level_name="Level_1"):
    db = get_db()
    q_str = str(question_number)
    MASTERY_THRESHOLD = 3

    # Point directly to your main collection: Level -> userid -> Level_1 -> q_str
    doc_ref = db.collection('Level') \
                .document(user_id) \
                .collection(level_name) \
                .document(q_str)

    doc = doc_ref.get()
    
    if not doc.exists:
        # Failsafe initialization if something bypassed the GET route
        doc_ref.set({
            'Question Number': int(question_number),
            'Error': [target_word.capitalize()],
            'success_count': 0,
        })
        current_success = 0
    else:
        current_success = doc.to_dict().get('success_count', 0)

    if is_correct:
        new_success_count = current_success + 1
        
        if new_success_count >= MASTERY_THRESHOLD:
            # THE GRADUATION TRIGGER: Purge the document entirely
            doc_ref.delete()
            print(f"(SUCCESS) Mastery reached! Document Q{q_str} completely removed from 'Level' collection.")
            return True, "mastered"
        else:
            # Safely increment the counter
            doc_ref.update({'success_count': Increment(1)})
            print(f"(PROGRESS) Correct answer. Success count now {new_success_count}/5.")
            return True, "progressing"
    else:
        # Penalty: reset consecutive counter back to 0
        doc_ref.update({
            'success_count': 0,
        })
        print(f"(PENALTY) Incorrect answer for '{target_word}'. Counter reset to 0.")
        return False, "reset"






# --- LEVEL 4 PROGRESS UPDATER (PARALLEL DICTIONARY SAFE) ---
def update_therapy_progress_l4(user_id, target_word, is_correct, question_number):
    try:
        db = get_db()
        q_str = str(question_number)
        MASTERY_THRESHOLD = 5
        
        doc_ref = db.collection('Level') \
                    .document(user_id) \
                    .collection('Level_4') \
                    .document(q_str)
                    
        doc = doc_ref.get()
        clean_target = str(target_word).strip().lower()
        
        # 1. Failsafe Initialization
        if not doc.exists:
            initial_array = [clean_target]
            initial_map = {clean_target: 1 if is_correct else 0}
            doc_ref.set({
                'Question Number': int(question_number),
                'Error': initial_array,
                'scores_tracker': initial_map,
                'success_count': 0
            })
            print(f"(L4 INIT) Seeded Q{q_str} with parallel tracking for '{clean_target}'.")
            return True, "progressing" if is_correct else "reset"
            
        data = doc.to_dict()
        raw_errors = data.get('Error', [])
        current_errors = [str(w).strip().lower() for w in (raw_errors if isinstance(raw_errors, list) else [raw_errors]) if str(w).strip()]
        
        scores_map = data.get('scores_tracker', {})
        if not isinstance(scores_map, dict):
            scores_map = {w: 0 for w in current_errors}
            
        if clean_target not in scores_map:
            scores_map[clean_target] = 0
            if clean_target not in current_errors:
                current_errors.append(clean_target)
                
        # 2. Process Outcomes Independently
        if is_correct:
            scores_map[clean_target] += 1
            current_score = scores_map[clean_target]
            
            if current_score >= MASTERY_THRESHOLD:
                updated_errors = [w for w in current_errors if w != clean_target]
                scores_map.pop(clean_target, None)
                
                if not updated_errors:
                    doc_ref.delete()
                    print(f"(L4 SUCCESS) Entire array mastered! Document Q{q_str} completely removed.")
                    return True, "mastered"
                else:
                    doc_ref.update({
                        'Error': updated_errors,
                        'scores_tracker': scores_map
                    })
                    print(f"(L4 PROGRESS) Word '{clean_target}' mastered! Shifted out. Remaining: {updated_errors}")
                    return True, "progressing"
            else:
                doc_ref.update({'scores_tracker': scores_map})
                print(f"(L4 PROGRESS) Correct match. '{clean_target}' count now {current_score}/{MASTERY_THRESHOLD}.")
                return True, "progressing"
        else:
            scores_map[clean_target] = 0
            doc_ref.update({'scores_tracker': scores_map})
            print(f"(L4 PENALTY) Incorrect match. '{clean_target}' counter reset to 0.")
            return False, "reset"
            
    except Exception as e:
        print(f"(CRITICAL ERROR) update_therapy_progress_l4 failed: {str(e)}")
        return False, "error"
 