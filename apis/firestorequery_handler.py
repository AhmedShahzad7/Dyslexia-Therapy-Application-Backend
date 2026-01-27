from config.firebase import get_db
from firebase_admin import firestore

def store_direction_error(user_id, direction_predicted,question_number):
    db = get_db()
    
    # 1. Reference the user's document
    doc_ref = db.collection('Assessment_Test').document(user_id).collection('Level_1').document(str(question_number))
    
    # Capitalize to match DB format (e.g., "Up", "Down")
    direction_key = direction_predicted.capitalize() 
    field_path = None
    if (question_number == "1") and (direction_predicted != "Up"):
        field_path = "error.Direction.Up"
    elif (question_number == "2") and (direction_predicted != "Left"):
        field_path = "error.Direction.Left"

    elif (question_number == "3") and (direction_predicted != "Down"):
        field_path = "error.Direction.Down" 
    elif (question_number == "4"):
        # 1. Ignore Success Messages
        # If it is "Completed" or "Correct Match...", we return True (no error to log)
        if direction_predicted == "Completed" or "Correct Match" in direction_predicted:
            return True
    elif (question_number == "5") and (direction_key != "Completed"):
        field_path = "error.Direction.Left"
    
    if not field_path:
        return True
    try:
        # ATTEMPT 1: Try to just update the specific field
        doc_ref.update({
            field_path: True
        })
        print(f"Updated existing document for user {user_id}")
        return True
    
    #IF NEW USER WITH NO ERROR CLASSIFICATION  
    except Exception as e:
        # ATTEMPT 2: If Update failed (likely 404), CREATE the document
        print(f"Document not found ({e}). Creating new document for user {user_id}...")
        
        # Define the full default structure
        # This is better than just setting one field because it keeps your DB clean!
        initial_data = {
            "error": {
                "Direction": {
                    "Up": False,
                    "Down": False,
                    "Left": False,
                    "Right": False
                }
            }
        }
        
        if(question_number==1):    
            # Set the one predicted direction to True
            initial_data["error"]["Direction"]["Up"] = True
            #PART 2
        elif(question_number==2) and direction_predicted!="Left":
            field_path = f"error.Direction.Left"
        elif(question_number==3) and direction_predicted!="Down":
            field_path = f"error.Direction.Down" 
        elif (question_number == "4"):
            # 1. Check if it is a success message (Don't log error)
            if direction_predicted == "Completed" or "Correct Match" in direction_predicted:
                pass 
            
            # 2. Extract and Log Error
            else:
                target_key = None
                
                # Check for "Error: Arrow(X)..." format
                if "Arrow(" in direction_predicted:
                    start = direction_predicted.find("Arrow(") + 6
                    end = direction_predicted.find(")", start)
                    if start > 5 and end > start:
                        target_key = direction_predicted[start:end].capitalize()
                
                # Fallback: Check if the simple string is in the keys (e.g. just "Up")
                elif direction_key in ["Up", "Down", "Left", "Right"]:
                    target_key = direction_key

                # Update the initial data if we found a valid key
                if target_key and target_key in initial_data["error"]["Direction"]:
                    initial_data["error"]["Direction"][target_key] = True
        elif (question_number == "5") and (direction_predicted != "Completed"):
            initial_data["error"]["Direction"]["Left"] = True
        else:
            initial_data["error"]["Direction"][direction_key] = True
            
        # Save it using .set() which creates the document
        doc_ref.set(initial_data)
        
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
                
    
    
    
    


