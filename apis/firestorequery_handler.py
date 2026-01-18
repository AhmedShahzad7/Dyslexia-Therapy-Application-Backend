from config.firebase import get_db
from firebase_admin import firestore

def store_direction_error(user_id, direction_predicted,question_number):
    db = get_db()
    
    # 1. Reference the user's document
    doc_ref = db.collection('Error_Classification').document(user_id)
    
    # Capitalize to match DB format (e.g., "Up", "Down")
    direction_key = direction_predicted.capitalize() 
    field_path = None
    if (question_number == "1") and (direction_predicted != "Up"):
        field_path = "error.Direction.Up"
    elif (question_number == "2") and (direction_predicted != "Left"):
        field_path = "error.Direction.Left"

    elif (question_number == "3") and (direction_predicted != "Down"):
        field_path = "error.Direction.Down" 
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
        else:
            initial_data["error"]["Direction"][direction_key] = True
            
        # Save it using .set() which creates the document
        doc_ref.set(initial_data)
        
    return True