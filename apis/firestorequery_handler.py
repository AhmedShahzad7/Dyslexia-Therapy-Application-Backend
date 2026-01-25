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
    elif (question_number == "4"):
        # 1. Ignore Success Messages
        # If it is "Completed" or "Correct Match...", we return True (no error to log)
        if direction_predicted == "Completed" or "Correct Match" in direction_predicted:
            return True

        # 2. Extract Arrow Name from Error String
        # Input format: "Error: Arrow(Up) matched with Word(Down)"
        if "Arrow(" in direction_predicted:
            # Extract the word between "Arrow(" and ")"
            start = direction_predicted.find("Arrow(") + 6
            end = direction_predicted.find(")", start)
            
            if start > 5 and end > start:
                extracted_direction = direction_predicted[start:end] # Becomes "Up", "Down", "Left", "Right"
                direction_key = extracted_direction.capitalize() # Ensure format "Up"
        
        # 3. Map to Database Path
        if direction_key in ["Up", "Down", "Left", "Right"]:
             field_path = f"error.Direction.{direction_key}"
        else:
            # If we couldn't parse a valid direction, do nothing
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



