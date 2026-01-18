#DO NOT TOUCH THIS!

import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    # Check if already initialized
    if not firebase_admin._apps:
        # POINT THIS TO THE NEW JSON YOU JUST DOWNLOADED
        cred = credentials.Certificate("config/dyslexiaid-78b8d-firebase-adminsdk-fbsvc-1fd0c51287.json")
        
        # For Firestore, you don't need the databaseURL!
        firebase_admin.initialize_app(cred)

def get_db():
    """Returns the Firestore client"""
    return firestore.client()