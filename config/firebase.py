

import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():

    if not firebase_admin._apps:

        cred = credentials.Certificate("config/dyslexiaid-78b8d-firebase-adminsdk-fbsvc-1fd0c51287.json")
        
      
        firebase_admin.initialize_app(cred)

def get_db():
    """Returns the Firestore client"""
    return firestore.client()