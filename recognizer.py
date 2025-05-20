import cv2
import numpy as np
import pickle
import uuid
from mtcnn import MTCNN
# from keras_facenet import FaceNet
from tensorflow.keras.layers import TFSMLayer
from tensorflow.keras import Model, Input

# Configuration
REQUIRED_SIZE = (160, 160)
RECOGNITION_THRESHOLD = 0.8

# Load the SavedModel as a TFSMLayer
# Replace 'serving_default' if your model has a different endpoint
tfsm_layer = TFSMLayer("facenet_model", call_endpoint="serving_default")

# Create a new model wrapper
inputs = Input(shape=(160, 160, 3))
outputs = tfsm_layer(inputs)
facenet_model = Model(inputs, outputs)
# Load the facenet model
tfsm_layer = TFSMLayer("facenet_model", call_endpoint="serving_default")
inputs = Input(shape=(160, 160, 3))
facenet = Model(inputs, tfsm_layer(inputs))

# Initialize once
# facenet = FaceNet()
detector = MTCNN()

def initialize_database():
    """Create a new empty database structure."""
    return {"names": [], "embeddings": [], "uuids": [], "nims": []}

def load_database(path='database.pkl'):
    try:
        with open(path, 'rb') as f:
            database = pickle.load(f)

        # Convert embeddings to list if they are numpy arrays
        if isinstance(database['embeddings'], np.ndarray):
            database['embeddings'] = database['embeddings'].tolist()
            
        # Add uuids field if it doesn't exist (for backwards compatibility)
        if 'uuids' not in database:
            database['uuids'] = [str(uuid.uuid4()) for _ in database['names']]
            save_database(database, path)
            
        # Add nims field if it doesn't exist (for backwards compatibility)
        if 'nims' not in database:
            database['nims'] = ["" for _ in database['names']]
            save_database(database, path)

        print(f"✅ Loaded face database with {len(database['names'])} entries.")
        return database
    except FileNotFoundError:
        print("❌ Warning: 'database.pkl' not found. Creating a new empty database.")
        new_database = initialize_database()
        save_database(new_database, path)
        return new_database


def save_database(database, path='database.pkl'):
    # Convert embeddings back to NumPy array before saving
    database['embeddings'] = np.array(database['embeddings'])
    with open(path, 'wb') as f:
        pickle.dump(database, f)
    print(f"✅ Database saved with {len(database['names'])} entries.")


def preprocess_face(face, required_size=REQUIRED_SIZE):
    try:
        face = cv2.resize(face, required_size)
        return np.asarray(face)
    except:
        return None

def get_embedding(face_array):
    """
    Get face embedding using the facenet model.
    
    Args:
        face_array: Preprocessed face image array
        
    Returns:
        Face embedding vector
    """
    # Normalize the face image 
    face_array = face_array.astype('float32')
    face_array = (face_array - 127.5) / 128.0  # normalize pixel values to [-1,1]
    
    # Expand dimensions to match model input shape [batch_size, height, width, channels]
    face_array = np.expand_dims(face_array, axis=0)
    
    # Get embeddings from the model
    output = facenet_model.predict(face_array)
    
    # Handle different output formats
    if isinstance(output, dict):
        # If output is a dictionary, find the embedding tensor
        # This is common when using SavedModel format
        # The key might be something like 'embeddings' or another name
        # Try common output names or print the keys to find it
        if 'embeddings' in output:
            embeddings = output['embeddings']
        elif 'embedding' in output:
            embeddings = output['embedding']
        else:
            # Get the first value as a fallback
            embeddings = list(output.values())[0]
    else:
        # If output is a tensor or array directly
        embeddings = output
    
    # Return the embedding vector from the first batch item
    if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1:
        return embeddings[0]
    return embeddings

def recognize_face(embedding, database, threshold=0.8):
    if database is None or len(database['embeddings']) == 0:
        return "Unknown", 0.0

    distances = np.linalg.norm(np.array(database['embeddings']) - embedding, axis=1)
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    if min_distance < threshold:
        name = database['names'][min_idx]
        confidence = max(0.0, (threshold - min_distance) / threshold)
        return name, confidence
    else:
        return "Unknown", 0.0



def detect_faces_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return detector.detect_faces(rgb_frame), rgb_frame
