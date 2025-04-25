import cv2
import numpy as np
import pickle
import os

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

def load_model():
    # Load OpenCV's pre-trained face detector - using LBP cascade for better performance
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Use a more powerful face recognizer with optimal parameters
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Radius parameter for LBP
        neighbors=8,        # Number of neighbors for LBP
        grid_x=8,           # Grid size X for LBPH histogram
        grid_y=8,           # Grid size Y for LBPH histogram
        threshold=100       # Base threshold (will be fine-tuned in match_user)
    )
    
    try:
        face_recognizer.read('models/face_recognizer_model.yml')
        print("Loaded existing face recognition model")
    except:
        print("No existing model found, a new one will be created when trained")
    
    return {'detector': face_detector, 'recognizer': face_recognizer}

# Improved face detection with better preprocessing
def get_face(model, frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise (optional)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect faces with optimized parameters
    detector = model['detector']
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,     # Lower scale factor improves detection accuracy
        minNeighbors=5,      # Higher neighbor count reduces false positives
        minSize=(30, 30),    # Minimum face size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return None, None
    
    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Extract face region with some margin (15%)
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    
    # Make sure we don't go outside the image boundaries
    y_start = max(0, y - margin_y)
    y_end = min(gray.shape[0], y + h + margin_y)
    x_start = max(0, x - margin_x)
    x_end = min(gray.shape[1], x + w + margin_x)
    
    face_roi = gray[y_start:y_end, x_start:x_end]
    
    # Normalize face for consistent processing
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Apply additional preprocessing
    face_roi = cv2.equalizeHist(face_roi)
    
    return face_roi, (x, y, w, h)

# Match against trained model with adaptive threshold
def match_user(conn, model, frame, base_threshold=90):  # Changed from 80 to 90
    face_roi, face_rect = get_face(model, frame)
    if face_roi is None:
        return None, None
    
    recognizer = model['recognizer']
    try:
        # Attempt to predict using the trained model
        user_id, confidence = recognizer.predict(face_roi)
        
        print(f"Match confidence: {confidence}")  # Debug
        
        # Use adaptive threshold based on number of samples
        c = conn.cursor()
        result = c.execute("SELECT num_samples FROM face_samples WHERE user_id IN (SELECT user_id FROM users WHERE id=?)", (user_id,)).fetchone()
        
        # Adjust threshold based on number of samples (more samples = stricter threshold)
        threshold = base_threshold
        if result and result[0]:
            num_samples = result[0]
            if num_samples >= 10:
                threshold = 75  # Stricter threshold with more samples (changed from 65)
            elif num_samples >= 5:
                threshold = 85  # Medium threshold (changed from 75)
        
        if confidence < threshold:  # Lower confidence is better in LBPH
            # Get the actual user_id string from the numeric ID
            result = c.execute("SELECT user_id, name FROM users WHERE id=?", (user_id,)).fetchone()
            if result:
                return result[0], result[1]  # Return user_id and name
    except Exception as e:
        print(f"Recognition error: {e}")
    
    return None, None

# Train the face recognition model with all users in the database
def train_model(conn, model):
    c = conn.cursor()
    
    # Get all face samples from database
    c.execute("SELECT user_id, face_samples FROM face_samples")
    user_samples = c.fetchall()
    
    if not user_samples:
        print("No face samples found in database")
        return False
        
    faces = []
    labels = []
    
    # Assign numeric IDs for each user
    user_id_map = {}
    
    # First pass: Create user ID mapping
    for i, (user_id, _) in enumerate(user_samples):
        user_id_map[user_id] = i
        # Update the numeric ID mapping in the database
        c.execute("UPDATE users SET id=? WHERE user_id=?", (i, user_id))
    
    # Second pass: Process all face samples
    for user_id, face_samples_blob in user_samples:
        face_samples = pickle.loads(face_samples_blob)
        for face in face_samples:
            # Add the original face
            faces.append(face)
            labels.append(user_id_map[user_id])
            
            # Augment with small rotations for better accuracy
            if len(face_samples) < 5:  # Only augment if we have few samples
                # Rotation augmentation
                for angle in [-5, 5]:
                    h, w = face.shape
                    center = (w // 2, h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(face, rotation_matrix, (w, h))
                    faces.append(rotated)
                    labels.append(user_id_map[user_id])
    
    if not faces:
        print("No face samples to train on")
        return False
        
    print(f"Training with {len(faces)} faces across {len(user_id_map)} users")
    
    # Train the recognizer with all collected faces
    model['recognizer'].train(faces, np.array(labels))
    model['recognizer'].save('models/face_recognizer_model.yml')
    conn.commit()
    return True