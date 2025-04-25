import mediapipe as mp
import numpy as np
import cv2
import time

mpfm = mp.solutions.face_mesh
# Eye landmarks from MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# Calculate Eye Aspect Ratio - improved precision
def eye_aspect_ratio(landmarks, eye_idxs, img_w, img_h):
    pts = [(int(landmarks[idx].x*img_w), int(landmarks[idx].y*img_h)) for idx in eye_idxs]
    # vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # horizontal
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0

# Enhanced blink detection generator with temporal smoothing
def blink_detector():
    # Use static_image_mode=False for better tracking of moving faces
    with mpfm.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        # Initialize variables for blink detection state machine
        frame_count = 0
        blinked = False
        open_eye_threshold = 0.25  # Higher value for more sensitive detection
        closed_eye_threshold = 0.20  # Threshold for closed eye
        
        # Variables for temporal smoothing
        ear_history = []
        history_length = 3
        
        # Variables for blink state tracking
        eyes_open = True
        last_blink_time = 0
        min_blink_interval = 1.0  # Minimum time between blinks in seconds
        
        while True:
            frame = yield blinked
            img_h, img_w = frame.shape[:2]
            blinked = False
            frame_count += 1
            
            # Process the frame with MediaPipe
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                
                # Calculate EAR for both eyes
                ear_l = eye_aspect_ratio(lm, LEFT_EYE, img_w, img_h)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE, img_w, img_h)
                
                # Use minimum of both eyes
                current_ear = min(ear_l, ear_r)
                
                # Add to history for smoothing
                ear_history.append(current_ear)
                if len(ear_history) > history_length:
                    ear_history.pop(0)
                
                # Smoothed EAR value
                smoothed_ear = sum(ear_history) / len(ear_history)
                
                # State machine for blink detection
                current_time = time.time()
                
                if eyes_open and smoothed_ear < closed_eye_threshold:
                    # Eyes just closed - potential start of a blink
                    eyes_open = False
                    
                elif not eyes_open and smoothed_ear > open_eye_threshold:
                    # Eyes just opened - complete blink detected
                    eyes_open = True
                    
                    # Check if enough time has passed since the last blink
                    if current_time - last_blink_time > min_blink_interval:
                        blinked = True
                        last_blink_time = current_time
                        frame_count = 0
            
            # Reset blink detection if no face found for several frames
            if frame_count > 10:
                blinked = False
                ear_history = []