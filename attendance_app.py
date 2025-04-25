import streamlit as st
import cv2
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np

from utils.db import init_db, add_log, fetch_logs
from utils.face_recog import load_model, match_user
from utils.liveness import blink_detector

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

# Initialize database and model
conn = init_db()
model = load_model()

# Set page config
st.set_page_config(
    page_title="Hệ Thống Chấm Công Khuôn Mặt",
    page_icon="👁️",
    layout="wide"
)

st.title("Chấm Công Bằng Khuôn Mặt")

st.markdown("""
### Hướng dẫn sử dụng:
1. Đặt khuôn mặt của bạn trước camera
2. Nhìn thẳng vào camera và chớp mắt để chấm công
3. Hệ thống sẽ tự động nhận diện và ghi nhận thời gian chấm công
""")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Placeholder for camera feed
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Add confidence meter
    confidence_placeholder = st.empty()

with col2:
    st.subheader("Lịch sử chấm công gần đây")
    log_placeholder = st.empty()
    
    # Show the most recent check-ins
    def update_logs():
        logs = fetch_logs(conn, limit=10)
        if logs:
            # Convert to DataFrame
            df = pd.DataFrame(logs, columns=["ID", "Họ Tên", "Thời Gian"])
            # Format time
            df["Thời Gian"] = pd.to_datetime(df["Thời Gian"]).dt.strftime("%d/%m/%Y %H:%M:%S")
            return df
        return pd.DataFrame(columns=["ID", "Họ Tên", "Thời Gian"])
    
    # Initial logs
    log_placeholder.dataframe(update_logs(), use_container_width=True)
    
    # Add tips for better recognition
    st.markdown("### Mẹo cải thiện nhận diện:")
    st.markdown("""
    - Đảm bảo khuôn mặt được chiếu sáng đầy đủ
    - Nhìn thẳng vào camera khi chớp mắt
    - Tránh đeo kính râm hoặc che khuôn mặt
    - Giữ khoảng cách thích hợp với camera (30-60cm)
    """)

# Initialize the blink detector
blink_gen = blink_detector()
next(blink_gen)  # Prime the generator

# Start the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Không thể mở camera. Vui lòng kiểm tra kết nối camera.")
    st.stop()

# Variables for tracking recognition stats
recognition_attempts = 0
successful_recognitions = 0
recent_confidence_values = []
eye_open_values = []

# For tracking blink and recognition state
last_blink_time = 0
last_recognition_time = 0
current_status = "Đang chờ..."
recognition_cooldown = 3  # seconds
blink_detected = False

# Debug info toggle
show_debug = st.sidebar.checkbox("Hiển thị thông tin debug", value=False)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể đọc khung hình từ camera.")
            break
            
        # Process for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model['detector'].detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces
        current_confidence = None
        current_name = None
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Pre-check recognition to show confidence values
            if show_debug:
                test_user_id, test_name = match_user(conn, model, frame, base_threshold=1000)  # High threshold to always get a result
                
                if test_user_id:
                    current_name = test_name
                    # Extract the confidence value (it's printed to console in match_user)
                    # The next few lines will approximately extract it from printed debug statements
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        match_user(conn, model, frame, base_threshold=1000)
                    
                    debug_output = f.getvalue()
                    if "Match confidence:" in debug_output:
                        try:
                            confidence_str = debug_output.split("Match confidence:")[1].strip().split()[0]
                            current_confidence = float(confidence_str) 
                            recent_confidence_values.append(current_confidence)
                            if len(recent_confidence_values) > 10:
                                recent_confidence_values.pop(0)
                        except:
                            pass
        
        # Display confidence meter if debug enabled
        if show_debug and current_confidence is not None:
            avg_confidence = sum(recent_confidence_values) / max(1, len(recent_confidence_values))
            confidence_text = f"Độ tin cậy: {avg_confidence:.2f} - {current_name}"
            confidence_meter = np.interp(avg_confidence, [0, 100], [100, 0])  # Invert scale (lower is better)
            confidence_placeholder.progress(int(confidence_meter), f"{confidence_text}")
        else:
            confidence_placeholder.empty()
        
        # Check for blink
        try:
            blinked = blink_gen.send(frame)
            
            current_time = time.time()
            
            # Handle blink detection
            if blinked and not blink_detected:
                blink_detected = True
                last_blink_time = current_time
                current_status = "Da phat hien chop mat!"
                
                # Check for face recognition after blink
                if current_time - last_recognition_time > recognition_cooldown:
                    recognition_attempts += 1
                    user_id, name = match_user(conn, model, frame)
                    
                    if user_id:
                        # Record check-in
                        add_log(conn, user_id, name)
                        last_recognition_time = current_time
                        current_status = f"✅ Chào {name}! Đã ghi nhận chấm công."
                        successful_recognitions += 1
                        
                        # Update the log display
                        log_placeholder.dataframe(update_logs(), use_container_width=True)
                    else:
                        current_status = "❌ Khong nhan dien duoc khuon mat."
            
            # Reset blink detection after a short time
            elif blink_detected and current_time - last_blink_time > 1.0:
                blink_detected = False
        
        except Exception as e:
            st.error(f"Lỗi xử lý: {e}")
        
        # Add status text to the frame
        cv2.putText(frame, current_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add recognition stats if debug enabled
        if show_debug:
            recognition_rate = successful_recognitions / max(1, recognition_attempts) * 100
            stats_text = f"Nhận diện: {successful_recognitions}/{recognition_attempts} ({recognition_rate:.1f}%)"
            cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Convert to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Display current status
        status_placeholder.info(current_status)
        
        # Small delay to reduce CPU usage
        time.sleep(0.05)
        
except Exception as e:
    st.error(f"Lỗi: {e}")
finally:
    cap.release()