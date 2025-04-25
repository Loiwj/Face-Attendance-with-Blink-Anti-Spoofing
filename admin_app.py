import streamlit as st
import cv2
import pickle
import uuid
import numpy as np
from datetime import datetime
import pandas as pd
import time
import os

from utils.db import init_db, add_user, add_face_samples, fetch_users, fetch_logs, delete_user, get_face_samples
from utils.face_recog import load_model, get_face, train_model

# Make sure the models directory exists
os.makedirs('models', exist_ok=True)

# Initialize database and model
conn = init_db()
model = load_model()

# Set page config
st.set_page_config(
    page_title="Hệ Thống Chấm Công Khuôn Mặt",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sidebar for navigation
st.sidebar.title("Quản Lý Chấm Công")
page = st.sidebar.radio("Trang", ["Thêm Người Dùng", "Quản Lý Người Dùng", "Lịch Sử Chấm Công", "Huấn Luyện Mô Hình"])

# Simplified camera function that doesn't auto-rerun
def capture_face_simple():
    st.write("Đang mở camera. Vui lòng đợi...")
    
    # Placeholder for camera feed
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    # Capture controls
    col1, col2 = st.columns(2)
    with col1:
        capture_button = st.button("Chụp Ảnh", key="capture_simple")
    with col2:
        stop_button = st.button("Dừng", key="stop_simple")
        
    # Start camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Không thể mở camera. Vui lòng kiểm tra kết nối camera.")
        return None
    
    face_img = None
    running = True
    
    # Simple loop without experimental_rerun
    while running:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể đọc khung hình từ camera.")
            break
            
        # Get face from frame
        face_roi, face_rect = get_face(model, frame)
        
        # Draw rectangle around face if detected
        if face_rect is not None:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            status = "Đã phát hiện khuôn mặt! Nhấn 'Chụp Ảnh' để lưu."
        else:
            status = "Không phát hiện khuôn mặt. Vui lòng điều chỉnh vị trí."
            
        # Display status
        status_text.text(status)
        
        # Convert to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Check for button clicks
        if capture_button and face_roi is not None:
            face_img = face_roi
            break
            
        if stop_button:
            break
            
        # Small delay to reduce CPU usage
        time.sleep(0.05)
        
    # Close camera
    cap.release()
    
    if face_img is not None:
        st.success("Đã chụp khuôn mặt thành công!")
        
    return face_img

# Add user page
if page == "Thêm Người Dùng":
    st.title("Thêm Người Dùng Mới")
    
    # Form to collect user information
    with st.form("user_form"):
        name = st.text_input("Họ và Tên")
        submit_button = st.form_submit_button("Tiếp tục")
        
        if submit_button and name:
            # Generate a unique ID for the user
            user_id = str(uuid.uuid4())[:8]
            
            # Add user to database
            add_user(conn, user_id, name)
            
            # Store in session state
            st.session_state['current_user_id'] = user_id
            st.session_state['current_user_name'] = name
            st.session_state['face_samples'] = []
            
            st.success(f"Đã tạo người dùng {name}. Tiếp tục thu thập mẫu khuôn mặt.")
            st.experimental_rerun()

    # If there's a current user, collect face samples
    if 'current_user_id' in st.session_state:
        st.subheader(f"Thu thập mẫu khuôn mặt cho {st.session_state['current_user_name']}")
        
        st.write(f"Số mẫu đã thu thập: {len(st.session_state.get('face_samples', []))}")
        st.write("Vui lòng thu thập ít nhất 5 mẫu với các góc khác nhau để cải thiện độ chính xác.")
        
        # Initialize mode state
        if 'capturing' not in st.session_state:
            st.session_state.capturing = False
        
        # Button to start/stop capturing
        if not st.session_state.capturing:
            if st.button("Bắt đầu thu thập mẫu", key="start_simple_capture"):
                st.session_state.capturing = True
                st.experimental_rerun()
        else:
            # In capture mode
            face_img = capture_face_simple()
                
            # If a face was captured
            if face_img is not None:
                # Add to session state
                if 'face_samples' not in st.session_state:
                    st.session_state['face_samples'] = []
                
                st.session_state['face_samples'].append(face_img)
                st.session_state.capturing = False
                st.success(f"Đã thu thập {len(st.session_state['face_samples'])} mẫu khuôn mặt!")
                st.experimental_rerun()
            else:
                # If stopped without capturing
                if st.button("Quay lại", key="back_from_capture"):
                    st.session_state.capturing = False
                    st.experimental_rerun()
            
        # Display captured samples
        if 'face_samples' in st.session_state and st.session_state['face_samples'] and not st.session_state.capturing:
            st.write("Mẫu khuôn mặt đã thu thập:")
            
            # Display in a grid - 3 columns
            cols = st.columns(3)
            for i, face in enumerate(st.session_state['face_samples']):
                cols[i % 3].image(face, width=100, caption=f"Mẫu {i+1}")
        
        # Save all face samples - only show when not capturing
        if not st.session_state.capturing and 'face_samples' in st.session_state and len(st.session_state['face_samples']) > 0:
            if st.button("Lưu tất cả mẫu khuôn mặt"):
                # Serialize face samples
                face_samples_blob = pickle.dumps(st.session_state['face_samples'])
                
                # Save to database
                add_face_samples(conn, st.session_state['current_user_id'], face_samples_blob)
                
                st.success("Đã lưu tất cả mẫu khuôn mặt thành công!")
                
                # Clear session state
                del st.session_state['current_user_id']
                del st.session_state['current_user_name']
                del st.session_state['face_samples']
                st.session_state.capturing = False
                
                # Train the model with the new data
                st.info("Đang huấn luyện mô hình...")
                if train_model(conn, model):
                    st.success("Đã huấn luyện mô hình thành công!")
                else:
                    st.warning("Không thể huấn luyện mô hình. Vui lòng kiểm tra dữ liệu.")
                
                # Redirect to user management
                st.experimental_rerun()

# User management page
elif page == "Quản Lý Người Dùng":
    st.title("Quản Lý Người Dùng")
    
    # Fetch all users
    users = fetch_users(conn)
    
    if not users:
        st.warning("Chưa có người dùng nào trong hệ thống.")
    else:
        # Convert to DataFrame for easier display
        df = pd.DataFrame(users, columns=["ID", "Họ Tên", "Ngày Tạo", "Số Mẫu Khuôn Mặt"])
        
        # Format date
        df["Ngày Tạo"] = pd.to_datetime(df["Ngày Tạo"]).dt.strftime("%d/%m/%Y %H:%M")
        
        # Add a button column for delete
        st.dataframe(df, use_container_width=True)
        
        # User selection for action
        selected_user = st.selectbox("Chọn người dùng:", 
                                    options=[f"{user[1]} (ID: {user[0]})" for user in users],
                                    format_func=lambda x: x.split(" (ID: ")[0])
        
        if selected_user:
            user_id = selected_user.split(" (ID: ")[1][:-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Xóa Người Dùng", type="primary", use_container_width=True):
                    delete_user(conn, user_id)
                    st.success(f"Đã xóa người dùng {selected_user.split(' (ID: ')[0]}")
                    st.experimental_rerun()
            
            with col2:
                if st.button("Thu Thập Thêm Mẫu Khuôn Mặt", use_container_width=True):
                    # Find the user in the users list
                    user_name = ""
                    for user in users:
                        if user[0] == user_id:
                            user_name = user[1]
                            break
                    
                    # Get existing face samples
                    existing_samples_blob = get_face_samples(conn, user_id)
                    existing_samples = []
                    
                    if existing_samples_blob:
                        existing_samples = pickle.loads(existing_samples_blob)
                    
                    # Store in session state
                    st.session_state['current_user_id'] = user_id
                    st.session_state['current_user_name'] = user_name
                    st.session_state['face_samples'] = existing_samples
                    
                    # Redirect to Add User page
                    st.experimental_rerun()

# Attendance log page
elif page == "Lịch Sử Chấm Công":
    st.title("Lịch Sử Chấm Công")
    
    logs = fetch_logs(conn, limit=100)
    
    if not logs:
        st.warning("Chưa có dữ liệu chấm công.")
    else:
        # Convert to DataFrame
        df = pd.DataFrame(logs, columns=["ID", "Họ Tên", "Thời Gian"])
        
        # Format time
        df["Thời Gian"] = pd.to_datetime(df["Thời Gian"]).dt.strftime("%d/%m/%Y %H:%M:%S")
        
        # Group by date for better display
        df['Ngày'] = pd.to_datetime(df["Thời Gian"]).dt.date
        
        # Create expandable sections for each date
        unique_dates = sorted(df['Ngày'].unique(), reverse=True)
        
        for date in unique_dates:
            date_str = date.strftime("%d/%m/%Y")
            with st.expander(f"Ngày: {date_str}", expanded=(date == unique_dates[0])):
                day_logs = df[df['Ngày'] == date].drop('Ngày', axis=1)
                st.dataframe(day_logs, use_container_width=True)

# Model training page
elif page == "Huấn Luyện Mô Hình":
    st.title("Huấn Luyện Mô Hình Nhận Diện Khuôn Mặt")
    
    st.write("""
// Huấn luyện mô hình nhận diện khuôn mặt với tất cả dữ liệu hiện có.
// Bạn nên thực hiện việc này sau khi thêm người dùng mới hoặc thu thập thêm mẫu khuôn mặt.
    """)
    
    if st.button("Huấn Luyện Mô Hình", type="primary"):
        with st.spinner("Đang huấn luyện mô hình..."):
            success = train_model(conn, model)
            
        if success:
            st.success("Huấn luyện mô hình thành công!")
        else:
            st.error("Không thể huấn luyện mô hình. Vui lòng kiểm tra dữ liệu.")