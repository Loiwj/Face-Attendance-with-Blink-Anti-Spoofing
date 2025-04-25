import os
import sys
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("===== Hệ Thống Chấm Công Khuôn Mặt =====")
    print("1. Quản lý người dùng (Admin)")
    print("2. Chấm công (Người dùng)")
    print("3. Thoát")
    
    choice = input("\nChọn chức năng (1-3): ")
    
    if choice == '1':
        clear_screen()
        print("Đang khởi động ứng dụng quản lý...")
        subprocess.run(["streamlit", "run", "admin_app.py"])
    elif choice == '2':
        clear_screen()
        print("Đang khởi động ứng dụng chấm công...")
        subprocess.run(["streamlit", "run", "attendance_app.py"])
    elif choice == '3':
        print("Cảm ơn bạn đã sử dụng hệ thống!")
        sys.exit(0)
    else:
        print("Lựa chọn không hợp lệ! Vui lòng chọn lại.")
        input("Nhấn Enter để tiếp tục...")
        main()

if __name__ == "__main__":
    main()