# Hệ Thống Chấm Công Khuôn Mặt Với Phát Hiện Chớp Mắt

## Giới thiệu

Đây là một hệ thống chấm công thông minh sử dụng công nghệ nhận diện khuôn mặt kết hợp với phát hiện chớp mắt (liveness detection). Hệ thống này giúp ngăn chặn việc giả mạo bằng ảnh tĩnh thông qua tính năng phát hiện chớp mắt, đảm bảo người dùng đang hiện diện thực sự trước camera.

## Tính năng chính

- **Nhận diện khuôn mặt**: Tự động nhận diện khuôn mặt của người dùng.
- **Phát hiện chớp mắt**: Kiểm tra tính xác thực của người dùng thông qua phát hiện chớp mắt.
- **Quản lý người dùng**: Giao diện admin để thêm, xóa và quản lý người dùng.
- **Chấm công tự động**: Hệ thống chấm công tự động khi nhận diện thành công.
- **Cơ sở dữ liệu SQLite**: Lưu trữ dữ liệu người dùng và lịch sử chấm công.

## Yêu cầu hệ thống

- Python 3.8 trở lên
- Webcam hoạt động tốt
- Các thư viện được liệt kê trong file `requirements.txt`

## Cài đặt

1. **Clone repository:**
   ```
   git clone https://github.com/Loiwj/Face-Attendance-with-Blink-Anti-Spoofing.git
   cd face_attendance_with_blink
   ```

2. **Cài đặt các thư viện cần thiết:**
   ```
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng:**
   ```
   python run.py
   ```

## Hướng dẫn sử dụng

### Đối với Admin

1. Khởi động hệ thống và chọn "Admin Mode".
2. Đăng nhập với tài khoản admin (mặc định là admin/admin nếu chưa thay đổi).
3. Thêm người dùng mới:
   - Nhập thông tin người dùng
   - Chụp hình khuôn mặt để huấn luyện hệ thống
   - Lưu thông tin người dùng
4. Quản lý người dùng: xem, cập nhật hoặc xóa thông tin người dùng.
5. Xem báo cáo chấm công.

### Đối với Người dùng

1. Khởi động hệ thống và chọn "Attendance Mode".
2. Đứng trước camera và để hệ thống nhận diện khuôn mặt.
3. Chớp mắt khi được yêu cầu để xác nhận.
4. Hệ thống sẽ tự động ghi nhận thời gian chấm công nếu nhận diện thành công.

## Cấu trúc dự án

- `run.py`: File chính để khởi động ứng dụng
- `admin_app.py`: Ứng dụng quản lý dành cho admin
- `attendance_app.py`: Ứng dụng chấm công dành cho người dùng
- `attendance.db`: Cơ sở dữ liệu SQLite
- `models/`: Chứa các mô hình đã được huấn luyện
- `utils/`: Chứa các module tiện ích
  - `db.py`: Xử lý cơ sở dữ liệu
  - `face_recog.py`: Xử lý nhận diện khuôn mặt
  - `liveness.py`: Xử lý phát hiện chớp mắt

## Bảo mật

- Hệ thống sử dụng phát hiện chớp mắt để đảm bảo người dùng đang hiện diện thực sự.
- Dữ liệu khuôn mặt được lưu trữ an toàn trong cơ sở dữ liệu.
- Chỉ admin mới có quyền truy cập và quản lý thông tin người dùng.

## Đóng góp

Chúng tôi luôn chào đón mọi đóng góp! Nếu bạn muốn cải thiện dự án, hãy:

1. Fork dự án
2. Tạo nhánh tính năng (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi của bạn (`git commit -m 'Add some amazing feature'`)
4. Push lên nhánh (`git push origin feature/amazing-feature`)
5. Mở Pull Request

## Giấy phép

Mã nguồn theo giấy phép MIT. Các mô hình InsightFace theo giấy phép Apache 2.0.

## Liên hệ

Tác giả: Dương Quốc Lợi

Email: Duongquocloi1010@gmail.com