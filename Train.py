import face_recognition
import os
import pickle

# Đường dẫn đến thư mục chứa dữ liệu khuôn mặt
data_dir = "data"
encodings_file = "face_encodings.pkl"

# Danh sách lưu trữ các mã hóa khuôn mặt và tên
known_face_encodings = []
known_face_names = []

# Duyệt qua các thư mục con
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if os.path.isdir(person_path):  # Nếu là thư mục (đại diện cho 1 người)G
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            try:
                # Load ảnh và tạo mã hóa khuôn mặt
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:  # Nếu tìm thấy khuôn mặt
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_name)
            except Exception as e:
                print(f"Lỗi khi xử lý {image_path}: {e}")

# Lưu dữ liệu huấn luyện vào file
with open(encodings_file, "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print(f"Đã lưu {len(known_face_encodings)} khuôn mặt vào tệp '{encodings_file}'.")
