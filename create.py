import face_recognition
import cv2
import numpy as np
import os

# Tự động train dữ liệu
known_face_encodings = []
known_face_names = []

data_dir = "data"
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    if os.path.isdir(person_path):  # Nếu là thư mục (đại diện cho 1 người)
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

print(f"Đã train {len(known_face_encodings)} khuôn mặt từ dữ liệu.")

# Giữ nguyên các phần còn lại của code
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknow"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('WEBCAM', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
