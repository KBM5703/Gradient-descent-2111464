import cv2
import dlib
import os
import numpy as np
import pickle
import time
from scipy.spatial import distance

# Sử dụng model HOG của dlib để phát hiện gương mặt
detector = dlib.get_frontal_face_detector()

# Khởi tạo face recognizer của dlib để trích xuất đặc trưng khuôn mặt
predictor = dlib.shape_predictor(r'C:\Users\acer\Desktop\NDKM_DATT\shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1(r'C:\Users\acer\Desktop\NDKM_DATT\dlib_face_recognition_resnet_model_v1.dat')

def get_face_embedding(image, face, predictor, face_rec_model):
    shape = predictor(image, face)
    face_embedding = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(face_embedding)

def is_duplicate(embedding, database, threshold=0.6):
    for registered_embedding, label in zip(database['embeddings'], database['labels']):
        dist = distance.euclidean(embedding, registered_embedding)
        if dist < threshold:
            return True, dist, label  # Trả về label của thư mục đã tồn tại
    return False, None, None

def load_database_from_directory(directory):
    """Tải dữ liệu khuôn mặt từ thư mục 'Nguoi_dang_ky'."""
    database = {'embeddings': [], 'labels': []}
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector(rgb_image)
                for face in faces:
                    embedding = get_face_embedding(rgb_image, face, predictor, face_rec_model)
                    database['embeddings'].append(embedding)
                    database['labels'].append(person_name)  # Lưu tên của người dùng đã đăng ký
    return database

def save_database(database):
    with open('embeddings_database.pkl', 'wb') as f:
        pickle.dump(database, f)

def delete_user_images(user_dir):
    """Xóa toàn bộ các file ảnh trong thư mục của người dùng."""
    for filename in os.listdir(user_dir):
        file_path = os.path.join(user_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Lỗi khi xóa file {file_path}: {e}")
    os.rmdir(user_dir)  # Xóa thư mục sau khi đã xóa hết file bên trong

def register_face(name, num_samples=5, delay_between_shots=2):
    cap = cv2.VideoCapture(0)
    embeddings = []

    # Load existing embeddings database from directory
    database = load_database_from_directory('C:\\Users\\acer\\Desktop\\NDKM_DATT\\Nguoi_dang_ky')

    # Bước 1: Kiểm tra toàn bộ dữ liệu để phát hiện trùng lặp
    print("Đang kiểm tra dữ liệu hiện có...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập vào webcam.")
            cap.release()
            cv2.destroyAllWindows()
            return

        faces = detector(frame)
        if len(faces) > 0:
            for face in faces:
                embedding = get_face_embedding(frame, face, predictor, face_rec_model)
                duplicate, dist, existing_label = is_duplicate(embedding, database)
                if duplicate:
                    print(f"Phát hiện khuôn mặt trùng lặp với người dùng đã đăng ký là '{existing_label}' (khoảng cách = {dist:.4f})")
                    response = input(f"Có phải bạn là '{existing_label}' không? (y/n): ").strip().lower()
                    if response == 'y':
                        print(f"Bạn đã đăng ký rồi với tên '{existing_label}'.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    else:
                        print("Gương mặt đã tồn tại trong hệ thống với tên khác. Không thể tiếp tục đăng ký.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            break

        cv2.imshow('Dang ky guong mat', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Bước 2: Nếu không trùng lặp, tạo thư mục mới và tiếp tục quá trình đăng ký
    user_dir = os.path.join('C:\\Users\\acer\\Desktop\\NDKM_DATT\\Nguoi_dang_ky', name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập vào webcam.")
            break

        for i in range(3, 0, -1):
            countdown_frame = frame.copy()
            cv2.putText(countdown_frame, f'{i}', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
            cv2.imshow('Dang ky guong mat', countdown_frame)
            cv2.waitKey(1000)

        faces = detector(frame)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y+h, x:x+w]

            face_filename = os.path.join(user_dir, f'{name}_{count+1}.jpg')
            cv2.imwrite(face_filename, face_img)

            embedding = get_face_embedding(frame, face, predictor, face_rec_model)

            embeddings.append(embedding)
            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} #{count}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            print(f"Đã lưu mẫu số {count}/{num_samples}")
            time.sleep(delay_between_shots)

        cv2.imshow('Dang ky guong mat', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save updated database
    for embedding in embeddings:
        database['embeddings'].append(embedding)
        database['labels'].append(name)
    save_database(database)

    if embeddings:
        print(f"Đã lưu embeddings cho {name} vào cơ sở dữ liệu.")
    else:
        print("Không thu thập được dữ liệu khuôn mặt.")

if __name__ == '__main__':
    name = input("Nhập tên để đăng ký: ")
    register_face(name)
