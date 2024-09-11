import cv2
import os
import numpy as np
from scipy.spatial import distance
import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Độ rộng khung hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Độ cao khung hình

# Khởi tạo bộ phát hiện khuôn mặt bằng MTCNN
detector = MTCNN()

# Khởi tạo mô hình FaceNet
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Ngưỡng khoảng cách để nhận diện (có thể điều chỉnh theo yêu cầu)
detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9])  # Điều chỉnh ngưỡng để tăng tốc
THRESHOLD = 0.5

# Hàm để trích xuất đặc trưng gương mặt bằng FaceNet
def get_face_embedding(image, face_box):
    y1, y2, x1, x2 = face_box
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))  # Resize về đúng kích thước FaceNet yêu cầu
    face = np.transpose(face, (2, 0, 1))  # Chuyển từ HWC sang CHW
    face_tensor = torch.tensor(face).float().unsqueeze(0) / 255.0  # Chuyển thành tensor và chuẩn hóa
    embedding = facenet_model(face_tensor).detach().numpy()
    return embedding.flatten()  # Chuyển thành vector 1-D

# Hàm tải các khuôn mặt đã đăng ký từ thư mục
def load_registered_faces():
    """Tải dữ liệu khuôn mặt đã đăng ký từ các thư mục người dùng."""
    registered_faces = {}
    base_dir = r'C:\Users\acer\Desktop\NDKM_DATT\Nguoi_dang_ky'  # Đường dẫn thư mục chứa ảnh của các người đã đăng ký
    valid_image_extensions = [".jpg", ".jpeg", ".png"]  # Các định dạng ảnh hợp lệ

    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):
            embeddings = []
            for image_name in os.listdir(person_dir):
                if any(image_name.lower().endswith(ext) for ext in valid_image_extensions):
                    image_path = os.path.join(person_dir, image_name)
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Phát hiện khuôn mặt trong ảnh bằng MTCNN
                    faces = detector.detect_faces(img_rgb)
                    if faces:
                        x, y, width, height = faces[0]['box']
                        face_box = (y, y+height, x, x+width)
                        embedding = get_face_embedding(img_rgb, face_box)  # Trích xuất embedding
                        embeddings.append(embedding)

            if embeddings:
                registered_faces[person_name] = embeddings

    return registered_faces


# Hàm nhận diện gương mặt
def recognize_face(live_embedding, registered_faces):
    best_match = None
    best_distance = float('inf')

    for person_name, embeddings in registered_faces.items():
        for registered_embedding in embeddings:
            # Đảm bảo rằng cả hai vector đều là 1-D
            dist = distance.euclidean(live_embedding.flatten(), registered_embedding.flatten())
            if dist < best_distance and dist < THRESHOLD:
                best_distance = dist
                best_match = person_name

    return best_match, best_distance


# Hàm nhận diện trực tiếp từ camera
def live_face_recognition():
    registered_faces = load_registered_faces()

    cap = cv2.VideoCapture(0)

    print("Bắt đầu nhận diện gương mặt...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập vào webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, width, height = face['box']
            face_box = (y, y+height, x, x+width)
            live_embedding = get_face_embedding(rgb_frame, face_box)
            person_name, dist = recognize_face(live_embedding, registered_faces)

            if person_name:
                label = f'{person_name} ({dist:.2f})'
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 2)
                cv2.putText(frame, "Nguoi_la", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Live Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_face_recognition()