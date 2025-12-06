import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from ultralytics import YOLO
import torch

# Class labels (переместили выше для использования в load_and_train_from_file)
class_labels = {
    1: "listening",
    2: "talking",
    3: "phone"
}

# === Функция загрузки и обучения модели из файла ===
def load_and_train_from_file(filename='calibration_data.pkl'):
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
            
            print(f"Найден файл {filename}, загружаем калибровку...")
            
            X = []
            y = []
            total_samples = 0
            for label, samples in loaded_data.items():
                if len(samples) == 0:
                    print(f"  Внимание: класс {class_labels.get(label, label)} пустой!")
                    continue
                for sample in samples:
                    X.append(sample)
                    y.append(label)
                total_samples += len(samples)
                print(f"  Класс {class_labels.get(label, label)}: {len(samples)} образцов")
            
            if len(X) > 0:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                print(f"Модель успешно обучена на {total_samples} образцах из {filename}!")
                return clf, loaded_data
            else:
                print("Нет данных для обучения.")
                return None, loaded_data
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return None, {}
    else:
        print(f"Файл {filename} не найден. Начинаем с чистой калибровки.")
        return None, {1: [], 2: [], 3: []}

# Загружаем лёгкую модель для детектирования лиц даже с 10–15 метров
face_detector = YOLO('yolo/yolov11n-face.pt')  # или yolov11n-face.pt (точнее)

# === Инициализация ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=12,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    static_image_mode=False
)
mp_drawing = mp.solutions.drawing_utils

# Расширенный список ключевых landmarks (добавили уши и больше точек для точности)
key_landmark_indices = [1,  # Кончик носа
                        33, 263,  # Внутренние углы глаз
                        61, 291,  # Углы рта
                        199,  # Подбородок
                        10,  # Верх лба
                        152,  # Нижняя часть подбородка
                        454, 234]  # Уши (левое и правое, если видимы)

# 3D модель точек для solvePnP (стандартные координаты для generic face model в мм)
model_points = np.array([
    (0.0, 0.0, 0.0),             # Кончик носа
    (-225.0, 170.0, -135.0),     # Левый глаз внутренний
    (225.0, 170.0, -135.0),      # Правый глаз внутренний
    (-150.0, -150.0, -125.0),    # Левый угол рта
    (150.0, -150.0, -125.0),     # Правый угол рта
    (0.0, -330.0, -65.0),        # Подбородок
    (0.0, 300.0, -100.0),        # Верх лба (примерно)
    (0.0, -400.0, -100.0),       # Нижняя часть подбородка (примерно)
    (-450.0, 0.0, -200.0),       # Левое ухо (примерно)
    (450.0, 0.0, -200.0)         # Правое ухо (примерно)
], dtype="double")

# Function to extract extended features from landmarks
def extract_features(landmarks, image_shape):
    height, width = image_shape[:2]
    
    # 1. Координаты ключевых точек (X, Y, Z) - нормализованные
    coords = []
    for idx in key_landmark_indices:
        lm = landmarks.landmark[idx]
        coords.extend([lm.x, lm.y, lm.z])  # Уже нормализованные [0,1]
    
    # 2. Углы поворота головы (yaw, pitch, roll) с использованием solvePnP
    image_points = np.array([
        (landmarks.landmark[idx].x * width, landmarks.landmark[idx].y * height)
        for idx in key_landmark_indices
    ], dtype="double")
    
    # Камера матрица (примерная, для веб-камеры; можно калибровать точнее)
    focal_length = width  # Примерно
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Без дисторсии
    
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if success:
        # Конвертируем rotation vector в Euler angles (yaw, pitch, roll)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        # Нормализуем углы (в градусах)
        angles = [yaw, pitch, roll]
    else:
        angles = [0.0, 0.0, 0.0]  # Default если не удалось
    
    # 3. Расстояния между ключевыми точками (Euclidean в 3D)
    distances = []
    pairs = [(0, 5),  # Нос к подбородку
             (1, 2),  # Глаза (между внутренними)
             (3, 4),  # Рот (между углами)
             (8, 9),  # Уши (между)
             (0, 6)]  # Нос к лбу
    for i, j in pairs:
        lm1 = landmarks.landmark[key_landmark_indices[i]]
        lm2 = landmarks.landmark[key_landmark_indices[j]]
        dist = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
        distances.append(dist)
    
    # 4. Относительные позиции (ratios)
    ratios = []
    # Пример: ширина глаз / высота лица
    eye_width = distances[1]  # Между глазами
    face_height = distances[0]  # Нос к подбородку
    ratios.append(eye_width / face_height if face_height != 0 else 0)
    # Другие ratios, например, ширина рта / ширина лица (уши)
    mouth_width = distances[2]
    ear_width = distances[3]
    ratios.append(mouth_width / ear_width if ear_width != 0 else 0)
    # Добавьте больше по необходимости
    
    # Собираем все features
    features = coords + angles + distances + ratios
    return np.array(features)

# Function to get face bounding box and center from landmarks
def get_face_bbox_and_center(landmarks, image_shape):
    height, width = image_shape[:2]
    x_coords = [lm.x * width for lm in landmarks.landmark]
    y_coords = [lm.y * height for lm in landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return min_x, min_y, max_x, max_y, (center_x, center_y)

# Простой трекер для присвоения ID лицам на основе расстояния (nearest neighbor)
def assign_ids(current_centers, previous_centers, max_id=9):
    ids = {}
    used_ids = set()
    for center in current_centers:
        min_dist = float('inf')
        closest_id = None
        for prev_id, prev_center in previous_centers.items():
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))
            if dist < min_dist and prev_id not in used_ids:
                min_dist = dist
                closest_id = prev_id
        if closest_id is None or min_dist > 100:  # Порог расстояния для нового ID
            new_id = min(set(range(max_id + 1)) - set(previous_centers.keys()))
            ids[new_id] = center
            used_ids.add(new_id)
        else:
            ids[closest_id] = center
            used_ids.add(closest_id)
    return ids

# === Загружаем данные при старте ===
clf, data = load_and_train_from_file('calibration_data.pkl')
calibration_mode = clf is None  # Если модель уже обучена — сразу в Auto mode!

# Если модель загружена — переключаемся в Auto
if clf is not None:
    calibration_mode = False
    print("Запуск в режиме AUTO (калибровка уже есть)")
else:
    print("Запуск в режиме КАЛИБРОВКИ")

# Остальные переменные
selected_tid = None
previous_centers = {}
# video_file = 'video/two_video.mp4'
# video_file = 'video/test_two_poco.mp4'
video_file = 'video/video2.mp4'
video_mode = 'camera'  # Начальный режим: 'camera' или 'file'
cap = cv2.VideoCapture(0 if video_mode == 'camera' else video_file)

print("Starting in calibration mode. Position your head and press keys to record samples:")
print(" - ',' for listening")
print(" - '.' for talking")
print(" - '/' for phone")
print(" - 's' to train the model")
print(" - 'v' to save calibration data")
print(" - 'c' to toggle between Calibration and Auto modes")
print(" - 'f' to toggle between Camera and Video File modes")
print(" - '0'-'9' to select ID for labeling (in calibration)")
print(" - 'q' to quit")
print("Aim for at least 50-100 samples per class for better accuracy.")

while True:
    ret, frame = cap.read()
    if not ret: 
        if video_mode == 'file':
            # Для видеофайла — перезапустить при конце
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break
    frame = cv2.flip(frame, 1)
    orig_frame = frame.copy()

    # ================== ДЕТЕКТИРОВАНИЕ + РЕСАЙЗ ==================
    yolo_results = face_detector(frame, device='cuda', conf=0.25, verbose=False)[0]
    boxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int) if yolo_results.boxes is not None else []

    processed_landmarks = []

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue

            h, w = roi.shape[:2]
            if h < 80 or w < 80:
                scale = max(120 / h, 120 / w, 2.5)
                roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_roi)

            if result.multi_face_landmarks:
                for landmarks in result.multi_face_landmarks:
                    # Пересчёт координат обратно в оригинальный кадр
                    scale_x = (x2 - x1) / roi.shape[1]
                    scale_y = (y2 - y1) / roi.shape[0]
                    for lm in landmarks.landmark:
                        lm.x = (lm.x * roi.shape[1] * scale_x + x1) / frame.shape[1]
                        lm.y = (lm.y * roi.shape[0] * scale_y + y1) / frame.shape[0]
                    processed_landmarks.append(landmarks)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    else:
        # Fallback — обычный Face Mesh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fallback = face_mesh.process(rgb)
        if fallback.multi_face_landmarks:
            processed_landmarks = fallback.multi_face_landmarks

    # ================== ОСНОВНАЯ ОБРАБОТКА ==================
    current_centers = []
    face_data = {}  # ← Обязательно объявляем здесь!
    face_entries = []  # (landmarks, center, features, bbox)

    if processed_landmarks:
        for landmarks in processed_landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )

            features = extract_features(landmarks, frame.shape)
            min_x, min_y, max_x, max_y, center = get_face_bbox_and_center(landmarks, frame.shape)

            current_centers.append(center)
            face_entries.append({
                'landmarks': landmarks,
                'features': features,
                'bbox': (min_x, min_y, max_x, max_y),
                'center': center
            })

        # Присвоение ID по центрам
        current_ids = assign_ids(current_centers, previous_centers)
        previous_centers = current_ids  # Обновляем для следующего кадра

        # Отображаем всё
        for idx, entry in enumerate(face_entries):
            center = entry['center']
            features = entry['features']
            min_x, min_y, _, _ = entry['bbox']

            # Находим ID по ближайшему центру (надёжно)
            tid = None
            for cid, saved_center in current_ids.items():
                if np.linalg.norm(np.array(center) - np.array(saved_center)) < 5:  # маленькая погрешность
                    tid = cid
                    break
            if tid is None:
                tid = 999  # если что-то пошло не так

            # ID над лицом
            cv2.putText(frame, f"ID:{tid}", (min_x, min_y - 35),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)

            # Предсказание активности (только в AUTO режиме)
            if not calibration_mode and clf is not None:
                pred = clf.predict([features])[0]
                label = class_labels.get(pred, "unknown")
                cv2.putText(frame, label, (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

            # Сохраняем для калибровки (по ID)
            face_data[tid] = (features, (min_x, min_y))

    # Режим и инструкции
    mode_text = "AUTO" if not calibration_mode else "CALIBRATION"
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if not calibration_mode else (0, 255, 0), 2)

    if calibration_mode:
        cv2.putText(frame, "0-9: select  ,:listening  .:talking  /:phone  s:train  v:save", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Head Pose Classifier", frame)

    # ================== КЛАВИАТУРА ==================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'):
        calibration_mode = not calibration_mode
        selected_tid = None
        print(f"Режим: {'CALIBRATION' if calibration_mode else 'AUTO'}")
    if key == ord('f'):
        cap.release()
        if video_mode == 'camera':
            video_mode = 'file'
            cap = cv2.VideoCapture(video_file)
            print(f"Переключено на видеофайл: {video_file}")
        else:
            video_mode = 'camera'
            cap = cv2.VideoCapture(0)
            print("Переключено на камеру")

    if calibration_mode:
        if ord('0') <= key <= ord('9'):
            num = key - ord('0')
            if num in face_data:
                selected_tid = num
                print(f"Выбран ID {num}")
        if selected_tid is not None and selected_tid in face_data:
            features, _ = face_data[selected_tid]
            if key == ord(','): data[1].append(features); print(f"ID {selected_tid} → listening ({len(data[1])})")
            if key == ord('.'): data[2].append(features); print(f"ID {selected_tid} → talking ({len(data[2])})")
            if key == ord('/'): data[3].append(features); print(f"ID {selected_tid} → phone ({len(data[3])})")
        if key == ord('s'):
            X, y = [], []
            for l, samples in data.items():
                X.extend(samples); y.extend([l] * len(samples))
            if len(X) > 10:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                print("Модель обучена!")
        if key == ord('v'):
            with open('calibration_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("Калибровка сохранена!")

cap.release()
cv2.destroyAllWindows()