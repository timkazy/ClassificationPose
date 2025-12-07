import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from ultralytics import YOLO
import torch
from datetime import datetime

# Class labels
class_labels = {
    1: "listening",
    2: "talking", 
    3: "phone"
}

# Цвета для каждого класса
class_colors = {
    1: (0, 255, 0),    # Зеленый - listening
    2: (255, 0, 0),    # Синий - talking
    3: (255, 0, 255),  # Фиолетовый - phone
    "unknown": (128, 128, 128)  # Серый - неизвестно
}

# === Функция создания файла калибровки если его нет ===
def create_calibration_file_if_not_exists(filename='calibration_data.pkl'):
    """Создает файл калибровки с пустыми данными, если он не существует"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не существует. Создаем новый с пустыми данными...")
        empty_data = {1: [], 2: [], 3: []}
        with open(filename, 'wb') as f:
            pickle.dump(empty_data, f)
        print(f"Файл {filename} успешно создан.")
        return empty_data
    return None

# === Функция загрузки и обучения модели из файла ===
def load_and_train_from_file(filename='trained_data.pkl'):
    """Загружает предобученную модель из trained_data.pkl"""
    try:
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Найден файл {filename}, загружаем обученную модель...")
        
        # Проверяем структуру данных
        if 'model' in loaded_data and 'class_labels' in loaded_data:
            clf = loaded_data['model']
            method_name = loaded_data.get('method_name', 'Unknown')
            print(f"Модель успешно загружена из {filename}!")
            print(f"Метод обучения: {method_name}")
            
            # Загружаем сырые данные для калибровки, если они есть
            if os.path.exists('raw_calibration_data.pkl'):
                with open('raw_calibration_data.pkl', 'rb') as f_calib:
                    calib_data = pickle.load(f_calib)
                print(f"Сырые данные калибровки: {sum(len(v) for v in calib_data.values())} образцов")
            else:
                calib_data = {1: [], 2: [], 3: []}
                
            return clf, calib_data
        else:
            print("Файл не содержит модель в ожидаемом формате.")
            return None, {1: [], 2: [], 3: []}
    except Exception as e:
        print(f"Ошибка при загрузке файла {filename}: {e}")
        print("Пытаемся загрузить старую калибровку...")
        # Пробуем загрузить старую калибровку
        return load_old_calibration()

def load_old_calibration(filename='calibration_data.pkl'):
    """Загружает старую калибровку для совместимости"""
    create_calibration_file_if_not_exists(filename)
    
    try:
        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Загружаем калибровку из {filename}...")
        
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
        print("Создаем чистую калибровку...")
        return None, {1: [], 2: [], 3: []}

# === Функция сохранения сырых данных ===
def save_raw_data(data, filename='raw_calibration_data.pkl'):
    """Сохраняет сырые данные калибровки в отдельный файл"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Сырые данные сохранены в {filename}")
        print(f"Общее количество образцов: {sum(len(v) for v in data.values())}")
        for class_id in [1, 2, 3]:
            print(f"  Класс {class_labels.get(class_id, class_id)}: {len(data.get(class_id, []))}")
    except Exception as e:
        print(f"Ошибка при сохранении сырых данных: {e}")

# === Функция очистки всех меток калибровки ===
def clear_calibration_data(filename='calibration_data.pkl'):
    """Очищает все метки в файле калибровки"""
    empty_data = {1: [], 2: [], 3: []}
    with open(filename, 'wb') as f:
        pickle.dump(empty_data, f)
    
    # Также очищаем сырые данные
    if os.path.exists('raw_calibration_data.pkl'):
        with open('raw_calibration_data.pkl', 'wb') as f:
            pickle.dump(empty_data, f)
    
    print(f"Все метки калибровки в {filename} были очищены.")
    return empty_data

# === Функция создания информационной панели ===
def create_info_panel(people_info, calibration_mode, calibration_counts, fps=None, frame_size=None):
    """Создает второе окно с информацией о людях"""
    
    # Размеры панели
    panel_width = 500
    panel_height = 600
    
    # Создаем черную панель
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Заголовок
    title = "PEOPLE INFORMATION" if not calibration_mode else "CALIBRATION MODE"
    title_color = (0, 200, 255) if not calibration_mode else (0, 255, 100)
    cv2.putText(panel, title, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, title_color, 2)
    
    # Время
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, f"Time: {current_time}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # FPS и статистика
    if fps:
        cv2.putText(panel, f"FPS: {fps:.1f}", (panel_width - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    # Разделительная линия
    cv2.line(panel, (10, 90), (panel_width - 10, 90), (100, 100, 100), 1)
    
    # === Секция статистики калибровки ===
    cv2.putText(panel, "Calibration Data:", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 1)
    
    y_pos = 135
    total_calibration_samples = sum(calibration_counts.values())
    
    # Отображаем количество образцов для каждого класса
    for class_id in [1, 2, 3]:
        class_name = class_labels.get(class_id, f"Class {class_id}")
        count = calibration_counts.get(class_id, 0)
        color = class_colors.get(class_id, (255, 255, 255))
        
        # Цветной квадрат
        cv2.rectangle(panel, (25, y_pos - 15), (45, y_pos + 5), color, -1)
        cv2.putText(panel, f"{class_name}: {count} samples", (55, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    # Общее количество образцов
    cv2.putText(panel, f"Total: {total_calibration_samples} samples", (55, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
    y_pos += 30
    
    # Разделительная линия
    cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
    y_pos += 20
    
    # === Сводная статистика текущих людей ===
    total_people = len(people_info)
    class_counts = {"listening": 0, "talking": 0, "phone": 0, "unknown": 0}
    
    for info in people_info.values():
        if "prediction" in info:
            pred_label = info["prediction"]
            if pred_label in class_counts:
                class_counts[pred_label] += 1
            else:
                class_counts["unknown"] += 1
    
    # Отображаем статистику
    cv2.putText(panel, f"Total People: {total_people}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 30
    
    # Легенда классов
    cv2.putText(panel, "Current Classes:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 1)
    y_pos += 25
    
    for class_id, class_name in class_labels.items():
        color = class_colors.get(class_id, (255, 255, 255))
        count = class_counts.get(class_name, 0)
        
        # Цветной квадрат
        cv2.rectangle(panel, (25, y_pos - 15), (45, y_pos + 5), color, -1)
        cv2.putText(panel, f"{class_name}: {count}", (55, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    y_pos += 10
    cv2.line(panel, (10, y_pos), (panel_width - 10, y_pos), (100, 100, 100), 1)
    y_pos += 20
    
    # Детальная информация по каждому человеку
    cv2.putText(panel, "Detailed Info:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
    y_pos += 30
    
    # Сортируем ID для удобного отображения
    sorted_ids = sorted(people_info.keys())
    
    for person_id in sorted_ids:
        info = people_info[person_id]
        
        # Если достигли нижней части панели, создаем новую колонку
        if y_pos > panel_height - 80:
            y_pos = 200
            x_offset = 250
        else:
            x_offset = 20
        
        # ID и позиция
        cv2.putText(panel, f"ID {person_id}:", (x_offset, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Класс (если есть)
        if "prediction" in info:
            pred_label = info["prediction"]
            color = class_colors.get(1 if pred_label == "listening" else 2 if pred_label == "talking" else 3 if pred_label == "phone" else "unknown", (255, 255, 255))
            cv2.putText(panel, pred_label, (x_offset + 80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Координаты (если есть)
        if "center" in info:
            x, y = info["center"]
            cv2.putText(panel, f"({int(x)}, {int(y)})", (x_offset + 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_pos += 25
    
    # Инструкции внизу
    y_pos = panel_height - 40
    instructions = "Press 'c': toggle mode | 'r': clear calibration | 'q': quit"
    cv2.putText(panel, instructions, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    return panel

# Загружаем лёгкую модель для детектирования лиц
face_detector = YOLO('yolo/yolov11l-face.pt')

# === Инициализация ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=12,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
    static_image_mode=False
)
mp_drawing = mp.solutions.drawing_utils

# Расширенный список ключевых landmarks
key_landmark_indices = [1,  # Кончик носа
                        33, 263,  # Внутренние углы глаз
                        61, 291,  # Углы рта
                        199,  # Подбородок
                        10,  # Верх лба
                        152,  # Нижняя часть подбородка
                        454, 234]  # Уши (левое и правое, если видимы)

# 3D модель точек для solvePnP
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
        coords.extend([lm.x, lm.y, lm.z])
    
    # 2. Углы поворота головы
    image_points = np.array([
        (landmarks.landmark[idx].x * width, landmarks.landmark[idx].y * height)
        for idx in key_landmark_indices
    ], dtype="double")
    
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        angles = [yaw, pitch, roll]
    else:
        angles = [0.0, 0.0, 0.0]
    
    # 3. Расстояния между ключевыми точками
    distances = []
    pairs = [(0, 5), (1, 2), (3, 4), (8, 9), (0, 6)]
    for i, j in pairs:
        lm1 = landmarks.landmark[key_landmark_indices[i]]
        lm2 = landmarks.landmark[key_landmark_indices[j]]
        dist = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
        distances.append(dist)
    
    # 4. Относительные позиции
    ratios = []
    eye_width = distances[1]
    face_height = distances[0]
    ratios.append(eye_width / face_height if face_height != 0 else 0)
    
    mouth_width = distances[2]
    ear_width = distances[3]
    ratios.append(mouth_width / ear_width if ear_width != 0 else 0)
    
    features = coords + angles + distances + ratios
    return np.array(features)

# Function to get face bounding box and center
def get_face_bbox_and_center(landmarks, image_shape):
    height, width = image_shape[:2]
    x_coords = [lm.x * width for lm in landmarks.landmark]
    y_coords = [lm.y * height for lm in landmarks.landmark]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return min_x, min_y, max_x, max_y, (center_x, center_y)

# Простой трекер для присвоения ID лицам
def assign_ids(current_centers, previous_centers, max_id=20):
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
        
        if closest_id is None or min_dist > 100:
            # Найти новый свободный ID
            for new_id in range(max_id + 1):
                if new_id not in previous_centers and new_id not in used_ids:
                    ids[new_id] = center
                    used_ids.add(new_id)
                    break
        else:
            ids[closest_id] = center
            used_ids.add(closest_id)
    
    return ids

# === Загружаем данные при старте ===
clf, data = load_and_train_from_file('trained_data.pkl')
calibration_mode = clf is None

if clf is not None:
    calibration_mode = False
    print("Запуск в режиме AUTO (обученная модель загружена)")
else:
    print("Запуск в режиме КАЛИБРОВКИ")

# Остальные переменные
selected_tid = None
previous_centers = {}
people_info = {}  # Словарь для хранения информации о людях

# Для расчета FPS
fps_start_time = cv2.getTickCount()
fps_frame_count = 0
current_fps = 0

video_file = 'video/video2.mp4'
video_mode = 'camera'
cap = cv2.VideoCapture(0 if video_mode == 'camera' else video_file)

print("\nУправление:")
print(" - 'c' - переключить режим (Calibration/Auto)")
print(" - '0'-'9' - выбрать ID для калибровки")
print(" - ',' - добавить listening")
print(" - '.' - добавить talking")
print(" - '/' - добавить phone")
print(" - 's' - обучить модель")
print(" - 'v' - сохранить калибровку")
print(" - 'r' - очистить ВСЕ метки калибровки")
print(" - 'q' - выход")

# Создаем окна заранее
cv2.namedWindow("Head Pose Classifier", cv2.WINDOW_NORMAL)
cv2.namedWindow("People Information", cv2.WINDOW_NORMAL)
cv2.resizeWindow("People Information", 500, 600)

while True:
    ret, frame = cap.read()
    if not ret: 
        if video_mode == 'video':
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break
    
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # Расчет FPS
    fps_frame_count += 1
    if fps_frame_count >= 30:
        fps_end_time = cv2.getTickCount()
        time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
        current_fps = fps_frame_count / time_diff
        fps_start_time = fps_end_time
        fps_frame_count = 0
    
    # ================== ДЕТЕКТИРОВАНИЕ ==================
    yolo_results = face_detector(frame, device='cuda', conf=0.3, verbose=False)[0]
    boxes = yolo_results.boxes.xyxy.cpu().numpy().astype(int) if yolo_results.boxes is not None else []

    processed_landmarks = []
    face_data = {}
    face_entries = []

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: 
                continue
            
            h, w = roi.shape[:2]
            if h < 80 or w < 80:
                scale = max(120 / h, 120 / w, 2.5)
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            result = face_mesh.process(roi)
            
            if result.multi_face_landmarks:
                # === ПРОСТОЙ ФИЛЬТР: берем только самую большую сетку ===
                if len(result.multi_face_landmarks) > 1:
                    # Находим сетку с максимальной площадью
                    max_area = 0
                    best_landmarks = None
                    
                    for landmarks in result.multi_face_landmarks:
                        # Вычисляем bounding box сетки
                        xs = [lm.x for lm in landmarks.landmark]
                        ys = [lm.y for lm in landmarks.landmark]
                        
                        width = max(xs) - min(xs)
                        height = max(ys) - min(ys)
                        area = width * height
                        
                        # Игнорируем очень маленькие сетки (глаза, рот)
                        if area > 0.05:  # Минимум 5% от ROI
                            if area > max_area:
                                max_area = area
                                best_landmarks = landmarks
                    
                    if best_landmarks is not None:
                        landmarks = best_landmarks
                    else:
                        # Если все сетки слишком маленькие, берем самую большую
                        landmarks = max(result.multi_face_landmarks, 
                                    key=lambda lm: (max([p.x for p in lm.landmark]) - min([p.x for p in lm.landmark])) * 
                                                    (max([p.y for p in lm.landmark]) - min([p.y for p in lm.landmark])))
                else:
                    landmarks = result.multi_face_landmarks[0]
                
                # Пересчёт координат из ROI в глобальные координаты кадра
                scale_x = (x2 - x1) / roi.shape[1]
                scale_y = (y2 - y1) / roi.shape[0]
                
                for lm in landmarks.landmark:
                    lm.x = (lm.x * roi.shape[1] * scale_x + x1) / frame.shape[1]
                    lm.y = (lm.y * roi.shape[0] * scale_y + y1) / frame.shape[0]
                
                processed_landmarks.append(landmarks)
                
                # Рисуем bounding box YOLO
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Отладочная информация
                cv2.putText(display_frame, f"MP: 1/{len(result.multi_face_landmarks)}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    else:
        # Fallback (оставляем как было, но тоже фильтруем)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fallback = face_mesh.process(rgb)
        
        if fallback.multi_face_landmarks:
            # Простой фильтр для fallback
            if len(fallback.multi_face_landmarks) > 1:
                # Вычисляем площади всех сеток
                areas = []
                for landmarks in fallback.multi_face_landmarks:
                    xs = [lm.x for lm in landmarks.landmark]
                    ys = [lm.y for lm in landmarks.landmark]
                    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                    areas.append(area)
                
                # Берем только сетки с площадью > 10% от максимальной
                max_area = max(areas)
                processed_landmarks = []
                for i, landmarks in enumerate(fallback.multi_face_landmarks):
                    if areas[i] > max_area * 0.1:  # Не менее 10% от самой большой
                        processed_landmarks.append(landmarks)
            else:
                processed_landmarks = fallback.multi_face_landmarks
    
    # ================== ОСНОВНАЯ ОБРАБОТКА ==================
    current_centers = []
    people_info = {}  # Сбрасываем информацию
    
    if processed_landmarks:
        for idx, landmarks in enumerate(processed_landmarks):
            # Рисуем landmarks
            mp_drawing.draw_landmarks(
                display_frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )
            
            # Извлекаем фичи
            features = extract_features(landmarks, frame.shape)
            min_x, min_y, max_x, max_y, center = get_face_bbox_and_center(landmarks, frame.shape)
            
            current_centers.append(center)
            
            face_entries.append({
                'landmarks': landmarks,
                'features': features,
                'bbox': (min_x, min_y, max_x, max_y),
                'center': center,
                'index': idx
            })
        
        # Присвоение ID
        current_ids = assign_ids(current_centers, previous_centers)
        previous_centers = current_ids
        
        # Обработка каждого лица
        for entry in face_entries:
            center = entry['center']
            features = entry['features']
            min_x, min_y, _, _ = entry['bbox']
            idx = entry['index']
            
            # Находим ID
            tid = None
            for cid, saved_center in current_ids.items():
                if np.linalg.norm(np.array(center) - np.array(saved_center)) < 5:
                    tid = cid
                    break
            if tid is None:
                tid = idx
            
            # Предсказание
            prediction_label = "unknown"
            if not calibration_mode and clf is not None:
                try:
                    pred = clf.predict([features])[0]
                    prediction_label = class_labels.get(pred, "unknown")
                    
                    # Отображаем предсказание
                    color = class_colors.get(pred, class_colors["unknown"])
                    cv2.putText(display_frame, prediction_label, 
                               (min_x, min_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except:
                    pass
            
            # Отображаем ID
            cv2.putText(display_frame, f"ID:{tid}", 
                       (min_x, min_y - 35),
                       cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
            
            # Сохраняем информацию для панели
            people_info[tid] = {
                'center': center,
                'prediction': prediction_label,
                'bbox': (min_x, min_y, max_x, max_y),
                'features': features
            }
            
            # Сохраняем для калибровки
            face_data[tid] = (features, (min_x, min_y))
    
    # ================== ОТОБРАЖЕНИЕ НА ОСНОВНОМ ОКНЕ ==================
    # Режим
    mode_text = "AUTO" if not calibration_mode else "CALIBRATION"
    mode_color = (0, 0, 255) if not calibration_mode else (0, 255, 0)
    cv2.putText(display_frame, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
    
    # FPS
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                (display_frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Количество людей
    cv2.putText(display_frame, f"People: {len(processed_landmarks)}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if calibration_mode:
        cv2.putText(display_frame, "0-9: select  ,:listen  .:talk  /:phone", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    cv2.imshow("Head Pose Classifier", display_frame)
    
    # ================== СОЗДАНИЕ И ОТОБРАЖЕНИЕ ИНФОРМАЦИОННОЙ ПАНЕЛИ ==================
    # Подсчитываем количество образцов в калибровочных данных
    calibration_counts = {}
    for class_id in [1, 2, 3]:
        calibration_counts[class_id] = len(data.get(class_id, []))
    
    info_panel = create_info_panel(people_info, calibration_mode, calibration_counts, current_fps, frame.shape[:2])
    cv2.imshow("People Information", info_panel)
    
    # ================== ОБРАБОТКА КЛАВИШ ==================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    
    if key == ord('c'):
        calibration_mode = not calibration_mode
        selected_tid = None
        print(f"Режим: {'CALIBRATION' if calibration_mode else 'AUTO'}")
    
    if key == ord('f'):
        cap.release()
        if video_mode == 'camera':
            video_mode = 'video'
            cap = cv2.VideoCapture(video_file)
            print(f"Переключено на видеофайл: {video_file}")
        else:
            video_mode = 'camera'
            cap = cv2.VideoCapture(0)
            print("Переключено на камеру")
    
    # Обработка калибровки
    if calibration_mode:
        if ord('0') <= key <= ord('9'):
            num = key - ord('0')
            if num in face_data:
                selected_tid = num
                print(f"Выбран ID {num}")
        
        if selected_tid is not None and selected_tid in face_data:
            features, _ = face_data[selected_tid]
            if key == ord(','): 
                data[1].append(features)
                print(f"ID {selected_tid} → listening ({len(data[1])})")
            if key == ord('.'): 
                data[2].append(features)
                print(f"ID {selected_tid} → talking ({len(data[2])})")
            if key == ord('/'): 
                data[3].append(features)
                print(f"ID {selected_tid} → phone ({len(data[3])})")
        
        if key == ord('s'):
            X, y = [], []
            for l, samples in data.items():
                X.extend(samples)
                y.extend([l] * len(samples))
            if len(X) > 10:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                calibration_mode = False
                print(f"Модель обучена на {len(X)} образцах!")
        
        if key == ord('v'):
            save_raw_data(data, 'raw_calibration_data.pkl')
            with open('calibration_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            print("Калибровка сохранена!")
    
    # Обработка кнопки 'r' для очистки калибровки
    if key == ord('r'):
        # Спрашиваем подтверждение
        print("\n=== ВНИМАНИЕ ===")
        print("Вы собираетесь удалить ВСЕ метки калибровки!")
        print("Это действие необратимо.")
        print("Нажмите 'y' для подтверждения или любую другую клавишу для отмены.")
        
        # Ждем подтверждения
        confirmation_key = cv2.waitKey(0) & 0xFF
        if confirmation_key == ord('y'):
            data = clear_calibration_data('calibration_data.pkl')
            clf = None
            calibration_mode = True
            print("Калибровка очищена. Переход в режим CALIBRATION.")
        else:
            print("Очистка отменена.")

cap.release()
cv2.destroyAllWindows()