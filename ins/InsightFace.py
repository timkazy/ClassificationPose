import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis # type: ignore
import math
import warnings
from collections import deque
import hashlib

# Подавляем предупреждения
warnings.filterwarnings('ignore')

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
def create_calibration_file_if_not_exists(filename='raw_calibration_data.pkl'):
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
        return load_raw_calibration()

def load_raw_calibration(filename='raw_calibration_data.pkl'):
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
def clear_calibration_data(filename='raw_calibration_data.pkl'):
    """Очищает все метки в файле калибровки"""
    empty_data = {1: [], 2: [], 3: []}
    if os.path.exists(filename):
        with open(filename, 'wb') as f:
            pickle.dump(empty_data, f)
    
    # Также очищаем сырые данные
    # if os.path.exists('raw_calibration_data.pkl'):
    #     with open('raw_calibration_data.pkl', 'wb') as f:
    #         pickle.dump(empty_data, f)
    
    print(f"Все метки калибровки в {filename} были очищены.")
    return empty_data

# ================== Функция создания информационной панели ==================
def create_info_panel(people_info, calibration_mode, calibration_counts, fps=None, frame_size=None):
    """Создает второе окно с информацией о людях"""
    
    # Размеры панели - УВЕЛИЧЕНА ВЫСОТА
    panel_width = 800
    panel_height = 800  # Увеличено с 600 до 800 пикселей
    
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
    
    # ЛЕВАЯ КОЛОНКА: Калибровочные данные
    left_col_x = 25
    
    for class_id in [1, 2, 3]:
        class_name = class_labels.get(class_id, f"Class {class_id}")
        count = calibration_counts.get(class_id, 0)
        color = class_colors.get(class_id, (255, 255, 255))
        
        # Цветной квадрат
        cv2.rectangle(panel, (left_col_x, y_pos - 15), (left_col_x + 20, y_pos + 5), color, -1)
        cv2.putText(panel, f"{class_name}: {count}", (left_col_x + 30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    # Общее количество образцов (в левой колонке)
    cv2.putText(panel, f"Total: {total_calibration_samples}", (left_col_x + 30, y_pos), 
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
    
    # ПРАВАЯ КОЛОНКА: Текущие классы
    y_pos = 110
    # Отображаем общую статистику
    cv2.putText(panel, f"Total People: {total_people}", (panel_width // 2 + 20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 30
    
    cv2.putText(panel, "Current Classes:", (panel_width // 2 + 20, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 1)
    y_pos += 25
    
    right_col_x = panel_width // 2 + 25
    
    for class_id, class_name in class_labels.items():
        color = class_colors.get(class_id, (255, 255, 255))
        count = class_counts.get(class_name, 0)
        
        # Цветной квадрат в правой колонке
        cv2.rectangle(panel, (right_col_x, y_pos - 15), (right_col_x + 20, y_pos + 5), color, -1)
        cv2.putText(panel, f"{class_name}: {count}", (right_col_x + 30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    # Неизвестные (если есть)
    unknown_count = class_counts.get("unknown", 0)
    if unknown_count > 0:
        color = class_colors.get("unknown", (128, 128, 128))
        cv2.rectangle(panel, (right_col_x, y_pos - 15), (right_col_x + 20, y_pos + 5), color, -1)
        cv2.putText(panel, f"unknown: {unknown_count}", (right_col_x + 30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 25
    
    y_pos += 20
    
    # === Детальная информация по каждому человеку С ФОТО ===
    cv2.putText(panel, "People Details:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
    y_pos += 30
    
    # Сортируем ID для удобного отображения
    sorted_ids = sorted(people_info.keys())
    
    # Начальные позиции для таблицы
    start_y = y_pos
    photo_size = 60  # Увеличили размер фото
    row_height = 65  # Увеличили высоту строки
    col_width = 380  # Ширина колонки
    col_offset = 20
    
    # Заголовки таблицы
    cv2.putText(panel, "ID", (50, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Photo", (80, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Activity", (160, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Position", (280, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_pos += 30
    
    # Первая колонка
    first_column_end = panel_height - 100  # Конец первой колонки (оставляем место для инструкций)
    people_in_first_column = 0
    
    for person_id in sorted_ids:
        info = people_info[person_id]
        
        # Проверяем, есть ли место в текущей колонке
        if y_pos > first_column_end:
            # Переходим на вторую колонку
            y_pos = start_y + 30  # Начинаем с той же высоты
            col_offset = 400  # Смещение для второй колонки
        
        # 1. Отображаем ID
        cv2.putText(panel, f"{person_id}", (col_offset + 30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 2. Отображаем фото (если есть)
        if "best_photo" in info and info["best_photo"] is not None:
            try:
                # Уменьшаем фото для отображения
                photo = info["best_photo"].copy()
                if photo.shape[0] > photo_size or photo.shape[1] > photo_size:
                    # Сохраняем соотношение сторон
                    aspect = photo.shape[1] / photo.shape[0]
                    if aspect > 1:
                        new_width = photo_size
                        new_height = int(photo_size / aspect)
                    else:
                        new_height = photo_size
                        new_width = int(photo_size * aspect)
                    
                    photo = cv2.resize(photo, (new_width, new_height))
                
                # Вставляем фото в панель
                x_start = col_offset + 70
                y_start = y_pos - photo_size // 2
                
                # Проверяем границы
                y_start = max(start_y + 10, min(y_start, panel_height - photo_size - 50))
                x_end = min(x_start + photo.shape[1], col_offset + 150)
                y_end = min(y_start + photo.shape[0], panel_height - 50)
                
                if x_end > x_start and y_end > y_start:
                    # Обрезаем фото если нужно
                    photo_cropped = photo[:y_end-y_start, :x_end-x_start]
                    panel[y_start:y_end, x_start:x_end] = photo_cropped
                    
                    # Рамка вокруг фото
                    cv2.rectangle(panel, (x_start-1, y_start-1), (x_end+1, y_end+1), (255, 255, 255), 1)
            except Exception as e:
                print(f"Ошибка отображения фото ID {person_id}: {e}")
                # Показываем placeholder если фото не загружено
                cv2.putText(panel, "[No Photo]", (col_offset + 70, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            # Показываем placeholder если фото нет
            cv2.putText(panel, "[No Photo]", (col_offset + 70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 3. Отображаем активность (деятельность)
        if "prediction" in info:
            pred_label = info["prediction"]
            color = class_colors.get(1 if pred_label == "listening" else 2 if pred_label == "talking" else 3 if pred_label == "phone" else "unknown", (255, 255, 255))
            cv2.putText(panel, pred_label, (col_offset + 150, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 4. Отображаем позицию (координаты)
        if "center" in info:
            x, y = info["center"]
            cv2.putText(panel, f"({int(x)}, {int(y)})", (col_offset + 250, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_pos += row_height
        
        # Считаем количество людей в первой колонке
        if col_offset == 20:
            people_in_first_column += 1
    
    # Инструкции внизу
    y_pos = panel_height - 40
    instructions = "Press 'c': toggle mode | 'r': clear calibration | 'q': quit"
    cv2.putText(panel, instructions, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    # Информация о количестве людей
    if len(sorted_ids) > people_in_first_column:
        cv2.putText(panel, f"Showing {len(sorted_ids)} people", (panel_width - 200, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    return panel

# ================== ИНИЦИАЛИЗАЦИЯ InsightFace ==================
print("Инициализация InsightFace...")
def initialize_insightface(gpu_id=0):
    """Инициализация InsightFace для детектирования лиц и ключевых точек"""
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=gpu_id, det_size=(640, 640))
    return app

insight_face = initialize_insightface(gpu_id=0)
print("InsightFace успешно инициализирован!")

# Function to extract features from InsightFace landmarks
def extract_features_insightface(landmarks_5, image_shape):
    """Извлекает фичи из 5 ключевых точек InsightFace"""
    height, width = image_shape[:2]
    
    features = []
    
    # 1. Нормализованные координаты 5 точек
    for point in landmarks_5:
        features.append(point[0] / width)   # X нормализованный
        features.append(point[1] / height)  # Y нормализованный
    
    # 2. Углы поворота головы (простой метод)
    left_eye = np.array(landmarks_5[0])
    right_eye = np.array(landmarks_5[1])
    nose = np.array(landmarks_5[2])
    left_mouth = np.array(landmarks_5[3])
    right_mouth = np.array(landmarks_5[4])
    
    # Yaw (поворот влево-вправо) по асимметрии глаз
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_offset_x = nose[0] - eye_center_x
    eye_distance = abs(right_eye[0] - left_eye[0])
    
    if eye_distance > 0:
        yaw_ratio = nose_offset_x / (eye_distance / 2)
        yaw = np.clip(yaw_ratio * 45, -45, 45)  # ±45 градусов
    else:
        yaw = 0
    
    # Pitch (наклон вверх-вниз) по вертикальному положению носа
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    face_height = abs(mouth_center_y - eye_center_y)
    
    if face_height > 0:
        nose_offset_y = nose[1] - eye_center_y
        # Нормализация: 0.3 = нос на нормальной высоте
        pitch_ratio = nose_offset_y / face_height
        pitch = np.clip((pitch_ratio - 0.3) * 60, -30, 30)
    else:
        pitch = 0
    
    # Roll (наклон головы) по углу линии глаз
    eye_delta_y = right_eye[1] - left_eye[1]
    eye_delta_x = right_eye[0] - left_eye[0]
    
    if eye_delta_x != 0:
        roll_rad = math.atan2(eye_delta_y, eye_delta_x)
        roll = math.degrees(roll_rad)
    else:
        roll = 0
    
    # Добавляем углы в фичи
    features.extend([yaw / 45.0, pitch / 30.0, roll / 45.0])  # Нормализованные
    
    # 3. Расстояния между ключевыми точками
    distances = []
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (2, 3), (2, 4)]
    for i, j in pairs:
        dist = np.linalg.norm(np.array(landmarks_5[i]) - np.array(landmarks_5[j]))
        distances.append(dist)
    
    # Нормализуем расстояния относительно расстояния между глазами
    if distances[0] > 0:
        normalized_distances = [d / distances[0] for d in distances]
        features.extend(normalized_distances)
    else:
        features.extend([0.0] * 6)
    
    return np.array(features)

# Function to get face bounding box and center
def get_face_bbox_and_center(landmarks_5, image_shape, padding=0.2):
    """Получает bounding box и центр лица из 5 ключевых точек"""
    height, width = image_shape[:2]
    
    # Преобразуем 5 точек в координаты для вычисления bbox
    x_coords = [point[0] * width for point in landmarks_5]
    y_coords = [point[1] * height for point in landmarks_5]
    
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Добавляем padding
    width_padding = int((max_x - min_x) * padding)
    height_padding = int((max_y - min_y) * padding)
    
    min_x = max(0, min_x - width_padding)
    min_y = max(0, min_y - height_padding)
    max_x = min(width - 1, max_x + width_padding)
    max_y = min(height - 1, max_y + height_padding)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    return min_x, min_y, max_x, max_y, (center_x, center_y)

# Класс для хранения фото лица
class FacePhotoManager:
    def __init__(self, max_photos_per_person=10, photo_size=(100, 100)):
        self.photos = {}  # person_id -> список фото
        self.best_photos = {}  # person_id -> лучшее фото
        self.max_photos = max_photos_per_person
        self.photo_size = photo_size
        
    def add_photo(self, person_id, face_roi, face_quality=1.0):
        """Добавляет фото лица для конкретного ID"""
        if person_id not in self.photos:
            self.photos[person_id] = deque(maxlen=self.max_photos)
        
        # Сохраняем фото с качеством
        self.photos[person_id].append({
            'photo': face_roi.copy(),
            'quality': face_quality,
            'timestamp': datetime.now()
        })
        
        # Обновляем лучшее фото
        self.update_best_photo(person_id)
    
    def update_best_photo(self, person_id):
        """Обновляет лучшее фото для человека"""
        if person_id in self.photos and self.photos[person_id]:
            # Находим фото с максимальным качеством
            best_entry = max(self.photos[person_id], key=lambda x: x['quality'])
            self.best_photos[person_id] = best_entry['photo']
    
    def get_best_photo(self, person_id):
        """Возвращает лучшее фото для человека"""
        return self.best_photos.get(person_id, None)
    
    def get_photo_quality(self, face_roi):
        """Оценивает качество фото лица"""
        # Простая оценка качества: размер и контрастность
        if face_roi is None or face_roi.size == 0:
            return 0
        
        height, width = face_roi.shape[:2]
        
        # 1. Оценка по размеру (больше = лучше)
        size_score = min(1.0, (height * width) / (self.photo_size[0] * self.photo_size[1]))
        
        # 2. Оценка по контрастности
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray) / 255.0  # Нормализованная контрастность
        
        # 3. Оценка по резкости (лапласиан)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian / 100.0)
        
        # Общий score
        quality = 0.4 * size_score + 0.3 * contrast + 0.3 * sharpness
        
        return min(1.0, max(0.0, quality))

# Простой трекер для присвоения ID лицам
class SimpleFaceTracker:
    def __init__(self, max_distance=100):
        self.tracked_faces = {}  # id: (center, features)
        self.max_distance = max_distance
        self.max_id = 50
        
    def _get_available_id(self):
        """Возвращает первый доступный ID начиная с 0"""
        # Ищем все занятые ID
        used_ids = set(self.tracked_faces.keys())
        
        # Ищем первый свободный ID от 0 до max_id-1
        for i in range(self.max_id):
            if i not in used_ids:
                return i
                
        # Если все ID заняты, возвращаем самый старый (первый в словаре)
        # Это нужно для ограниченного количества треков
        return list(self.tracked_faces.keys())[0]
    
    def update(self, current_faces):
        """Обновляет трекер с новыми лицами"""
        # current_faces: список словарей с ключами 'center', 'features', 'landmarks'
        
        if not current_faces:
            return []
        
        # Если нет отслеживаемых лиц, присваиваем ID начиная с 0
        if not self.tracked_faces:
            for i, face in enumerate(current_faces):
                face_id = i % self.max_id  # Начинаем с 0
                self.tracked_faces[face_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox']
                }
                face['id'] = face_id
            return current_faces
        
        # Ищем соответствия между текущими и отслеживаемыми лицами
        matched_ids = []
        updated_faces = []
        
        for face in current_faces:
            min_dist = float('inf')
            matched_id = None
            
            for face_id, tracked_face in self.tracked_faces.items():
                if face_id in matched_ids:
                    continue
                
                dist = np.linalg.norm(np.array(face['center']) - np.array(tracked_face['center']))
                
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched_id = face_id
            
            if matched_id is not None:
                # Обновляем существующий трек
                face['id'] = matched_id
                self.tracked_faces[matched_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox']
                }
                matched_ids.append(matched_id)
            else:
                # Создаем новый трек со свободным ID
                face_id = self._get_available_id()
                face['id'] = face_id
                self.tracked_faces[face_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox']
                }
            
            updated_faces.append(face)
        
        # Удаляем старые треки, которые больше не активны
        active_ids = [face['id'] for face in updated_faces]
        to_remove = [face_id for face_id in self.tracked_faces if face_id not in active_ids]
        for face_id in to_remove:
            del self.tracked_faces[face_id]
        
        return updated_faces
    
# ================== ОСНОВНАЯ ПРОГРАММА ==================
# === Загружаем данные при старте ===
print("\nЗагрузка данных и модели...")
clf, data = load_and_train_from_file('trained_data.pkl')
calibration_mode = clf is None

if clf is not None:
    calibration_mode = False
    print("Запуск в режиме AUTO (обученная модель загружена)")
else:
    print("Запуск в режиме КАЛИБРОВКИ")

# Остальные переменные
selected_tid = None
tracker = SimpleFaceTracker(max_distance=100)
photo_manager = FacePhotoManager(max_photos_per_person=5, photo_size=(100, 100))
people_info = {}  # Словарь для хранения информации о людях

# Для расчета FPS
fps_start_time = cv2.getTickCount()
fps_frame_count = 0
current_fps = 0

# === НОВЫЕ ПЕРЕМЕННЫЕ ДЛЯ ПРОПУСКА КАДРОВ ===
frame_skip_counter = 0
frame_skip_interval = 6  # Обрабатывать каждый N-й кадр
process_this_frame = True  # Флаг для обработки текущего кадра

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
print(f" - Настройка пропуска кадров: сейчас каждый {frame_skip_interval}-й кадр обрабатывается")

# Создаем окна заранее
cv2.namedWindow("Head Pose Classifier", cv2.WINDOW_NORMAL)
cv2.namedWindow("People Information", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Head Pose Classifier", 640, 480)
cv2.resizeWindow("People Information", 800, 800)

print("\nНачало обработки видео...")

while True:
    ret, frame = cap.read()
    if not ret:
        if video_mode == 'video':
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            break
    
    # Зеркалим кадр для камеры
    if video_mode == 'camera':
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
    
    # === ПРОПУСК КАДРОВ: решаем, обрабатывать ли этот кадр ===
    frame_skip_counter += 1
    process_this_frame = (frame_skip_counter % frame_skip_interval == 0)
    
    # ================== ДЕТЕКТИРОВАНИЕ ЛИЦ ==================
    # Используем только InsightFace для детектирования лиц
    detected_faces = []
    
    if process_this_frame:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = insight_face.get(rgb_frame)
        
        for i, face in enumerate(faces):
            landmarks_5 = face.kps.tolist() if face.kps is not None else None
            
            if landmarks_5 and len(landmarks_5) == 5:
                # Bounding box от InsightFace
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Добавляем padding для лучшего отображения лица
                padding = 20
                h, w = frame.shape[:2]
                roi_x1 = max(0, x1 - padding)
                roi_y1 = max(0, y1 - padding)
                roi_x2 = min(w, x2 + padding)
                roi_y2 = min(h, y2 + padding)
                
                # Извлекаем ROI лица для фото
                face_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Нормализованные координаты для извлечения фич
                global_landmarks = []
                for point in landmarks_5:
                    # Сохраняем нормализованные координаты
                    global_x = point[0] / frame.shape[1]
                    global_y = point[1] / frame.shape[0]
                    global_landmarks.append([global_x, global_y])
                
                # Извлекаем фичи
                features = extract_features_insightface(global_landmarks, frame.shape)
                
                # Центр
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detected_faces.append({
                    'landmarks': global_landmarks,
                    'features': features,
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'face_roi': face_roi,  # Сохраняем ROI для фото
                    'index': i
                })
        
        # ================== ТРЕКИНГ ==================
        tracked_faces = tracker.update(detected_faces)
    else:
        # Если не обрабатываем этот кадр, используем предыдущие данные для отображения
        # но обновляем координаты на основе простой экстраполяции
        if 'last_tracked_faces' in locals():
            tracked_faces = last_tracked_faces
        else:
            tracked_faces = []
    
    # Сохраняем текущие лица для следующего кадра
    if process_this_frame:
        last_tracked_faces = tracked_faces.copy()
    
    # ================== ОТОБРАЖЕНИЕ ==================
    people_info = {}
    face_data = {}  # Для калибровки
    
    for face in tracked_faces:
        face_id = face['id']
        center = face['center']
        features = face['features']
        x1, y1, x2, y2 = face['bbox']
        landmarks = face['landmarks']
        face_roi = face.get('face_roi', None)
        
        # Сохраняем фото лица (если ROI доступен и обрабатываем этот кадр)
        if process_this_frame and face_roi is not None and face_roi.size > 0:
            quality = photo_manager.get_photo_quality(face_roi)
            photo_manager.add_photo(face_id, face_roi, quality)
        
        # Рисуем bounding box InsightFace (желтый)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Рисуем ключевые точки InsightFace (зеленые) - только если обрабатываем кадр
        if process_this_frame:
            for point in landmarks:
                px = int(point[0] * frame.shape[1])
                py = int(point[1] * frame.shape[0])
                cv2.circle(display_frame, (px, py), 3, (0, 255, 0), -1)
        
        # Предсказание класса
        prediction_label = "unknown"
        if not calibration_mode and clf is not None:
            try:
                pred = clf.predict([features])[0]
                prediction_label = class_labels.get(pred, "unknown")
                color = class_colors.get(pred, class_colors["unknown"])
                
                # Отображаем предсказание
                cv2.putText(display_frame, prediction_label,
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f"Ошибка предсказания: {e}")
        
        # Отображаем ID
        cv2.putText(display_frame, f"ID:{face_id}",
                   (x1, y1 - 35),
                   cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
        
        # Сохраняем информацию для панели (включая лучшее фото)
        best_photo = photo_manager.get_best_photo(face_id)
        people_info[face_id] = {
            'center': center,
            'prediction': prediction_label,
            'bbox': (x1, y1, x2, y2),
            'features': features,
            'best_photo': best_photo
        }
        
        # Сохраняем для калибровки
        face_data[face_id] = (features, (x1, y1))
    
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
    
    # Индикатор пропуска кадров
    skip_indicator = f"Skip: {frame_skip_interval-1}/{frame_skip_interval}"
    cv2.putText(display_frame, skip_indicator,
                (display_frame.shape[1] - 150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
    
    # Количество людей
    cv2.putText(display_frame, f"People: {len(tracked_faces)}",
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
    
    # НОВЫЕ КЛАВИШИ ДЛЯ УПРАВЛЕНИЯ ПРОПУСКОМ КАДРОВ
    if key == ord('+') or key == ord('='):
        frame_skip_interval = min(10, frame_skip_interval + 1)
        print(f"Увеличен интервал пропуска кадров: теперь {frame_skip_interval}")
    
    if key == ord('-') or key == ord('_'):
        frame_skip_interval = max(1, frame_skip_interval - 1)
        print(f"Уменьшен интервал пропуска кадров: теперь {frame_skip_interval}")
    
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
            for label, samples in data.items():
                X.extend(samples)
                y.extend([label] * len(samples))
            if len(X) > 10:
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                calibration_mode = False
                print(f"Модель обучена на {len(X)} образцах!")
        
        if key == ord('v'):
            save_raw_data(data, 'raw_calibration_data.pkl')
            # with open('calibration_data.pkl', 'wb') as f:
            #     pickle.dump(data, f)
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