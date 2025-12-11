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

# Для идентификации
import glob
from sklearn.metrics.pairwise import cosine_similarity

class FaceDatabase:
    def __init__(self, database_path='faces_database', similarity_threshold=0.6):
        self.database_path = database_path
        self.similarity_threshold = similarity_threshold
        self.embeddings = []  # Список эмбеддингов
        self.names = []       # Список имен (ФИО)
        self.loaded = False
        
    def load_database(self, insight_app):
        """Загружает все лица из базы и вычисляет эмбеддинги"""
        print(f"Загрузка базы лиц из {self.database_path}...")
        
        if not os.path.exists(self.database_path):
            print(f"Папка {self.database_path} не найдена! Создаем...")
            os.makedirs(self.database_path, exist_ok=True)
            self.loaded = True
            return
        
        # Поддерживаемые форматы изображений
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(self.database_path, ext)))
        
        print(f"Найдено {len(image_paths)} изображений в базе")
        
        if len(image_paths) == 0:
            print("База лиц пуста! Добавьте фото в папку faces_database/")
            self.loaded = True
            return
        
        self.embeddings = []
        self.names = []
        
        for img_path in image_paths:
            try:
                # Загружаем изображение
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Не удалось загрузить: {img_path}")
                    continue
                
                # Конвертируем в RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Детектируем лица на фото
                faces = insight_app.get(rgb_img)
                
                if len(faces) == 0:
                    print(f"Лица не найдены на: {img_path}")
                    continue
                
                # Берем первое (и, надеемся, единственное) лицо на фото
                face = faces[0]
                
                # Получаем эмбеддинг (512-мерный вектор)
                embedding = face.normed_embedding
                
                # Получаем имя из названия файла (без расширения)
                name = os.path.splitext(os.path.basename(img_path))[0]
                
                self.embeddings.append(embedding)
                self.names.append(name)
                
                print(f"  Загружено: {name}")
                
            except Exception as e:
                print(f"Ошибка при обработке {img_path}: {e}")
        
        if len(self.embeddings) > 0:
            # Преобразуем в numpy array для быстрых вычислений
            self.embeddings = np.array(self.embeddings)
            print(f"База лиц загружена: {len(self.embeddings)} записей")
        else:
            print("База лиц пуста после обработки")
        
        self.loaded = True
    
    # В методе identify_face класса FaceDatabase (около строки 116):
    def identify_face(self, embedding, face_roi=None, face_storage=None):
        """Идентифицирует лицо по эмбеддингу"""
        if not self.loaded or embedding is None:  # ИЗМЕНЕНО: проверяем self.loaded вместо self.face_database
            return None, 0.0
        
        if len(self.embeddings) == 0:
            print("База лиц пуста!")
            return None, 0.0
        
        try:
            # Преобразуем embedding в правильную форму
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Вычисляем косинусное сходство со всеми эмбеддингами в базе
            similarities = cosine_similarity(embedding, self.embeddings)[0]
            
            # Находим максимальное сходство
            max_index = np.argmax(similarities)
            max_similarity = similarities[max_index]
                        
            # Если сходство выше порога - возвращаем имя
            if max_similarity >= self.similarity_threshold:
                return self.names[max_index], float(max_similarity)
            else:
                return None, float(max_similarity)
            
        except Exception as e:
            print(f"Ошибка при идентификации: {e}")
            return None, 0.0
    
    def add_face(self, embedding, name, insight_app=None):
        """Добавляет новое лицо в базу (и сохраняет фото)"""
        if embedding is None:
            return False
        
        # Добавляем в память
        if len(self.embeddings) == 0:
            self.embeddings = np.array([embedding])
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        self.names.append(name)
        
        # Сохраняем фото в базу (если нужно)
        # Для этого нужен кадр с лицом, который можно передать отдельно
        
        print(f"Лицо добавлено в базу: {name}")
        return True
    
    def get_all_names(self):
        """Возвращает список всех имен в базе"""
        return self.names.copy()
    
    def save_database(self):
        """Сохраняет базу в файл для быстрой загрузки"""
        if len(self.embeddings) == 0:
            return
        
        database_file = os.path.join(self.database_path, 'face_database.npz')
        np.savez_compressed(
            database_file,
            embeddings=self.embeddings,
            names=self.names
        )
        print(f"База лиц сохранена в {database_file}")
    
    def load_from_file(self):
        """Загружает базу из файла (если есть)"""
        database_file = os.path.join(self.database_path, 'face_database.npz')
        
        if os.path.exists(database_file):
            try:
                data = np.load(database_file, allow_pickle=True)
                self.embeddings = data['embeddings']
                self.names = data['names'].tolist()
                self.loaded = True
                print(f"База лиц загружена из файла: {len(self.names)} записей")
                return True
            except Exception as e:
                print(f"Ошибка загрузки базы из файла: {e}")
        
        return False

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
    """Создает второе окно с информацией о людях С ФОТО И ИМЕНАМИ"""
    
    # Размеры панели
    panel_width = 800
    panel_height = 800
    
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
    # ФИКС: проверяем что name не None
    identified_count = sum(1 for info in people_info.values() if info.get('name') is not None)
    class_counts = {"listening": 0, "talking": 0, "phone": 0, "unknown": 0}
    
    for info in people_info.values():
        if "prediction" in info:
            pred_label = info["prediction"]
            if pred_label in class_counts:
                class_counts[pred_label] += 1
            else:
                class_counts["unknown"] += 1
    
    # ПРАВАЯ КОЛОНКА: Текущие классы и идентификация
    right_col_x = panel_width // 2 + 20
    y_pos_right = 110
    
    # Отображаем общую статистику
    cv2.putText(panel, f"Total People: {total_people}", (right_col_x, y_pos_right), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos_right += 25
    
    cv2.putText(panel, f"Identified: {identified_count}", (right_col_x, y_pos_right), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100) if identified_count > 0 else (200, 200, 200), 1)
    y_pos_right += 25
    
    cv2.putText(panel, "Current Classes:", (right_col_x, y_pos_right), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 1)
    y_pos_right += 25
    
    for class_id, class_name in class_labels.items():
        color = class_colors.get(class_id, (255, 255, 255))
        count = class_counts.get(class_name, 0)
        
        # Цветной квадрат в правой колонке
        cv2.rectangle(panel, (right_col_x, y_pos_right - 15), (right_col_x + 20, y_pos_right + 5), color, -1)
        cv2.putText(panel, f"{class_name}: {count}", (right_col_x + 30, y_pos_right), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos_right += 25
    
    # Неизвестные (если есть)
    unknown_count = class_counts.get("unknown", 0)
    if unknown_count > 0:
        color = class_colors.get("unknown", (128, 128, 128))
        cv2.rectangle(panel, (right_col_x, y_pos_right - 15), (right_col_x + 20, y_pos_right + 5), color, -1)
        cv2.putText(panel, f"unknown: {unknown_count}", (right_col_x + 30, y_pos_right), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos_right += 25
    
    y_pos_right += 20
    
    # === Детальная информация по каждому человеку С ФОТО И ИМЕНАМИ ===
    cv2.putText(panel, "People Details:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 1)
    y_pos += 30
    
    # Сортируем ID для удобного отображения
    sorted_ids = sorted(people_info.keys())
    
    # Начальные позиции для таблицы
    start_y = y_pos
    photo_size = 60
    row_height = 65
    col_width = 380
    col_offset = 20
    
    # Заголовки таблицы (ДОБАВЛЕНО ИМЯ)
    cv2.putText(panel, "ID", (col_offset + 20, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Photo", (col_offset + 60, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Name", (col_offset + 140, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(panel, "Activity", (col_offset + 220, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_pos += 30
    
    # Первая колонка
    first_column_end = panel_height - 100
    people_in_first_column = 0
    
    for person_id in sorted_ids:
        info = people_info[person_id]
        
        # Проверяем, есть ли место в текущей колонке
        if y_pos > first_column_end:
            # Переходим на вторую колонку
            y_pos = start_y + 30
            col_offset = 400
        
        # 1. Отображаем ID
        cv2.putText(panel, f"{person_id}", (col_offset + 20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 2. Отображаем фото (если есть)
        if "best_photo" in info and info["best_photo"] is not None:
            try:
                # Пробуем получить фото из хранилища для идентифицированных
                display_photo = None
                name = info.get('name', None)
                
                if name and face_storage and face_storage.has_photo(name):
                    # Берем фото из хранилища
                    display_photo = face_storage.get_face_photo(name)
                else:
                    # Иначе берем лучшее фото из трекера
                    display_photo = info["best_photo"]
                
                if display_photo is not None:
                    # Уменьшаем фото для отображения
                    photo = display_photo.copy()
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
                    x_start = col_offset + 60
                    y_start = y_pos - photo_size // 2
                    
                    # Проверяем границы
                    y_start = max(start_y + 10, min(y_start, panel_height - photo_size - 50))
                    x_end = min(x_start + photo.shape[1], col_offset + 140)
                    y_end = min(y_start + photo.shape[0], panel_height - 50)
                    
                    if x_end > x_start and y_end > y_start:
                        # Обрезаем фото если нужно
                        photo_cropped = photo[:y_end-y_start, :x_end-x_start]
                        panel[y_start:y_end, x_start:x_end] = photo_cropped
                        
                        # Рамка вокруг фото
                        frame_color = (0, 255, 0) if name else (200, 200, 200)  # Зеленая для идентифицированных
                        cv2.rectangle(panel, (x_start-1, y_start-1), (x_end+1, y_end+1), frame_color, 1)
                else:
                    raise ValueError("Photo is None")
                    
            except Exception as e:
                print(f"Ошибка отображения фото ID {person_id}: {e}")
                # Показываем placeholder если фото не загружено
                placeholder_text = "[No Photo]" if not name else "[Saved Photo]"
                placeholder_color = (150, 150, 150) if not name else (100, 200, 100)
                
                cv2.putText(panel, placeholder_text, (col_offset + 60, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, placeholder_color, 1)
        else:
            # Показываем placeholder если фото нет
            name = info.get('name', None)
            placeholder_text = "[No Photo]" if not name else "[No Photo]"
            placeholder_color = (150, 150, 150)
            
            cv2.putText(panel, placeholder_text, (col_offset + 60, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, placeholder_color, 1)
        
        # 3. Отображаем ИМЯ (НОВОЕ!) - ФИКС: проверяем на None
        name = info.get('name', None)
        similarity = info.get('similarity', 0.0)
        
        if name is not None:
            # Укорачиваем длинные имена - ФИКС: проверяем что name это строка
            if isinstance(name, str):
                display_name = name[:10] + "..." if len(name) > 10 else name
            else:
                display_name = str(name)[:10] + "..." if len(str(name)) > 10 else str(name)
                
            name_color = (100, 255, 100)  # Зеленый для идентифицированных
            
            # Добавляем сходство если оно высокое
            name_text = display_name
            if similarity > 0.7:
                name_text = f"{display_name}({similarity:.1f})"
            
            cv2.putText(panel, name_text, (col_offset + 140, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 1)
        else:
            # Для неидентифицированных
            unknown_text = "Unknown"
            if similarity > 0:  # Если была попытка идентификации
                unknown_text = f"Unknown({similarity:.1f})"
            
            cv2.putText(panel, unknown_text, (col_offset + 140, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 4. Отображаем активность (деятельность)
        if "prediction" in info:
            pred_label = info["prediction"]
            color = class_colors.get(
                1 if pred_label == "listening" else 
                2 if pred_label == "talking" else 
                3 if pred_label == "phone" else "unknown", 
                (255, 255, 255)
            )
            cv2.putText(panel, pred_label, (col_offset + 220, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # 5. Отображаем позицию (координаты) - если есть место
        if "center" in info and col_offset == 20:  # Только в первой колонке
            x, y = info["center"]
            cv2.putText(panel, f"({int(x)}, {int(y)})", (col_offset + 300, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_pos += row_height
        
        # Считаем количество людей в первой колонке
        if col_offset == 20:
            people_in_first_column += 1
    
    # Инструкции внизу (ОБНОВЛЕННЫЕ)
    y_pos = panel_height - 40
    instructions = "Press 'c': toggle mode | 'r': clear calibration | 'a': add to DB | 'd': show DB | 'q': quit"
    cv2.putText(panel, instructions, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
    # Информация о количестве людей и идентификации
    if len(sorted_ids) > 0:
        info_text = f"Showing {len(sorted_ids)} people"
        if identified_count > 0:
            info_text += f" ({identified_count} identified)"
        
        cv2.putText(panel, info_text, (panel_width - 250, y_pos), 
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

# Добавьте этот класс после FacePhotoManager

class IdentifiedFaceStorage:
    """Хранит фото идентифицированных людей"""
    def __init__(self, storage_path='identified_faces'):
        self.storage_path = storage_path
        self.face_photos = {}  # name -> photo
        
        # Создаем папку для хранения
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)
        
        # Загружаем уже сохраненные фото
        self.load_saved_photos()
    
    def load_saved_photos(self):
        """Загружает сохраненные фото из файлов"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        for ext in image_extensions:
            for img_path in glob.glob(os.path.join(self.storage_path, ext)):
                try:
                    name = os.path.splitext(os.path.basename(img_path))[0]
                    photo = cv2.imread(img_path)
                    if photo is not None:
                        self.face_photos[name] = photo
                        print(f"Загружено фото для: {name}")
                except Exception as e:
                    print(f"Ошибка загрузки фото {img_path}: {e}")
    
    def save_face_photo(self, name, face_photo):
        """Сохраняет фото лица в файл и в память"""
        if face_photo is None or face_photo.size == 0:
            return False
        
        try:
            # Убедимся, что имя безопасно для файловой системы
            safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)
            safe_name = safe_name.strip()
            
            if not safe_name:
                return False
            
            # Сохраняем фото
            file_path = os.path.join(self.storage_path, f"{safe_name}.jpg")
            cv2.imwrite(file_path, face_photo)
            
            # Сохраняем в память
            self.face_photos[name] = face_photo.copy()
            
            print(f"Фото сохранено для: {name}")
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения фото для {name}: {e}")
            return False
    
    def get_face_photo(self, name):
        """Возвращает сохраненное фото по имени"""
        return self.face_photos.get(name, None)
    
    def has_photo(self, name):
        """Проверяет, есть ли сохраненное фото для имени"""
        return name in self.face_photos

# Простой трекер для присвоения ID лицам
class SimpleFaceTracker:
    def __init__(self, max_distance=100):
        self.tracked_faces = {}  # id: (center, features, name, embedding)
        self.max_distance = max_distance
        self.max_id = 50
        self.face_database = None  # Ссылка на базу лиц
        self.identification_cache = {}  # Кэш идентификаций: embedding_hash -> name
        self.embedding_to_id = {}  # embedding_hash -> face_id
        self.id_to_embedding = {}  # face_id -> embedding_hash
        self.id_to_permanent_name = {}  # face_id -> постоянное имя
        self.identified_ids = set()  # ID которые уже были идентифицированы
        
    def set_face_database(self, database):
        """Устанавливает базу лиц для идентификации"""
        self.face_database = database

    def _get_available_id(self):
        """Возвращает первый доступный ID начиная с 0"""
        # Ищем все занятые ID
        used_ids = set(self.tracked_faces.keys())
        
        # Ищем первый свободный ID от 0 до max_id-1
        for i in range(self.max_id):
            if i not in used_ids:
                return i
        
        # Если все ID заняты, возвращаем самый старый (первый в словаре)
        # Но сначала убедимся что словарь не пустой
        if self.tracked_faces:
            return list(self.tracked_faces.keys())[0]
        else:
            return 0
    
    def _get_embedding_hash(self, embedding):
        """Создает хэш эмбеддинга для кэширования"""
        # ИСПРАВЛЕНИЕ: Правильная проверка numpy массива
        if embedding is None or len(embedding) == 0:
            return None
        
        # Преобразуем embedding в байты для хэширования
        try:
            # Берем первые 16 значений для хэша
            if isinstance(embedding, np.ndarray):
                return hashlib.md5(embedding[:16].tobytes()).hexdigest()
            else:
                # Если это список или другой тип
                embedding_array = np.array(embedding)
                return hashlib.md5(embedding_array[:16].tobytes()).hexdigest()
        except Exception as e:
            print(f"Ошибка создания хэша эмбеддинга: {e}")
            return None

    def identify_face(self, embedding, face_roi=None, face_storage=None):
        """Идентифицирует лицо с использованием базы и кэша"""
        if self.face_database is None or embedding is None:  # Эта строка уже правильная
            return None, 0.0
        
        # Проверяем что embedding не пустой
        if len(embedding) == 0:
            return None, 0.0
        
        embedding_hash = self._get_embedding_hash(embedding)
        
        # ПРОВЕРЯЕМ: Есть ли уже ID для этого эмбеддинга
        if embedding_hash and embedding_hash in self.embedding_to_id:
            face_id = self.embedding_to_id[embedding_hash]
            # НОВОЕ: Если ID уже был идентифицирован, возвращаем сохраненное имя
            if face_id in self.id_to_permanent_name:
                return self.id_to_permanent_name[face_id], 1.0
        
        # Проверяем кэш
        if embedding_hash and embedding_hash in self.identification_cache:
            cached_result = self.identification_cache[embedding_hash]
            if cached_result[0] is not None:  # Если в кэше есть имя
                return cached_result
        
        # Идентифицируем через базу
        name, similarity = self.face_database.identify_face(embedding)
    
        # НОВОЕ: Сохраняем фото если лицо идентифицировано
        if name and similarity >= 0.6 and face_roi is not None and face_storage is not None:
            # Сохраняем фото в хранилище
            if not face_storage.has_photo(name):
                face_storage.save_face_photo(name, face_roi)
        
        # НОВОЕ: Если идентификация успешна, сохраняем как постоянное имя
        if name and similarity >= 0.6:  # Порог сходства
            if embedding_hash and embedding_hash in self.embedding_to_id:
                face_id = self.embedding_to_id[embedding_hash]
                self.id_to_permanent_name[face_id] = name
                self.identified_ids.add(face_id)
        
        # Сохраняем в кэш (даже если не идентифицировали)
        if embedding_hash:
            self.identification_cache[embedding_hash] = (name, similarity)
            
            # Ограничиваем размер кэша
            if len(self.identification_cache) > 100:
                # Удаляем самый старый элемент
                oldest_key = next(iter(self.identification_cache))
                del self.identification_cache[oldest_key]
        
        return name, similarity
    
    def update(self, current_faces):
        """Обновляет трекер с новыми лицами"""
        if not current_faces:
            return []
        
        # Если нет отслеживаемых лиц, присваиваем ID на основе эмбеддинга
        if not self.tracked_faces:
            for face in current_faces:
                embedding = face.get('embedding', None)
                # ИСПРАВЛЕНИЕ: Правильная проверка numpy массива
                embedding_hash = None
                if embedding is not None and len(embedding) > 0:
                    embedding_hash = self._get_embedding_hash(embedding)
                
                # Пытаемся найти существующий ID для этого эмбеддинга
                face_id = None
                if embedding_hash and embedding_hash in self.embedding_to_id:
                    face_id = self.embedding_to_id[embedding_hash]
                else:
                    # Создаем новый ID
                    face_id = self._get_available_id()
                    # Сохраняем связь эмбеддинг -> ID
                    if embedding_hash:
                        self.embedding_to_id[embedding_hash] = face_id
                        self.id_to_embedding[face_id] = embedding_hash
                
                # НОВОЕ: Используем сохраненное имя если оно есть
                saved_name = self.id_to_permanent_name.get(face_id, None)
                current_name = face.get('name', None)
                
                # Приоритет: сохраненное имя > текущее имя
                final_name = saved_name if saved_name is not None else current_name
                
                self.tracked_faces[face_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox'],
                    'name': final_name,  # Используем финальное имя
                    'embedding': embedding,
                    'embedding_hash': embedding_hash,  # Сохраняем хэш
                    'similarity': face.get('similarity', 0.0)
                }
                face['id'] = face_id
                face['name'] = final_name  # Обновляем имя в текущем лице
            return current_faces
        
        # Ищем соответствия на основе эмбеддинга в первую очередь
        matched_ids = []
        updated_faces = []
        
        for face in current_faces:
            embedding = face.get('embedding', None)
            # ИСПРАВЛЕНИЕ: Правильная проверка numpy массива
            embedding_hash = None
            if embedding is not None and len(embedding) > 0:
                embedding_hash = self._get_embedding_hash(embedding)
            
            # ШАГ 1: Пробуем сопоставить по эмбеддингу
            matched_id = None
            
            if embedding_hash:
                # Пытаемся найти ID по эмбеддингу
                if embedding_hash in self.embedding_to_id:
                    matched_id = self.embedding_to_id[embedding_hash]
            
            # ШАГ 2: Если не нашли по эмбеддингу, ищем по расстоянию
            if matched_id is None:
                min_dist = float('inf')
                
                for face_id, tracked_face in self.tracked_faces.items():
                    if face_id in matched_ids:
                        continue
                    
                    dist = np.linalg.norm(np.array(face['center']) - np.array(tracked_face['center']))
                    
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        matched_id = face_id
            
            # ШАГ 3: Если нашли соответствие
            if matched_id is not None:
                # Обновляем существующий трек
                face['id'] = matched_id
                
                # Сохраняем связь эмбеддинг -> ID (если есть новый эмбеддинг)
                if embedding_hash and embedding_hash not in self.embedding_to_id:
                    self.embedding_to_id[embedding_hash] = matched_id
                    self.id_to_embedding[matched_id] = embedding_hash
                
                # Сохраняем имя и эмбеддинг из предыдущего трека (если были)
                old_name = self.tracked_faces[matched_id].get('name', None)
                old_embedding = self.tracked_faces[matched_id].get('embedding', None)
                old_embedding_hash = self.tracked_faces[matched_id].get('embedding_hash', None)
                
                # НОВОЕ: Приоритет имен: сохраненное > старое > новое
                saved_name = self.id_to_permanent_name.get(matched_id, None)
                current_name_from_face = face.get('name', None)
                
                # Определяем финальное имя
                if saved_name is not None:
                    final_name = saved_name
                elif old_name is not None:
                    final_name = old_name
                else:
                    final_name = current_name_from_face
                
                # Используем эмбеддинг из предыдущего трека, если он есть и новый отсутствует
                # ИСПРАВЛЕНИЕ: Правильная проверка numpy массива
                if old_embedding is not None and (embedding is None or len(embedding) == 0):
                    new_embedding = old_embedding
                    new_embedding_hash = old_embedding_hash
                else:
                    new_embedding = embedding
                    new_embedding_hash = embedding_hash
                
                self.tracked_faces[matched_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox'],
                    'name': final_name,
                    'embedding': new_embedding,
                    'embedding_hash': new_embedding_hash,
                    'similarity': face.get('similarity', face.get('similarity', 0.0))
                }
                face['name'] = final_name  # Обновляем имя в текущем лице
                matched_ids.append(matched_id)
            else:
                # Создаем новый трек на основе эмбеддинга
                face_id = None
                
                if embedding_hash and embedding_hash in self.embedding_to_id:
                    # Уже есть ID для этого эмбеддинга
                    face_id = self.embedding_to_id[embedding_hash]
                else:
                    # Создаем новый ID
                    face_id = self._get_available_id()
                    # Сохраняем связь эмбеддинг -> ID
                    if embedding_hash:
                        self.embedding_to_id[embedding_hash] = face_id
                        self.id_to_embedding[face_id] = embedding_hash
                
                # НОВОЕ: Используем сохраненное имя если оно есть
                saved_name = self.id_to_permanent_name.get(face_id, None)
                current_name = face.get('name', None)
                final_name = saved_name if saved_name is not None else current_name
                
                face['id'] = face_id
                self.tracked_faces[face_id] = {
                    'center': face['center'],
                    'features': face['features'],
                    'landmarks': face['landmarks'],
                    'bbox': face['bbox'],
                    'name': final_name,
                    'embedding': embedding,
                    'embedding_hash': embedding_hash,
                    'similarity': face.get('similarity', 0.0)
                }
                face['name'] = final_name  # Обновляем имя в текущем лице
            
            updated_faces.append(face)
        
        # Удаляем старые треки
        active_ids = [face['id'] for face in updated_faces]
        to_remove = [face_id for face_id in self.tracked_faces if face_id not in active_ids]
        for face_id in to_remove:
            # Удаляем связь ID -> эмбеддинг
            if face_id in self.id_to_embedding:
                embedding_hash = self.id_to_embedding[face_id]
                if embedding_hash in self.embedding_to_id:
                    del self.embedding_to_id[embedding_hash]
                del self.id_to_embedding[face_id]
            
            del self.tracked_faces[face_id]
        
        return updated_faces

    def assign_id_to_embedding(self, face_id, embedding):
        """Явно привязывает ID к эмбеддингу"""
        # ИСПРАВЛЕНИЕ: Правильная проверка numpy массива
        if embedding is None or len(embedding) == 0:
            return False
        
        embedding_hash = self._get_embedding_hash(embedding)
        if embedding_hash is None:
            return False
        
        # Сохраняем связь
        self.embedding_to_id[embedding_hash] = face_id
        self.id_to_embedding[face_id] = embedding_hash
        
        # Обновляем запись в tracked_faces
        if face_id in self.tracked_faces:
            self.tracked_faces[face_id]['embedding'] = embedding
            self.tracked_faces[face_id]['embedding_hash'] = embedding_hash
        
        return True

    def set_permanent_name(self, face_id, name):
        """Устанавливает постоянное имя для ID"""
        self.id_to_permanent_name[face_id] = name
        self.identified_ids.add(face_id)
        
        # Обновляем имя в текущем треке если он существует
        if face_id in self.tracked_faces:
            self.tracked_faces[face_id]['name'] = name
        
        return True
    
    def get_permanent_name(self, face_id):
        """Возвращает постоянное имя для ID"""
        return self.id_to_permanent_name.get(face_id, None)
    
    def is_identified(self, face_id):
        """Проверяет, был ли ID уже идентифицирован"""
        return face_id in self.identified_ids

    
# ================== ИНИЦИАЛИЗАЦИЯ БАЗЫ ЛИЦ ==================
print("\nИнициализация базы лиц...")
face_database = FaceDatabase(database_path='faces_database', similarity_threshold=0.4)
print("\nИнициализация хранилища идентифицированных лиц...")
face_storage = IdentifiedFaceStorage(storage_path='identified_faces')

# Пробуем загрузить из файла, если не получится - загрузим из изображений
if not face_database.load_from_file():
    face_database.load_database(insight_face)

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
tracker.set_face_database(face_database)
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

video_file = 'video/video_cut.mp4'
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
print(" - 'a' - добавить ФИО человека")
print(" - 'd' - список всех лиц")
print(" - 'p' - сохранить фото выбранного человека")
print(" - 'i' - информация об идентификации")
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
                # Bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # Добавляем padding
                padding = 20
                h, w = frame.shape[:2]
                roi_x1 = max(0, x1 - padding)
                roi_y1 = max(0, y1 - padding)
                roi_x2 = min(w, x2 + padding)
                roi_y2 = min(h, y2 + padding)
                
                # ROI лица
                face_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Нормализованные координаты
                global_landmarks = []
                for point in landmarks_5:
                    global_x = point[0] / frame.shape[1]
                    global_y = point[1] / frame.shape[0]
                    global_landmarks.append([global_x, global_y])
                
                # Извлекаем фичи для классификации позы
                features = extract_features_insightface(global_landmarks, frame.shape)
                
                 # Получаем эмбеддинг для идентификации
                embedding = face.normed_embedding if hasattr(face, 'normed_embedding') else None
                
                # Идентифицируем лицо (если есть эмбеддинг)
                name = None
                similarity = 0.0
                if embedding is not None and len(embedding) > 0:
                    name, similarity = tracker.identify_face(embedding, face_roi, face_storage)
                    embedding_hash = tracker._get_embedding_hash(embedding)
                
                # Центр лица
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detected_faces.append({
                    'landmarks': global_landmarks,
                    'features': features,
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'face_roi': face_roi,
                    'embedding': embedding,
                    'embedding_hash': face.get('embedding_hash', None),
                    'name': name,
                    'similarity': similarity,
                    'index': i
                })
        
        # Трекинг
        tracked_faces = tracker.update(detected_faces)
    else:
        if 'last_tracked_faces' in locals():
            tracked_faces = last_tracked_faces
        else:
            tracked_faces = []
    
    if process_this_frame:
        last_tracked_faces = tracked_faces.copy()
    
    # ================== ОТОБРАЖЕНИЕ ==================
    people_info = {}
    face_data = {}
    
    for face in tracked_faces:
        face_id = face['id']
        center = face['center']
        features = face['features']
        x1, y1, x2, y2 = face['bbox']
        landmarks = face['landmarks']
        face_roi = face.get('face_roi', None)
        name = face.get('name', None)
        similarity = face.get('similarity', 0.0)
        
        # Сохраняем фото
        if process_this_frame and face_roi is not None and face_roi.size > 0:
            quality = photo_manager.get_photo_quality(face_roi)
            photo_manager.add_photo(face_id, face_roi, quality)
        
        # Рисуем bounding box
        box_color = (0, 255, 255)  # Желтый по умолчанию
        
        # Меняем цвет если лицо идентифицировано
        if name:
            box_color = (0, 165, 255)  # Оранжевый для идентифицированных
        
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Ключевые точки
        if process_this_frame:
            for point in landmarks:
                px = int(point[0] * frame.shape[1])
                py = int(point[1] * frame.shape[0])
                cv2.circle(display_frame, (px, py), 3, (0, 255, 0), -1)
        
        # Предсказание класса (позы головы)
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
        
        # Отображаем ID и имя
        id_text = f"ID:{face_id}"

        if name is not None and len(str(name)) > 0:
            # Укорачиваем имя если слишком длинное
            display_name = str(name)[:15] + "..." if len(str(name)) > 15 else str(name)
            name_text = f"Name:{display_name}"
            
            # Показываем сходство если оно > 0
            if similarity > 0:
                name_text += f"({similarity:.2f})"
            
            # Цвет для имени
            name_color = (0, 255, 255)  # Желтый для идентифицированных
        else:
            name_text = "Name:Unknown"
            name_color = (200, 200, 200)  # Серый для неидентифицированных

        # Позиционирование текста (смещаем выше чтобы не перекрывать предсказание позы)
        cv2.putText(display_frame, id_text,
                (x1, y1 - 60),  # Подняли выше
                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)

        cv2.putText(display_frame, name_text,
                (x1, y1 - 35),  # Подняли выше
                cv2.FONT_HERSHEY_PLAIN, 1.2, name_color, 2)

        # Предсказание позы теперь ниже
        cv2.putText(display_frame, prediction_label,
                (x1, y1 - 10),  # Опустили ниже
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Сохраняем информацию для панели
        best_photo = photo_manager.get_best_photo(face_id)
        people_info[face_id] = {
            'center': center,
            'prediction': prediction_label,
            'bbox': (x1, y1, x2, y2),
            'features': features,
            'best_photo': best_photo,
            'name': name,
            'similarity': similarity
        }
        
        # Для калибровки
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

    # Добавляем информацию о базе лиц
    if face_database and len(face_database.names) > 0:
        db_info = f"DB:{len(face_database.names)} faces"
        cv2.putText(display_frame, db_info,
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
    
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
        
        # НОВАЯ КЛАВИША: 'a' - добавить текущее лицо в базу
        if key == ord('a') and selected_tid is not None and selected_tid in face_data:
            # Просим ввести имя
            print("\n=== ДОБАВЛЕНИЕ ЛИЦА В БАЗУ ===")
            print("Введите ФИО для лица ID", selected_tid, "(или нажмите Enter для отмены):")
            
            # Открываем диалог для ввода имени
            name = input("ФИО: ").strip()
            
            if name:
                # Находим лицо с выбранным ID
                for face in tracked_faces:
                    if face['id'] == selected_tid and 'embedding' in face:
                        embedding = face['embedding']
                        face_roi = face.get('face_roi', None)  # Получаем фото лица
                        
                        if embedding is not None:
                            success = face_database.add_face(embedding, name)
                            if success:
                                print(f"Лицо ID {selected_tid} добавлено в базу как '{name}'")
                                tracker.set_permanent_name(selected_tid, name)
                                face_database.save_database()
                                
                                # СОХРАНЯЕМ ФОТО АВТОМАТИЧЕСКИ
                                if face_roi is not None and face_roi.size > 0:
                                    if not face_storage.has_photo(name):
                                        face_storage.save_face_photo(name, face_roi)
                                        print(f"Фото сохранено для: {name}")
                            else:
                                print("Ошибка добавления лица в базу")
                        else:
                            print("Не удалось получить эмбеддинг лица")
                        break
            else:
                print("Добавление отменено")

        # НОВАЯ КЛАВИША: 'd' - показать список имен в базе
        if key == ord('d'):
            print("\n=== БАЗА ЛИЦ ===")
            names = face_database.get_all_names()
            if names:
                for i, name in enumerate(names):
                    print(f"{i+1}. {name}")
            else:
                print("База лиц пуста")
            print(f"Всего: {len(names)} записей")
        
        if key == ord('p'):  # Сохранить фото выбранного человека
            if selected_tid is not None and selected_tid in people_info:
                info = people_info[selected_tid]
                name = info.get('name', None)
                best_photo = info.get('best_photo', None)
                
                if name and best_photo is not None:
                    success = face_storage.save_face_photo(name, best_photo)
                    if success:
                        print(f"Фото сохранено для: {name}")
                    else:
                        print(f"Не удалось сохранить фото для: {name}")
                else:
                    print(f"Нет имени или фото для ID {selected_tid}")
    
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
    
    if key == ord('i'):  # Информация об идентификации
        print("\n=== ИНФОРМАЦИЯ ОБ ИДЕНТИФИКАЦИИ ===")
        print(f"Порог сходства в базе: {face_database.similarity_threshold}")
        print(f"Количество лиц в базе: {len(face_database.names)}")
        
        # Показать текущие идентификации
        for face_id, info in people_info.items():
            name = info.get('name')
            similarity = info.get('similarity', 0)
            if name:
                print(f"ID {face_id}: '{name}' (similarity: {similarity:.3f})")
    


cap.release()
cv2.destroyAllWindows()