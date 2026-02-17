# monitoring_final_2025_calibrate.py
# Полностью рабочая версия с ОБУЧЕНИЕМ под конкретный ракурс камеры
# Ultralytics YOLO11 + Pose + ByteTrack + Онлайн-калибровка + Сохранение настроек

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from PIL import ImageFont, ImageDraw, Image
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==================== МОДЕЛЬ И ТРЕКЕР ====================
MODEL_NAME = "yolo/yolo11n-pose.pt"      # Лучшая лёгкая pose-модель 2025
TRACKER = "bytetrack.yaml"

# ==================== ШРИФТ (кириллица) ====================
FONT_PATH = "C:/Windows/Fonts/arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ==================== НАСТРОЙКИ (TUNE) ====================
TUNE = {
    "model_points": np.array([
        (0.0,    0.0,    0.0),       # 0: Нос
        (-65.0,  75.0,  50.0),       # 1: Левый глаз
        (65.0,   75.0,  50.0),       # 2: Правый глаз
        (-125.0, -75.0, -50.0),      # 3: Левое ухо
        (125.0, -75.0, -50.0),       # 4: Правое ухо
    ], dtype="double"),

    "history_maxlen": 10,
    "min_frames_for_stable": 6,
    "min_keypoint_conf": 0.55,
    "min_coordinate_value": 10,
    "focal_length": 900,
    "max_allowed_jump": 18,

    "font_size_small": 20,
    "font_size_big": 32,

    "smooth_alpha": 0.7,
    "history_for_mean": 12,
}

MODEL_POINTS = TUNE["model_points"]
pose_history = defaultdict(lambda: deque(maxlen=20))

# ==================== КЛАССЫ АКТИВНОСТИ ====================
ACTIVITIES = {
    "listening": "Слушает",
    "talking":   "Разговор",
    "phone":     "Телефон"
}

# ==================== Глобальные переменные калибровки ====================
calibration_data = []           # [(yaw, pitch, label), ...]
calibration_mode = False
selected_tid = None
classifier = None
scaler = None
calibration_file = "calibration_auditorium.pkl"

# ==================== Русский текст ====================
def put_text_ru(img, text, org, color=(0, 255, 0), size=TUNE["font_size_small"]):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except:
        font = ImageFont.load_default()
        size = 18
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== Калибровка: загрузка/сохранение ====================
def load_calibration():
    global classifier, scaler
    if os.path.exists(calibration_file):
        try:
            data = joblib.load(calibration_file)
            classifier = data['clf']
            scaler = data['scaler']
            print(f"Калибровка загружена: {len(data.get('samples', []))} образцов")
            return True
        except Exception as e:
            print(f"Ошибка загрузки калибровки: {e}")
    return False

def save_calibration():
    if len(calibration_data) < 10 or classifier is None:
        print("Нечего сохранять или мало данных")
        return
    joblib.dump({
        'clf': classifier,
        'scaler': scaler,
        'samples': calibration_data
    }, calibration_file)
    print(f"Калибровка сохранена → {calibration_file} ({len(calibration_data)} меток)")

# ==================== Обучение классификатора ====================
def plot_tree_matplotlib():
    if classifier is None:
        print("Сначала обучите модель (клавиша 's')")
        return

    import matplotlib.pyplot as plt

    # For RandomForest, we can't plot individual trees easily
    # Instead, we'll show feature importance
    if hasattr(classifier, 'feature_importances_'):
        print("Показываем важность признаков для RandomForest...")

        feature_names = ['yaw', 'pitch']
        importances = classifier.feature_importances_

        plt.figure(figsize=(8, 6), dpi=100)
        plt.barh(range(len(feature_names)), importances, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Важность признака')
        plt.title('Важность признаков в RandomForest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        print("Не удалось показать дерево решений (не поддерживается для RandomForest)")

def train_classifier():
    global classifier, scaler
    if len(calibration_data) < 15:
        print(f"Мало данных: {len(calibration_data)} (нужно ≥15)")
        return False

    X = np.array([[yaw, pitch] for yaw, pitch, _ in calibration_data])
    y = np.array([label for _, _, label in calibration_data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    classifier.fit(X_scaled, y)

    acc = classifier.score(X_scaled, y)
    print(f"\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"   Точность на ваших метках: {acc*100:.1f}%")
    print(f"   Всего использовано меток: {len(calibration_data)}")
    print("   Теперь система работает АВТОМАТИЧЕСКИ под этот ракурс!\n")
    return True

# ==================== Предсказание по модели ====================
def predict_activity(yaw, pitch):
    if classifier is None or scaler is None:
        return "listening"
    try:
        X = scaler.transform([[yaw, pitch]])
        return classifier.predict(X)[0]
    except:
        return "listening"

# ==================== Фильтр дёрганий ====================
def filter_outliers(yaw, pitch, prev_yaw, prev_pitch):
    jump = TUNE["max_allowed_jump"]
    if prev_yaw is not None:
        if abs(yaw - prev_yaw) > jump:
            yaw = prev_yaw + np.clip(yaw - prev_yaw, -jump, jump)
        if abs(pitch - prev_pitch) > jump:
            pitch = prev_pitch + np.clip(pitch - prev_pitch, -jump, jump)
    return yaw, pitch

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def process_video(source=0, flag_plot_tree=0):
    global calibration_mode, selected_tid, classifier, scaler

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Ошибка открытия источника: {source}")
        return

    # Автозагрузка калибровки при запуске
    auto_mode = load_calibration()
    if auto_mode:
        if flag_plot_tree:
            plot_tree_matplotlib()
        print("Калибровка загружена — система готова к автоматической работе!")
    else:
        print("Калибровка не найдена.")

    h, w = int(cap.get(4)), int(cap.get(3))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nРазрешение видео: {w}x{h}, {fps}")
    f = TUNE.get("focal_length", w)
    camera_matrix = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    model = YOLO(MODEL_NAME)

    print("\nМониторинг аудитории — Система с самообучением 2025")
    print("="*60)
    print("УПРАВЛЕНИЕ:")
    print("   c       → включить/выключить режим калибровки")
    print("   0-9     → выбрать человека по ID (в режиме калибровки)")
    print("   ,       → пометить: Слушает лекцию")
    print("   .       → пометить: Разговаривает")
    print("   /       → пометить: Смотрит в телефон")
    print("   s       → обучить модель по собранным меткам")
    print("   v       → сохранить калибровку в файл")
    print("   q       → выход")
    print("="*60)

    last_yaw_pitch = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось")
            break

        results = model.track(frame, persist=True, tracker=TRACKER, verbose=False)[0]
        annotated_frame = results.plot(boxes=True, masks=False, labels=False)

        count = len(results.boxes.id) if results.boxes.id is not None else 0

        # Индикатор режима
        mode_text = "КАЛИБРОВКА" if calibration_mode else ("АВТО" if classifier else "БАЗОВЫЙ")
        mode_color = (0, 0, 255) if calibration_mode else (0, 255, 0) if classifier else (100, 100, 255)
        annotated_frame = put_text_ru(annotated_frame, mode_text, (30, 70), color=mode_color, size=20)

        annotated_frame = put_text_ru(annotated_frame,
            f"Меток: {len(calibration_data)}", (30, 100), color=(255,255,255), size=20)

        if results.boxes.id is not None and results.keypoints is not None:
            ids = results.boxes.id.int().cpu().numpy()
            kpts = results.keypoints.xy.cpu().numpy()
            confs = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

            for i, tid in enumerate(ids):
                tid = int(tid)
                pts = kpts[i][[0,1,2,3,4]]
                conf = confs[i][[0,1,2]] if confs is not None else np.ones(3)
                box = results.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                # More robust visibility check - require all 3 key points (nose, left eye, right eye) to be visible
                # and have high confidence
                visible = all(c > TUNE["min_coordinate_value"] for c in pts[:3].flatten())
                has_high_confidence = np.all(conf > TUNE["min_keypoint_conf"])

                # Only process if both visibility and confidence conditions are met
                if has_high_confidence and visible:
                    success, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts.astype("double"),
                                                    camera_matrix, dist_coeffs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        # === Получаем углы Эйлера ===
                        rmat = cv2.Rodrigues(rvec)[0]
                        proj = np.hstack((rmat, np.zeros((3, 1))))
                        euler = cv2.decomposeProjectionMatrix(proj)[6]

                        raw_yaw = euler[1, 0]  # поворот влево-вправо
                        raw_pitch = euler[0, 0] - 45  # наклон вверх-вниз (коррекция)

                        # === 1. Жёсткая фильтрация скачков ===
                        if tid in last_yaw_pitch:
                            prev_yaw, prev_pitch = last_yaw_pitch[tid]
                            yaw = prev_yaw + np.clip(raw_yaw - prev_yaw, -18, 18)
                            pitch = prev_pitch + np.clip(raw_pitch - prev_pitch, -18, 18)
                        else:
                            yaw, pitch = raw_yaw, raw_pitch

                        last_yaw_pitch[tid] = (yaw, pitch)
                        pose_history[tid].append((yaw, pitch))

                        # === 2. Экспоненциальное сглаживание (главное!) ===
                        hist = np.array(pose_history[tid])
                        if len(hist) >= 2:
                            # Чем новее кадр — тем больший вес
                            weights = np.geomspace(0.5, 1.0, len(hist))
                            weights /= weights.sum()
                            smooth_yaw = np.sum(hist[:, 0] * weights)
                            smooth_pitch = np.sum(hist[:, 1] * weights)
                        else:
                            smooth_yaw, smooth_pitch = yaw, pitch

                        final_yaw = smooth_yaw
                        final_pitch = smooth_pitch

                        # === 3. ОПРЕДЕЛЕНИЕ АКТИВНОСТИ ===
                        if classifier is not None:
                            act = predict_activity(final_yaw, final_pitch)
                            activity_text = ACTIVITIES[act]
                        else:
                            # Классификация не происходит, можно оставить пустое значение или не делать ничего
                            activity_text = ""

                        # === Отладка: показываем сглаженные углы ===
                        annotated_frame = put_text_ru(annotated_frame,
                                                      f"Y:{final_yaw:+.0f}° P:{final_pitch:+.0f}°",
                                                      (x1 + 8, y1 - 18),
                                                      color=(255, 255, 0), size=20)

                        # === Подсветка выбранного для калибровки ===
                        if calibration_mode and selected_tid == tid:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                            annotated_frame = put_text_ru(annotated_frame, "ВЫБРАН",
                                                          (x1, y1 - 70), color=(0, 255, 255), size=30)

                    else:
                        # solvePnP не сошёлся
                        activity_text = ACTIVITIES["listening"]

                    # === ЕДИНСТВЕННАЯ ОСНОВНАЯ ПОДПИСЬ ===
                    annotated_frame = put_text_ru(annotated_frame,
                                                  f"ID {tid} · {activity_text}",
                                                  (x1 + 8, y1 - 45),
                                                  color=(50, 0, 255), size=TUNE["font_size_small"])

                else:
                    # Ключевые точки не видны или низкая уверенность
                    activity_text = ACTIVITIES["listening"]
                    annotated_frame = put_text_ru(annotated_frame,
                                                  f"ID {tid} · {activity_text}",
                                                  (x1 + 8, y1 - 45),
                                                  color=(100, 100, 100), size=TUNE["font_size_small"])


        # Счётчик людей
        annotated_frame = put_text_ru(annotated_frame,
            f"Людей в кадре: {count}", (20, 30), color=(0,255,255), size=20)

        cv2.imshow("Мониторинг аудитории — Самообучающаяся версия 2025", annotated_frame)

        # === Обработка клавиш ===
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            selected_tid = None
            print(f"\nРежим калибровки: {'ВКЛЮЧЕН' if calibration_mode else 'ВЫКЛЮЧЕН'}")
        elif key == ord('s'):
            train_classifier()
        elif key == ord('v'):
            save_calibration()

        # Выбор по ID (0–9)
        if calibration_mode and ord('0') <= key <= ord('9'):
            num = key - ord('0')
            if num in ids:
                selected_tid = num
                print(f"→ Выбран ID {num} для разметки")

        # Разметка выбранного человека
        if calibration_mode and selected_tid is not None and selected_tid in last_yaw_pitch:
            # Используем сглаженные значения!
            yaw, pitch = final_yaw, final_pitch
            if key == ord(','):
                calibration_data.append((yaw, pitch, "listening"))
                print(f"ID {selected_tid} → Слушает лекцию (Y:{yaw:+.1f}° P:{pitch:+.1f}°)")
            elif key == ord('.'):
                calibration_data.append((yaw, pitch, "talking"))
                print(f"ID {selected_tid} → Разговаривает (Y:{yaw:+.1f}° P:{pitch:+.1f}°)")
            elif key == ord('/'):
                calibration_data.append((yaw, pitch, "phone"))
                print(f"ID {selected_tid} → Телефон (Y:{yaw:+.1f}° P:{pitch:+.1f}°)")

    cap.release()
    cv2.destroyAllWindows()
    if classifier:
        save_calibration()
    print("Программа завершена.")

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    process_video(0,0)
    #process_video("http://192.168.0.107:8080/video")  # веб-камера
    #process_video("video/two_video.mp4", 0) # видео
