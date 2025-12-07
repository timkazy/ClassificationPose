import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import time

# Классы
class_labels = {
    1: "listening",
    2: "talking", 
    3: "phone"
}

# Цвета для визуализации
class_colors = {
    1: 'green',    # listening
    2: 'blue',     # talking
    3: 'purple'    # phone
}

def load_raw_data(filename='raw_calibration_data.pkl'):
    """Загружает сырые данные из файла"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Загружены сырые данные из {filename}")
        return data
    except Exception as e:
        print(f"Ошибка при загрузке файла {filename}: {e}")
        return None

def prepare_data(data):
    """Подготавливает данные для обучения"""
    X = []
    y = []
    class_counts = {}
    
    for label, samples in data.items():
        if len(samples) > 0:
            X.extend(samples)
            y.extend([label] * len(samples))
            class_counts[label] = len(samples)
            print(f"Класс {class_labels.get(label, label)}: {len(samples)} образцов")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nОбщее количество образцов: {len(X)}")
    print(f"Количество признаков: {X.shape[1]}")
    
    return X, y, class_counts

def get_classification_methods():
    """Возвращает список доступных методов классификации"""
    methods = {
        'rf': {
            'name': 'Random Forest',
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'description': 'Ансамбль решающих деревьев, устойчив к переобучению'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'class': SVC,
            'params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'description': 'Ищет оптимальную разделяющую гиперплоскость'
        },
        'knn': {
            'name': 'K-Nearest Neighbors',
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5,
                'weights': 'distance',
                'n_jobs': -1
            },
            'description': 'Классифицирует по k ближайшим соседям'
        },
        'gb': {
            'name': 'Gradient Boosting',
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'description': 'Бустинг деревьев с градиентным спуском'
        },
        'ada': {
            'name': 'AdaBoost',
            'class': AdaBoostClassifier,
            'params': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            },
            'description': 'Адаптивный бустинг с весами образцов'
        },
        'mlp': {
            'name': 'Neural Network (MLP)',
            'class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'max_iter': 500,
                'random_state': 42
            },
            'description': 'Многослойный перцептрон (нейронная сеть)'
        },
        'dt': {
            'name': 'Decision Tree',
            'class': DecisionTreeClassifier,
            'params': {
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'description': 'Одно решающее дерево (базовый метод)'
        }
    }
    return methods

def train_model_with_method(X_train, y_train, method_key='rf'):
    """Обучает модель выбранным методом"""
    methods = get_classification_methods()
    
    if method_key not in methods:
        print(f"Метод {method_key} не найден. Использую Random Forest.")
        method_key = 'rf'
    
    method_info = methods[method_key]
    print(f"\n=== ОБУЧЕНИЕ МОДЕЛИ: {method_info['name']} ===")
    print(f"Описание: {method_info['description']}")
    
    # Создаем модель
    model = method_info['class'](**method_info['params'])
    
    # Кросс-валидация
    start_time = time.time()
    
    if len(X_train) >= 10:
        print("\nПроводим кросс-валидацию...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), n_jobs=-1)
        print(f"Результаты кросс-валидации ({len(cv_scores)}-fold):")
        print(f"  Средняя точность: {cv_scores.mean():.4f}")
        print(f"  Стандартное отклонение: {cv_scores.std():.4f}")
        print(f"  Все значения: {[f'{x:.4f}' for x in cv_scores]}")
    else:
        print("Недостаточно данных для кросс-валидации")
    
    # Обучение на всех данных
    print("\nОбучаем модель на тренировочных данных...")
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Время обучения: {training_time:.2f} секунд")
    
    return model, method_info, training_time

def compare_methods(X_train, y_train, X_test, y_test):
    """Сравнивает разные методы обучения"""
    methods = get_classification_methods()
    results = {}
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МЕТОДОВ КЛАССИФИКАЦИИ")
    print("="*70)
    
    for method_key, method_info in methods.items():
        print(f"\nМетод: {method_info['name']}")
        print(f"Описание: {method_info['description']}")
        
        try:
            start_time = time.time()
            model = method_info['class'](**method_info['params'])
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            if len(X_test) > 0:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[method_key] = {
                    'name': method_info['name'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'training_time': training_time,
                    'model': model,
                    'method_info': method_info  # Сохраняем полную информацию о методе
                }
                
                print(f"  Точность: {accuracy:.4f}")
                print(f"  F1-score: {f1:.4f}")
                print(f"  Время обучения: {training_time:.2f} сек")
            else:
                # Если нет тестовых данных, используем кросс-валидацию
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), n_jobs=-1)
                results[method_key] = {
                    'name': method_info['name'],
                    'accuracy': cv_scores.mean(),
                    'precision': cv_scores.mean(),  # Приблизительно
                    'recall': cv_scores.mean(),
                    'f1': cv_scores.mean(),
                    'training_time': training_time,
                    'cv_scores': cv_scores,
                    'model': model,
                    'method_info': method_info  # Сохраняем полную информацию о методе
                }
                print(f"  Кросс-валидация: {cv_scores.mean():.4f}")
                print(f"  Время обучения: {training_time:.2f} сек")
                
        except Exception as e:
            print(f"  Ошибка: {e}")
            results[method_key] = None
    
    return results

def plot_comparison(results, ax):
    """Рисует сравнение методов"""
    if not results:
        ax.text(0.5, 0.5, 'Нет результатов\nдля сравнения', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Фильтруем успешные результаты
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        ax.text(0.5, 0.5, 'Нет валидных\nрезультатов', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    methods = list(valid_results.keys())
    names = [valid_results[m]['name'] for m in methods]
    accuracies = [valid_results[m]['accuracy'] for m in methods]
    
    # Цвета в зависимости от точности
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(methods)))
    
    bars = ax.barh(names, accuracies, color=colors)
    ax.set_xlabel('Точность')
    ax.set_title('Сравнение методов классификации', fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения
    for bar, acc in zip(bars, accuracies):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', ha='left', va='center', fontweight='bold')

def evaluate_model(clf, X_test, y_test):
    """Оценивает модель на тестовых данных"""
    print("\n=== ОЦЕНКА МОДЕЛИ ===")
    
    if len(X_test) == 0:
        print("Нет тестовых данных для оценки!")
        return None, 0, None, None, None, None
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Точность: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Время предсказания: {inference_time:.4f} сек")
    
    print("\nПодробный отчет о классификации:")
    print(classification_report(y_test, y_pred, 
                               target_names=[class_labels.get(i, f"Class {i}") for i in sorted(class_labels.keys())]))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nМатрица ошибок:")
    print(cm)
    
    return y_pred, accuracy, cm, precision, recall, f1

def plot_confusion_matrix(cm, ax):
    """Рисует матрицу ошибок"""
    if cm is None:
        ax.text(0.5, 0.5, 'Нет данных\nдля матрицы ошибок', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    classes = [class_labels.get(i, f"Class {i}") for i in sorted(class_labels.keys())]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

def visualize_results(clf, method_info, X_train, y_train, X_test, y_test, y_pred, cm, 
                     precision, recall, f1, training_time, comparison_results=None):
    """Визуализирует результаты обучения"""
    
    # Получаем полную информацию о методе, если передана только строка
    if isinstance(method_info, str) or 'params' not in method_info:
        methods = get_classification_methods()
        # Находим метод по имени
        for key, info in methods.items():
            if info['name'] == method_info:
                method_info = info
                break
        if 'params' not in method_info:
            # Создаем базовую информацию
            method_info = {
                'name': method_info if isinstance(method_info, str) else 'Unknown',
                'params': {},
                'description': 'Неизвестный метод'
            }
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Результаты обучения: {method_info["name"]}\nВремя: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                fontsize=14, fontweight='bold')
    
    # 1. Сравнение методов (если есть)
    if comparison_results:
        ax1 = plt.subplot(3, 3, 1)
        plot_comparison(comparison_results, ax1)
    else:
        ax1 = plt.subplot(3, 3, 1)
        ax1.text(0.5, 0.5, 'Сравнение методов\nне проводилось', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Сравнение методов', fontweight='bold')
    
    # 2. Матрица ошибок
    ax2 = plt.subplot(3, 3, 2)
    plot_confusion_matrix(cm, ax2)
    ax2.set_title('Матрица ошибок', fontweight='bold')
    ax2.set_ylabel('Истинные значения')
    ax2.set_xlabel('Предсказанные значения')
    
    # 3. Метрики качества
    ax3 = plt.subplot(3, 3, 3)
    if cm is not None and y_pred is not None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy_score(y_test, y_pred) if y_pred is not None else 0, 
                 precision if precision is not None else 0,
                 recall if recall is not None else 0,
                 f1 if f1 is not None else 0]
        
        colors = ['green', 'blue', 'orange', 'red']
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_title('Метрики качества', fontweight='bold')
        ax3.set_ylabel('Значение')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Нет данных\nдля метрик', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Метрики качества', fontweight='bold')
    
    # 4. Важность признаков (если доступно)
    ax4 = plt.subplot(3, 3, 4)
    if hasattr(clf, 'feature_importances_'):
        feature_importance = clf.feature_importances_
        top_n = min(10, len(feature_importance))
        sorted_idx = np.argsort(feature_importance)[-top_n:]
        
        y_pos = np.arange(len(sorted_idx))
        ax4.barh(y_pos, feature_importance[sorted_idx])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"Feat {i}" for i in sorted_idx])
        ax4.set_title(f'Топ-{top_n} важных признаков', fontweight='bold')
        ax4.set_xlabel('Важность')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Нет данных\nо важности признаков', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Важность признаков', fontweight='bold')
    
    # 5. Распределение классов
    ax5 = plt.subplot(3, 3, 5)
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = [class_labels.get(i, f"Class {i}") for i in unique]
    colors = [class_colors.get(i, 'gray') for i in unique]
    
    bars = ax5.bar(class_names, counts, color=colors, alpha=0.7)
    ax5.set_title('Распределение классов (тренировка)', fontweight='bold')
    ax5.set_xlabel('Классы')
    ax5.set_ylabel('Количество')
    ax5.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 6. Точность по классам
    ax6 = plt.subplot(3, 3, 6)
    if y_pred is not None and len(y_test) > 0:
        class_accuracies = {}
        for class_id in np.unique(y_test):
            mask = y_test == class_id
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                class_accuracies[class_id] = class_acc
        
        if class_accuracies:
            sorted_classes = sorted(class_accuracies.keys())
            acc_values = [class_accuracies[cls] for cls in sorted_classes]
            class_names = [class_labels.get(cls, f"Class {cls}") for cls in sorted_classes]
            colors = [class_colors.get(cls, 'gray') for cls in sorted_classes]
            
            bars = ax6.bar(class_names, acc_values, color=colors, alpha=0.7)
            ax6.set_title('Точность по классам (тест)', fontweight='bold')
            ax6.set_xlabel('Классы')
            ax6.set_ylabel('Точность')
            ax6.set_ylim(0, 1.1)
            ax6.grid(True, alpha=0.3)
            
            for bar, acc in zip(bars, acc_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Нет тестовых данных', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Точность по классам', fontweight='bold')
    
    # 7. Уверенность предсказаний
    ax7 = plt.subplot(3, 3, 7)
    if len(X_test) > 0 and hasattr(clf, 'predict_proba'):
        try:
            probabilities = clf.predict_proba(X_test)
            confidence = np.max(probabilities, axis=1)
            
            ax7.hist(confidence, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax7.set_title('Уверенность предсказаний', fontweight='bold')
            ax7.set_xlabel('Уверенность')
            ax7.set_ylabel('Количество')
            ax7.axvline(x=0.7, color='r', linestyle='--', linewidth=2, label='Порог 70%')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        except:
            ax7.text(0.5, 0.5, 'Ошибка при расчете\nуверенности', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Уверенность предсказаний', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Нет данных\nдля уверенности', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Уверенность предсказаний', fontweight='bold')
    
    # 8. Информация о модели
    ax8 = plt.subplot(3, 3, 8)
    model_info = []
    model_info.append(f"Метод: {method_info['name']}")
    model_info.append(f"Время обучения: {training_time:.2f} сек")
    model_info.append(f"Тренировочные данные: {len(X_train)}")
    model_info.append(f"Тестовые данные: {len(X_test)}")
    
    if hasattr(clf, 'n_estimators'):
        model_info.append(f"Количество деревьев: {clf.n_estimators}")
    if hasattr(clf, 'n_features_in_'):
        model_info.append(f"Признаков: {clf.n_features_in_}")
    if hasattr(clf, 'n_classes_'):
        model_info.append(f"Классов: {clf.n_classes_}")
    
    ax8.text(0.1, 0.9, '\n'.join(model_info), transform=ax8.transAxes,
            verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Информация о модели', fontweight='bold')
    ax8.axis('off')
    
    # 9. Дополнительная информация
    ax9 = plt.subplot(3, 3, 9)
    extra_info = ["Рекомендации:"]
    
    if len(X_train) < 50:
        extra_info.append("- Собрать больше данных")
    if cm is not None and np.trace(cm) / np.sum(cm) < 0.8:
        extra_info.append("- Проверить баланс классов")
    if training_time > 10:
        extra_info.append("- Уменьшить сложность модели")
    
    if len(extra_info) == 1:
        extra_info.append("- Модель готова к использованию")
    
    ax9.text(0.1, 0.9, '\n'.join(extra_info), transform=ax9.transAxes,
            verticalalignment='top', fontfamily='monospace', color='darkgreen')
    ax9.set_title('Рекомендации', fontweight='bold')
    ax9.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод информации о модели в консоль
    print("\n" + "="*70)
    print(f"ИТОГИ ДЛЯ МЕТОДА: {method_info['name']}")
    print("="*70)
    
    # Безопасный вывод параметров
    if 'params' in method_info and method_info['params']:
        print(f"Параметры модели:")
        for key, value in method_info['params'].items():
            print(f"  {key}: {value}")
    else:
        print("Параметры модели: По умолчанию")
    
    print(f"Время обучения: {training_time:.2f} секунд")
    print(f"Тренировочных образцов: {len(X_train)}")
    print(f"Тестовых образцов: {len(X_test) if X_test is not None else 0}")
    
    if hasattr(clf, 'estimators_') and clf.estimators_:
        depths = [est.tree_.max_depth for est in clf.estimators_]
        print(f"Глубина деревьев: средняя={np.mean(depths):.1f}")

def save_trained_model(clf, method_info, filename='trained_data.pkl'):
    """Сохраняет обученную модель в файл"""
    # Получаем ключ метода
    methods = get_classification_methods()
    method_key = None
    for key, info in methods.items():
        if info['name'] == method_info['name']:
            method_key = key
            break
    
    model_data = {
        'model': clf,
        'method_name': method_info['name'],
        'method_key': method_key,
        'method_info': method_info,  # Сохраняем всю информацию
        'class_labels': class_labels,
        'class_colors': class_colors,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_features': clf.n_features_in_ if hasattr(clf, 'n_features_in_') else 0,
        'n_classes': clf.n_classes_ if hasattr(clf, 'n_classes_') else 0
    }
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Модель сохранена в файл: {filename}")
        print(f"  Метод: {method_info['name']}")
        print(f"  Размер файла: {os.path.getsize(filename) / 1024:.2f} KB")
        print(f"  Время сохранения: {model_data['timestamp']}")
        return True
    except Exception as e:
        print(f"✗ Ошибка при сохранении модели: {e}")
        return False

def select_method_interactive():
    """Интерактивный выбор метода обучения"""
    methods = get_classification_methods()
    
    print("\n" + "="*70)
    print("ВЫБОР МЕТОДА КЛАССИФИКАЦИИ")
    print("="*70)
    
    for i, (key, info) in enumerate(methods.items(), 1):
        print(f"{i}. {info['name']:25} - {info['description']}")
    
    print("\nДоступные команды:")
    print("  'all' - сравнить все методы и выбрать лучший")
    print("  'rf', 'svm', 'knn', 'gb', 'ada', 'mlp', 'dt' - выбрать конкретный метод")
    print("  'q' - выйти")
    
    while True:
        choice = input("\nВыберите метод или введите команду: ").strip().lower()
        
        if choice == 'q':
            return None
        elif choice == 'all':
            return 'all'
        elif choice in methods:
            return choice
        elif choice.isdigit() and 1 <= int(choice) <= len(methods):
            # Преобразуем номер в ключ
            keys = list(methods.keys())
            return keys[int(choice) - 1]
        else:
            print("Некорректный выбор. Попробуйте снова.")

def main():
    """Основная функция обучения"""
    print("=" * 70)
    print("ТРЕНИРОВКА МОДЕЛИ ДЛЯ КЛАССИФИКАЦИИ ПОЗЫ ГОЛОВЫ")
    print("=" * 70)
    
    # 1. Загрузка данных
    print("\n1. ЗАГРУЗКА ДАННЫХ...")
    data = load_raw_data()
    if data is None:
        print("✗ Не удалось загрузить данные.")
        print("  Убедитесь, что файл raw_calibration_data.pkl существует")
        print("  и содержит данные калибровки.")
        return
    
    # 2. Подготовка данных
    print("\n2. ПОДГОТОВКА ДАННЫХ...")
    X, y, class_counts = prepare_data(data)
    
    if len(X) == 0:
        print("✗ Нет данных для обучения!")
        print("  Сначала соберите данные калибровки в face.py")
        return
    
    # 3. Разделение данных
    print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ...")
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"✓ Тренировочные: {len(X_train)}, Тестовые: {len(X_test)}")
    else:
        print(f"⚠ Мало данных ({len(X)}), используем все для тренировки")
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])
    
    # 4. Выбор метода
    print("\n4. ВЫБОР МЕТОДА ОБУЧЕНИЯ...")
    selected_method = select_method_interactive()
    
    if selected_method is None:
        print("Выход...")
        return
    
    comparison_results = None
    best_model = None
    best_method_info = None
    training_time = 0
    
    # 5. Обучение и оценка
    if selected_method == 'all':
        print("\n5. СРАВНЕНИЕ ВСЕХ МЕТОДОВ...")
        comparison_results = compare_methods(X_train, y_train, X_test, y_test)
        
        # Находим лучший метод
        best_accuracy = 0
        best_method_key = None
        for method_key, result in comparison_results.items():
            if result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = result['model']
                best_method_info = result['method_info']  # Используем сохраненную информацию
                best_method_key = method_key
                training_time = result['training_time']
        
        if best_model:
            print(f"\n✓ Лучший метод: {best_method_info['name']}")
            print(f"  Точность: {best_accuracy:.4f}")
            print(f"  Ключ метода: {best_method_key}")
        else:
            print("✗ Не удалось найти подходящий метод!")
            return
        
    else:
        print(f"\n5. ОБУЧЕНИЕ МЕТОДОМ {selected_method.upper()}...")
        best_model, best_method_info, training_time = train_model_with_method(
            X_train, y_train, selected_method
        )
    
    if best_model is None:
        print("✗ Не удалось обучить модель!")
        return
    
    # 6. Оценка модели
    print("\n6. ОЦЕНКА МОДЕЛИ...")
    y_pred, accuracy, cm, precision, recall, f1 = evaluate_model(
        best_model, X_test, y_test
    )
    
    # 7. Визуализация
    print("\n7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    visualize_results(best_model, best_method_info, X_train, y_train, 
                     X_test, y_test, y_pred, cm, precision, recall, f1,
                     training_time, comparison_results)
    
    # 8. Сохранение
    print("\n8. СОХРАНЕНИЕ МОДЕЛИ...")
    save_choice = input("\nСохранить обученную модель? (y/n): ").strip().lower()
    if save_choice == 'y':
        if save_trained_model(best_model, best_method_info):
            print("\n" + "="*70)
            print("✓ Модель готова к использованию!")
            print(f"  Метод: {best_method_info['name']}")
            print("  Запустите face.py для использования в режиме AUTO")
            print("="*70)
    else:
        print("\nМодель не сохранена.")

if __name__ == "__main__":
    main()