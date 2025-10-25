# Скопируйте этот код в ячейку ноутбука для исправления

# Импорт классов из experiments/datasets.py
import sys
import importlib

# Добавляем корень проекта в sys.path
project_root = str(ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импортируем модули
try:
    from experiments.datasets import ImageFolderSimple, get_transforms
    print("✅ Успешно импортированы классы из experiments.datasets")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    # Альтернативный способ импорта
    sys.path.append(str(ROOT / "experiments"))
    from datasets import ImageFolderSimple, get_transforms
    print("✅ Импорт через альтернативный путь")

# Создание датасетов - ИСПРАВЛЕНО: убран второй параметр
train_t, val_t = get_transforms(IMAGE_SIZE)
train_ds = ImageFolderSimple(str(TRAIN_DIR), transform=train_t)
val_ds = ImageFolderSimple(str(VAL_DIR), transform=val_t, classes=list(train_ds.class_to_idx.keys()))

class_names = list(train_ds.class_to_idx.keys())
num_classes = len(class_names)

print(f"Классы: {class_names}")
print(f"Количество классов: {num_classes}")
print(f"Тренировочных образцов: {len(train_ds)}")
print(f"Валидационных образцов: {len(val_ds)}")
