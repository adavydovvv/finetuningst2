#!/usr/bin/env python3
"""
Тест импорта для проверки работоспособности
"""

import sys
from pathlib import Path

# Настройка путей
ROOT = Path(".").resolve()
print(f"ROOT: {ROOT}")

# Добавляем корень проекта в sys.path
project_root = str(ROOT)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"sys.path: {sys.path[:3]}...")  # Показываем первые 3 элемента

# Тест импорта
try:
    from experiments.datasets import ImageFolderSimple, get_transforms
    print("✅ Успешно импортированы классы из experiments.datasets")
    
    # Тест создания трансформаций
    train_t, val_t = get_transforms(224, 'medium')
    print("✅ Успешно созданы трансформации")
    
    # Тест создания датасета
    DATA_DIR = ROOT / "data" / "raw"
    TRAIN_DIR = DATA_DIR / "train"
    
    if TRAIN_DIR.exists():
        train_ds = ImageFolderSimple(str(TRAIN_DIR), transform=train_t)
        print(f"✅ Успешно создан датасет: {len(train_ds)} образцов")
        print(f"   Классы: {list(train_ds.class_to_idx.keys())}")
    else:
        print(f"❌ Папка {TRAIN_DIR} не найдена")
        
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Пробуем альтернативный способ...")
    
    try:
        sys.path.append(str(ROOT / "experiments"))
        from datasets import ImageFolderSimple, get_transforms
        print("✅ Импорт через альтернативный путь")
    except ImportError as e2:
        print(f"❌ Альтернативный импорт тоже не работает: {e2}")

except Exception as e:
    print(f"❌ Другая ошибка: {e}")
