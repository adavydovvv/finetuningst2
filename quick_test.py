#!/usr/bin/env python3
"""
Быстрый тест для проверки работоспособности
"""

import os
import sys
from pathlib import Path

# Добавляем путь к experiments в sys.path
sys.path.append(str(Path(__file__).parent / "experiments"))

def test_imports():
    """Тест импортов"""
    print("🔍 Тестирование импортов...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import timm
        print(f"✅ timm: {timm.__version__}")
    except ImportError as e:
        print(f"❌ timm: {e}")
        return False
    
    try:
        import albumentations as A
        print(f"✅ albumentations: {A.__version__}")
    except ImportError as e:
        print(f"❌ albumentations: {e}")
        return False
    
    try:
        from sklearn.metrics import confusion_matrix
        print("✅ scikit-learn")
    except ImportError as e:
        print(f"❌ scikit-learn: {e}")
        return False
    
    return True

def test_data_structure():
    """Тест структуры данных"""
    print("\n🔍 Тестирование структуры данных...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Папка {data_dir} не найдена")
        return False
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        print(f"❌ Папка {train_dir} не найдена")
        return False
    
    if not val_dir.exists():
        print(f"❌ Папка {val_dir} не найдена")
        return False
    
    # Проверка классов
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"✅ Классы в train: {train_classes}")
    print(f"✅ Классы в val: {val_classes}")
    
    # Подсчет изображений
    total_train = 0
    total_val = 0
    
    for cls in train_classes:
        cls_dir = train_dir / cls
        count = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png")))
        total_train += count
        print(f"   {cls}: {count} изображений")
    
    for cls in val_classes:
        cls_dir = val_dir / cls
        count = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png")))
        total_val += count
    
    print(f"✅ Всего изображений: train={total_train}, val={total_val}")
    
    if total_train == 0 or total_val == 0:
        print("❌ Нет изображений в папках")
        return False
    
    return True

def test_model_creation():
    """Тест создания модели"""
    print("\n🔍 Тестирование создания модели...")
    
    try:
        import torch
        import timm
        
        # Создание простой модели
        model = timm.create_model('resnet18', pretrained=True, num_classes=3)
        print("✅ ResNet18 создана успешно")
        
        # Подсчет параметров
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Параметров: {total_params:,}")
        
        # Тест forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания модели: {e}")
        return False

def main():
    print("🚀 Быстрый тест системы")
    print("="*40)
    
    all_good = True
    
    # Тест импортов
    all_good &= test_imports()
    
    # Тест данных
    all_good &= test_data_structure()
    
    # Тест модели
    all_good &= test_model_creation()
    
    print("\n" + "="*40)
    
    if all_good:
        print("🎉 Все тесты пройдены! Система готова к работе.")
        print("\nДля запуска экспериментов:")
        print("1. jupyter notebook experiments/notebooks/comprehensive_experiments.ipynb")
        print("2. python experiments/train.py --model_name resnet18 --epochs 2")
    else:
        print("❌ Есть проблемы. Проверьте установку зависимостей:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
