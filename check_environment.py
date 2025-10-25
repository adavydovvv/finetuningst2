#!/usr/bin/env python3
"""
Проверка окружения для экспериментов
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    else:
        print("✅ Версия Python подходит")
        return True

def check_package(package_name, import_name=None):
    """Проверка установки пакета"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: не установлен")
        return False

def check_cuda():
    """Проверка CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна: {torch.cuda.get_device_name()}")
            print(f"   Количество GPU: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  CUDA недоступна, будет использоваться CPU")
            return False
    except ImportError:
        print("❌ PyTorch не установлен")
        return False

def check_data_structure():
    """Проверка структуры данных"""
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print("❌ Папка data/raw не найдена")
        return False
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists():
        print("❌ Папка data/raw/train не найдена")
        return False
    
    if not val_dir.exists():
        print("❌ Папка data/raw/val не найдена")
        return False
    
    # Проверка классов
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"✅ Найдены классы: {train_classes}")
    
    # Подсчет изображений
    total_train = 0
    total_val = 0
    
    for cls in train_classes:
        train_count = len(list((train_dir / cls).glob("*.jpg"))) + len(list((train_dir / cls).glob("*.png")))
        val_count = len(list((val_dir / cls).glob("*.jpg"))) + len(list((val_dir / cls).glob("*.png")))
        total_train += train_count
        total_val += val_count
        print(f"   {cls}: train={train_count}, val={val_count}")
    
    print(f"✅ Всего изображений: train={total_train}, val={total_val}")
    
    if total_train == 0 or total_val == 0:
        print("❌ Нет изображений в папках")
        return False
    
    return True

def main():
    print("🔍 Проверка окружения для экспериментов")
    print("="*50)
    
    all_good = True
    
    # Проверка Python
    print("\n1. Проверка Python:")
    all_good &= check_python_version()
    
    # Проверка пакетов
    print("\n2. Проверка пакетов:")
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("albumentations", "albumentations"),
        ("PIL", "PIL"),
        ("jupyter", "jupyter"),
    ]
    
    for package, import_name in required_packages:
        all_good &= check_package(package, import_name)
    
    # Проверка CUDA
    print("\n3. Проверка CUDA:")
    check_cuda()
    
    # Проверка данных
    print("\n4. Проверка структуры данных:")
    all_good &= check_data_structure()
    
    print("\n" + "="*50)
    
    if all_good:
        print("🎉 Всё готово для запуска экспериментов!")
        print("\nДля запуска используйте:")
        print("python run_experiments.py --mode notebook")
    else:
        print("❌ Есть проблемы с окружением")
        print("\nДля установки недостающих пакетов:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
