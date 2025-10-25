#!/usr/bin/env python3
"""
Скрипт для запуска экспериментов по fine-tuning моделей
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Запуск экспериментов по fine-tuning")
    parser.add_argument("--mode", type=str, choices=["notebook", "basic", "hyperparams", "freezing"], 
                       default="notebook", help="Режим запуска")
    parser.add_argument("--model", type=str, choices=["resnet18", "efficientnet_b0", "both"], 
                       default="both", help="Модель для обучения")
    parser.add_argument("--epochs", type=int, default=12, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча")
    
    args = parser.parse_args()
    
    print("🚀 Запуск экспериментов по fine-tuning")
    print("="*50)
    
    # Проверка данных
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("❌ Папка с данными не найдена: data/raw")
        print("Убедитесь, что данные находятся в правильной структуре:")
        print("data/raw/train/{class_name}/")
        print("data/raw/val/{class_name}/")
        return
    
    # Проверка классов
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Не найдены папки train или val")
        return
    
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"✅ Найдены классы: {train_classes}")
    print(f"✅ Валидационные классы: {val_classes}")
    
    if train_classes != val_classes:
        print("⚠️  Классы в train и val не совпадают!")
    
    # Подсчет изображений
    for cls in train_classes:
        train_count = len(list((train_dir / cls).glob("*.jpg"))) + len(list((train_dir / cls).glob("*.png")))
        val_count = len(list((val_dir / cls).glob("*.jpg"))) + len(list((val_dir / cls).glob("*.png")))
        print(f"   {cls}: train={train_count}, val={val_count}")
    
    print("\n" + "="*50)
    
    if args.mode == "notebook":
        print("📓 Запуск Jupyter ноутбука с полными экспериментами...")
        print("Выполните в терминале:")
        print("jupyter notebook experiments/notebooks/comprehensive_experiments.ipynb")
        
    elif args.mode == "basic":
        print("🔧 Запуск базового обучения...")
        if args.model == "both":
            models = ["resnet18", "efficientnet_b0"]
        else:
            models = [args.model]
        
        for model in models:
            print(f"\nОбучение {model}...")
            cmd = f"python experiments/train.py --model_name {model} --epochs {args.epochs} --batch_size {args.batch_size}"
            print(f"Команда: {cmd}")
            os.system(cmd)
    
    elif args.mode == "hyperparams":
        print("🔍 Запуск подбора гиперпараметров...")
        print("Это может занять много времени!")
        cmd = "python experiments/hyperparameter_tuning.py"
        print(f"Команда: {cmd}")
        os.system(cmd)
    
    elif args.mode == "freezing":
        print("🧊 Запуск анализа стратегий замораживания...")
        cmd = "python experiments/freezing_strategies.py"
        print(f"Команда: {cmd}")
        os.system(cmd)
    
    print("\n✅ Готово!")

if __name__ == "__main__":
    main()
