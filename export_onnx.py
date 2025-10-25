#!/usr/bin/env python3
"""
Скрипт для экспорта обученной модели в ONNX
"""

import os
import sys
import torch
import onnx
import timm
import argparse
from pathlib import Path

# Добавляем путь к experiments
ROOT = Path(".").resolve()
sys.path.append(str(ROOT / "experiments"))

from configs import BEST_MODEL_CONFIG

def export_model_to_onnx(model_name, num_classes=3, image_size=224, models_dir=None):
    """Экспорт модели в ONNX"""
    
    if models_dir is None:
        models_dir = ROOT / "models"
    else:
        models_dir = Path(models_dir)
    
    print(f"🚀 Экспорт модели {model_name} в ONNX")
    print("="*50)
    
    print(f"Модель: {model_name}")
    print(f"Классы: {num_classes}")
    print(f"Размер изображения: {image_size}")
    print(f"Директория: {models_dir}")
    
    # Пути к файлам
    weights_path = models_dir / f"best_{model_name}.pth"
    onnx_path = models_dir / f"{model_name}.onnx"
    
    # Проверяем наличие весов
    if not weights_path.exists():
        print(f"❌ Веса модели не найдены: {weights_path}")
        print("Сначала обучите модель:")
        print(f"python experiments/train.py --model_name {model_name} --epochs 12")
        return False
    
    try:
        # Создаем модель
        print(f"\n📦 Создание модели {model_name}...")
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Загружаем веса
        print(f"📥 Загрузка весов из {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        
        # Создаем dummy input
        dummy_input = torch.randn(1, 3, image_size, image_size)
        
        # Экспорт в ONNX
        print(f"🔄 Экспорт в ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,  # Используем более новую версию
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_shapes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Проверяем ONNX модель
        print(f"✅ Проверка ONNX модели...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Информация о файле
        file_size = onnx_path.stat().st_size / 1024 / 1024
        print(f"\n🎉 Экспорт завершен успешно!")
        print(f"📁 Файл: {onnx_path}")
        print(f"📏 Размер: {file_size:.2f} MB")
        
        # Тест инференса
        print(f"\n🧪 Тест инференса...")
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        # Тестовый инференс
        test_input = torch.randn(1, 3, image_size, image_size).numpy().astype('float32')
        outputs = session.run(None, {'input': test_input})
        
        print(f"✅ Тест инференса прошел успешно!")
        print(f"📊 Выходной размер: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка экспорта: {e}")
        return False

def main():
    """Главная функция с поддержкой командной строки"""
    parser = argparse.ArgumentParser(description='Экспорт модели в ONNX')
    parser.add_argument('--model_name', type=str, default=BEST_MODEL_CONFIG.model_name,
                       help=f'Имя модели (по умолчанию: {BEST_MODEL_CONFIG.model_name})')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Количество классов (по умолчанию: 3)')
    parser.add_argument('--image_size', type=int, default=BEST_MODEL_CONFIG.image_size,
                       help=f'Размер изображения (по умолчанию: {BEST_MODEL_CONFIG.image_size})')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Директория с моделями (по умолчанию: models)')
    
    args = parser.parse_args()
    
    success = export_model_to_onnx(
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        models_dir=args.models_dir
    )
    
    if success:
        print(f"\n🚀 Теперь можно запустить приложение:")
        print(f"python app/gradio_app.py")
    else:
        print(f"\n❌ Экспорт не удался. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()
