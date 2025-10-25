#!/usr/bin/env python3
"""
Тестовый скрипт для проверки приложения
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

# Добавляем путь к experiments
ROOT = Path(".").resolve()
sys.path.append(str(ROOT / "experiments"))

from datasets import ImageFolderSimple

def test_onnx_model():
    """Тест ONNX модели"""
    
    print("🧪 Тестирование ONNX модели")
    print("="*40)
    
    # Пути
    onnx_path = ROOT / "models" / "efficientnet_b0.onnx"
    data_dir = ROOT / "data" / "raw" / "val"
    
    # Проверяем файлы
    if not onnx_path.exists():
        print(f"❌ ONNX модель не найдена: {onnx_path}")
        print("Сначала экспортируйте модель:")
        print("python export_onnx.py")
        return False
    
    if not data_dir.exists():
        print(f"❌ Данные не найдены: {data_dir}")
        return False
    
    try:
        # Загружаем ONNX модель
        print(f"📦 Загрузка ONNX модели...")
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        # Получаем классы
        train_ds = ImageFolderSimple(str(ROOT / "data" / "raw" / "train"))
        class_names = list(train_ds.class_to_idx.keys())
        print(f"🏷️ Классы: {class_names}")
        
        # Предобработка (как в приложении)
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        preprocess = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # Тестируем на нескольких изображениях
        print(f"\n🔍 Тестирование на валидационных данных...")
        
        correct = 0
        total = 0
        
        for class_name in class_names:
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
                
            print(f"\n📁 Тестирование класса: {class_name}")
            
            for img_file in list(class_dir.glob("*.jpg"))[:3]:  # Тестируем по 3 изображения
                # Загружаем изображение
                img = Image.open(img_file).convert('RGB')
                
                # Предобработка
                img_array = np.array(img)
                img_processed = preprocess(image=img_array)['image']
                img_tensor = np.transpose(img_processed, (2, 0, 1)).astype(np.float32)[None, ...]
                
                # Инференс
                outputs = session.run(None, {'input': img_tensor})
                logits = outputs[0][0]
                
                # Softmax
                exp = np.exp(logits - np.max(logits))
                probs = exp / exp.sum()
                
                # Результат
                pred_class = class_names[np.argmax(probs)]
                pred_prob = np.max(probs)
                
                is_correct = pred_class == class_name
                correct += is_correct
                total += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} {img_file.name}: {pred_class} ({pred_prob:.3f})")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Результаты тестирования:")
        print(f"   Правильных: {correct}/{total}")
        print(f"   Точность: {accuracy:.3f}")
        
        return accuracy > 0.8
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def test_gradio_app():
    """Тест Gradio приложения"""
    
    print(f"\n🌐 Тестирование Gradio приложения")
    print("="*40)
    
    try:
        # Импортируем функции из приложения
        sys.path.append(str(ROOT / "app"))
        from gradio_app import predict, LABELS
        
        print(f"✅ Импорт функций успешен")
        print(f"🏷️ Классы: {LABELS}")
        
        # Создаем тестовое изображение
        test_img = Image.new('RGB', (224, 224), color='red')
        
        # Тестируем функцию предсказания
        results, text = predict(test_img)
        
        print(f"✅ Функция predict работает")
        print(f"📊 Результат: {text}")
        print(f"🏷️ Вероятности: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования приложения: {e}")
        return False

def main():
    """Главная функция"""
    
    print("🚀 Тестирование системы")
    print("="*50)
    
    # Тест ONNX модели
    onnx_ok = test_onnx_model()
    
    # Тест приложения
    app_ok = test_gradio_app()
    
    print(f"\n📋 Итоги тестирования:")
    print(f"   ONNX модель: {'✅' if onnx_ok else '❌'}")
    print(f"   Gradio приложение: {'✅' if app_ok else '❌'}")
    
    if onnx_ok and app_ok:
        print(f"\n🎉 Все тесты пройдены! Система готова к работе.")
        print(f"🚀 Запустите приложение: python app/gradio_app.py")
    else:
        print(f"\n❌ Есть проблемы. Проверьте ошибки выше.")

if __name__ == "__main__":
    main()
