#!/usr/bin/env python3
"""
Быстрый экспорт лучшей модели в ONNX
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Быстрый экспорт с проверками"""
    
    print("🚀 Быстрый экспорт лучшей модели в ONNX")
    print("="*50)
    
    # Проверяем наличие весов
    models_dir = Path("models")
    best_weights = models_dir / "best_efficientnet_b0.pth"
    
    if not best_weights.exists():
        print("❌ Веса модели не найдены!")
        print("Сначала обучите модель:")
        print("python experiments/train.py --model_name efficientnet_b0 --epochs 12")
        return False
    
    # Запускаем экспорт
    print("🔄 Запуск экспорта...")
    try:
        result = subprocess.run([
            sys.executable, "export_onnx.py", 
            "--model_name", "efficientnet_b0"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Экспорт завершен успешно!")
        print(result.stdout)
        
        # Проверяем результат
        onnx_file = models_dir / "efficientnet_b0.onnx"
        if onnx_file.exists():
            size_mb = onnx_file.stat().st_size / 1024 / 1024
            print(f"\n📁 ONNX файл: {onnx_file}")
            print(f"📏 Размер: {size_mb:.2f} MB")
            
            print(f"\n🚀 Теперь можно запустить приложение:")
            print(f"python app/gradio_app.py")
            
            return True
        else:
            print("❌ ONNX файл не создан!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка экспорта: {e}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
