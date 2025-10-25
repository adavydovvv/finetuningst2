#!/usr/bin/env python3
"""
Тест Gradio приложения
"""

import sys
from pathlib import Path

# Добавляем путь к app
ROOT = Path(".").resolve()
sys.path.append(str(ROOT / "app"))

def test_gradio_import():
    """Тест импорта Gradio приложения"""
    try:
        from gradio_app import predict, LABELS, create_interface
        print("✅ Импорт Gradio приложения успешен")
        print(f"🏷️ Классы: {LABELS}")
        return True
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_interface_creation():
    """Тест создания интерфейса"""
    try:
        from gradio_app import create_interface
        iface = create_interface()
        print("✅ Создание интерфейса успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания интерфейса: {e}")
        return False

def main():
    """Главная функция"""
    print("🧪 Тестирование Gradio приложения")
    print("="*40)
    
    # Тест импорта
    import_ok = test_gradio_import()
    
    if import_ok:
        # Тест создания интерфейса
        interface_ok = test_interface_creation()
        
        if interface_ok:
            print("\n🎉 Все тесты пройдены!")
            print("🚀 Приложение готово к запуску:")
            print("python app/gradio_app.py")
        else:
            print("\n❌ Есть проблемы с созданием интерфейса")
    else:
        print("\n❌ Есть проблемы с импортом")

if __name__ == "__main__":
    main()
