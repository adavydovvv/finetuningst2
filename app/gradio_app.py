# app/gradio_app.py
import gradio as gr
import onnxruntime as ort
import numpy as np
import albumentations as A
from PIL import Image
import os
import sys
from pathlib import Path

# Добавляем путь к experiments
ROOT = Path(".").resolve()
sys.path.append(str(ROOT / "experiments"))

from datasets import ImageFolderSimple

# -----------------------------
# Настройки - ЛУЧШАЯ МОДЕЛЬ ПО РЕЗУЛЬТАТАМ ЭКСПЕРИМЕНТОВ
# -----------------------------
IMAGE_SIZE = 224
DATA_DIR = "data/raw/train"  # путь к тренировочным данным, чтобы получить class_to_idx

# Получаем правильные метки классов из тренировочных данных
train_ds = ImageFolderSimple(DATA_DIR)
LABELS = [k for k, v in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
print("✅ Using labels:", LABELS)

# Предобработка для inference (ТОЧНО КАК ПРИ ВАЛИДАЦИИ)
preprocess = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Загрузка ONNX модели - ЛУЧШАЯ МОДЕЛЬ
MODEL_PATH = "models/efficientnet_b0.onnx"  # Лучшая модель по результатам экспериментов
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# -----------------------------
# Функция предсказания
# -----------------------------
def predict(image: Image.Image):
    """Предсказание класса для загруженного изображения"""
    try:
        # Конвертируем в RGB и применяем предобработку (как при валидации)
        img = np.array(image.convert('RGB'))
        img = preprocess(image=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)[None, ...]  # B,C,H,W

        # Инференс через ONNX Runtime
        outputs = session.run(None, {'input': img})
        logits = outputs[0][0]

        # Softmax для вероятностей
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        # Возвращаем результаты
        results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        
        # Находим лучший класс
        best_class = max(results, key=results.get)
        best_prob = results[best_class]
        
        return results, f"Лучший класс: {best_class} (вероятность: {best_prob:.3f})"
        
    except Exception as e:
        return {label: 0.0 for label in LABELS}, f"Ошибка: {str(e)}"

# -----------------------------
# Интерфейс Gradio
# -----------------------------
def create_interface():
    """Создание интерфейса Gradio"""
    
    with gr.Blocks(title="Классификация устройств", theme=gr.themes.Soft()) as iface:
        gr.Markdown("""
        # 🖥️ Классификация компьютерных устройств
        
        **Лучшая модель:** EfficientNet-B0 (96.67% точность)
        
        Загрузите изображение клавиатуры, мыши или звуковой карты для классификации.
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Загрузите изображение",
                    height=300
                )
                predict_btn = gr.Button("🔍 Классифицировать", variant="primary")
                
            with gr.Column():
                label_output = gr.Label(
                    label="Вероятности классов",
                    num_top_classes=len(LABELS)
                )
                text_output = gr.Textbox(
                    label="Результат",
                    interactive=False
                )
        
        # Примеры изображений (загружаем реальные файлы)
        gr.Markdown("### Примеры изображений:")
        
        # Загружаем примеры изображений
        examples = []
        val_dir = ROOT / "data" / "raw" / "val"
        for class_name in LABELS:
            class_dir = val_dir / class_name
            if class_dir.exists():
                # Берем первое изображение из каждой папки
                for img_file in class_dir.glob("*.jpg"):
                    examples.append([str(img_file)])
                    break
        
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
                label="Примеры из валидационного набора"
            )
        
        # Обработчик
        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, text_output]
        )
        
        # Автоматическое предсказание при загрузке изображения
        image_input.change(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, text_output]
        )
    
    return iface

if __name__ == "__main__":
    print("🚀 Запуск приложения Gradio...")
    print(f"📁 Модель: {MODEL_PATH}")
    print(f"🏷️ Классы: {LABELS}")
    
    iface = create_interface()
    iface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
