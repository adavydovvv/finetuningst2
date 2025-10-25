# app/gradio_app.py
import gradio as gr
import onnxruntime as ort
import numpy as np
import albumentations as A
from PIL import Image
import os
from experiments.datasets import ImageFolderSimple

# -----------------------------
# Настройки
# -----------------------------
IMAGE_SIZE = 224
DATA_DIR = "data/raw/train"  # путь к тренировочным данным, чтобы получить class_to_idx

# Получаем правильные метки классов из тренировочных данных
train_ds = ImageFolderSimple(DATA_DIR)
LABELS = [k for k, v in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]
print("✅ Using labels:", LABELS)

# Предобработка для inference (как при валидации)
preprocess = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Загрузка ONNX модели
MODEL_PATH = "models/computer_devices.onnx"  # убедись, что файл существует
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# -----------------------------
# Функция предсказания
# -----------------------------
def predict(image: Image.Image):
    img = np.array(image.convert('RGB'))
    img = preprocess(image=img)['image']
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)[None, ...]  # B,C,H,W

    outputs = session.run(None, {'input': img})
    logits = outputs[0][0]

    # Softmax для вероятностей
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

# -----------------------------
# Интерфейс Gradio
# -----------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=len(LABELS))
)

if __name__ == "__main__":
    iface.launch()
