# 🖥️ Классификация компьютерных устройств

## 📊 Обзор проекта

Проект демонстрирует процесс fine-tuning двух предварительно обученных моделей (ResNet18 и EfficientNet-B0) на наборе данных изображений компьютерных устройств (клавиатуры, мыши, звуковые карты) с использованием transfer learning и аугментации данных.

**🏆 Лучшая модель:** EfficientNet-B0 с точностью **96.67%**

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Структура данных

```
data/raw/
├── train/
│   ├── keyboard/    (32 изображения)
│   ├── mouse/       (32 изображения)
│   └── soundcard/   (32 изображения)
└── val/
    ├── keyboard/    (10 изображений)
    ├── mouse/       (10 изображений)
    └── soundcard/   (10 изображений)
```

### 3. Обучение модели

```bash
# Обучение EfficientNet-B0 (лучшая модель)
python experiments/train.py --model_name efficientnet_b0 --epochs 12

# Или обучение ResNet18 для сравнения
python experiments/train.py --model_name resnet18 --epochs 12
```

### 4. Экспорт в ONNX

```bash
# Экспорт лучшей модели (EfficientNet-B0)
python export_onnx.py

# Или экспорт конкретной модели
python export_onnx.py --model_name resnet18

# С дополнительными параметрами
python export_onnx.py --model_name efficientnet_b0 --image_size 224 --models_dir models
```

### 5. Запуск веб-приложения

```bash
# Запуск простого Gradio приложения (рекомендуется)
python app/simple_gradio_app.py

# Или полное приложение с примерами
python app/gradio_app.py
```

Приложение будет доступно по адресу: http://127.0.0.1:7860

## 📈 Результаты экспериментов

| Модель | Семейство | Точность | Параметры | Время обучения |
|--------|-----------|----------|-----------|----------------|
| ResNet18 | ResNet | 93.33% | 11.2M | ~10 мин        |
| **EfficientNet-B0** | **EfficientNet** | **96.67%** | **4.0M** | **~15 мин**    |

## 🔧 Технические детали

### Оптимальные гиперпараметры (EfficientNet-B0):
- **Learning Rate:** 1e-3
- **Batch Size:** 16
- **Epochs:** 12
- **Weight Decay:** 1e-4
- **Optimizer:** AdamW
- **Scheduler:** StepLR (step_size=5, gamma=0.1)

### Стратегия замораживания:
1. **Эпохи 1-5:** Заморожен backbone, обучается только классификатор
2. **Эпоха 5:** Размораживание backbone с уменьшением LR в 10 раз
3. **Эпохи 6-12:** Обучение всей модели

### Аугментации данных:
- RandomResizedCrop (scale=0.8-1.0)
- HorizontalFlip (p=0.5)
- Rotate (limit=20°, p=0.3)
- ColorJitter (0.2, p=0.3)
- Нормализация ImageNet

## 📁 Структура проекта

```
fine-tuning/
├── app/
│   └── gradio_app.py          # Веб-приложение
├── data/
│   └── raw/                   # Данные
├── experiments/
│   ├── notebooks/
│   │   └── comprehensive_experiments.ipynb  # Jupyter ноутбук
│   ├── train.py               # Скрипт обучения
│   ├── configs.py             # Конфигурации
│   ├── datasets.py            # Обработка данных
│   └── utils.py               # Утилиты
├── models/
│   ├── best_efficientnet_b0.pth  # Веса лучшей модели
│   └── efficientnet_b0.onnx      # ONNX модель
└── requirements.txt
```

## 🎯 Использование

### Обучение модели:
```bash
python experiments/train.py --model_name efficientnet_b0 --epochs 12 --batch_size 16
```

### Экспорт в ONNX:
```bash
# Экспорт лучшей модели
python export_onnx.py

# Экспорт конкретной модели
python export_onnx.py --model_name resnet18
```

### Веб-приложение:
```bash
python app/gradio_app.py
```

## 📊 Анализ результатов

### Jupyter ноутбук:
```bash
jupyter notebook experiments/notebooks/comprehensive_experiments.ipynb
```

### Расширенные эксперименты:
```bash
# Подбор гиперпараметров
python experiments/hyperparameter_tuning.py

# Анализ стратегий замораживания
python experiments/freezing_strategies.py
```

## 🔍 Предобработка данных

**Важно:** Предобработка в приложении точно соответствует валидационной предобработке:

```python
# Валидационная предобработка (обучение)
A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Предобработка в приложении (инференс)
# Точно такая же!
```

## 🚀 Производительность

- **Точность:** 96.67% на валидационном наборе
- **Размер модели:** 4.0M параметров
- **Время инференса:** ~50ms на CPU
- **Размер ONNX:** ~15MB

## 📝 Лицензия

MIT License
