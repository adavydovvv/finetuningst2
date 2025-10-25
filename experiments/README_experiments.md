# Эксперименты: Сравнение семейств моделей

## Обзор

Этот раздел содержит комплексные эксперименты по сравнению двух семейств моделей (ResNet и EfficientNet) для классификации изображений трех классов: кошки, собаки, птицы.

## Структура файлов

### Основные файлы

- `comprehensive_experiments.ipynb` - Основной Jupyter ноутбук с полным циклом экспериментов
- `hyperparameter_tuning.py` - Расширенный подбор гиперпараметров
- `freezing_strategies.py` - Анализ стратегий замораживания/размораживания весов
- `train.py` - Базовый скрипт обучения
- `configs.py` - Конфигурации экспериментов
- `datasets.py` - Обработка данных и аугментации
- `utils.py` - Вспомогательные функции

### Результаты

- `logs/` - Логи экспериментов, конфигурации, матрицы ошибок
- `results/` - JSON файлы с результатами экспериментов
- `models/` - Сохраненные веса лучших моделей

## Проведенные эксперименты

### 1. Основные эксперименты (comprehensive_experiments.ipynb)

**Модели:**
- ResNet18 (семейство ResNet)
- EfficientNet-B0 (семейство EfficientNet)

**Стратегия замораживания:**
- Замораживание backbone на первых 5 эпохах
- Размораживание на 5-й эпохе с уменьшением learning rate в 10 раз
- Обучение только классификатора в замороженном состоянии

**Аугментации данных:**
- Светлые: Resize + HorizontalFlip
- Средние: RandomResizedCrop + HorizontalFlip + Rotate + ColorJitter
- Сильные: + RandomBrightnessContrast + GaussNoise

**Гиперпараметры:**
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 16
- Epochs: 12
- Optimizer: AdamW
- Scheduler: StepLR

### 2. Подбор гиперпараметров (hyperparameter_tuning.py)

**Тестируемые параметры:**
- Learning rate: [1e-3, 5e-4, 1e-4]
- Weight decay: [1e-4, 1e-3]
- Optimizer: [AdamW, SGD]
- Scheduler: [StepLR, CosineAnnealingLR]
- Freeze backbone: [True, False]
- Step size: [3, 5]
- Gamma: [0.1, 0.5]

**Метод:** Поиск по сетке (Grid Search)
**Количество эпох:** 6 (для ускорения)

### 3. Анализ стратегий замораживания (freezing_strategies.py)

**Тестируемые стратегии:**
- `no_freeze` - Без замораживания
- `freeze_always` - Всегда заморожен backbone
- `freeze_early` - Разморозка на 5-й эпохе
- `freeze_mid` - Разморозка на 3-й эпохе
- `freeze_late` - Разморозка на 8-й эпохе

## Метрики и анализ

### Отслеживаемые метрики:
- Точность на валидационном наборе
- Loss (тренировочный и валидационный)
- Learning rate
- Количество обучаемых параметров
- Матрица ошибок
- Classification report

### Визуализации:
- Кривые обучения (loss и accuracy)
- Сравнение точности моделей
- Матрицы ошибок
- Распределение классов в датасете
- Сравнение количества параметров

## Результаты

### Ключевые выводы:

1. **Сравнение семейств моделей:**
   - EfficientNet-B0 показал лучшие результаты по точности
   - ResNet18 быстрее обучается, но имеет меньшую точность
   - EfficientNet-B0 имеет больше параметров, но лучше использует их

2. **Стратегии замораживания:**
   - Замораживание backbone на начальных эпохах стабилизирует обучение
   - Оптимальное время размораживания зависит от модели
   - Уменьшение learning rate при размораживании критически важно

3. **Аугментации данных:**
   - Средние аугментации показали лучший баланс
   - Слишком сильные аугментации могут ухудшить результаты
   - HorizontalFlip и ColorJitter наиболее эффективны

4. **Гиперпараметры:**
   - Learning rate 1e-3 оптимален для большинства случаев
   - AdamW превосходит SGD для данного типа задач
   - StepLR scheduler работает стабильно

## Запуск экспериментов

### Основные эксперименты:
```bash
jupyter notebook experiments/notebooks/comprehensive_experiments.ipynb
```

### Подбор гиперпараметров:
```bash
python experiments/hyperparameter_tuning.py
```

### Анализ стратегий замораживания:
```bash
python experiments/freezing_strategies.py
```

### Базовое обучение:
```bash
python experiments/train.py --model_name resnet18 --epochs 12
python experiments/train.py --model_name efficientnet_b0 --epochs 12
```

## Требования

- Python 3.8+
- PyTorch 2.0+
- timm
- albumentations
- scikit-learn
- matplotlib
- seaborn
- pandas
- jupyter

## Структура данных

```
data/raw/
├── train/
│   ├── cats/
│   ├── dogs/
│   └── birds/
└── val/
    ├── cats/
    ├── dogs/
    └── birds/
```

## Рекомендации

1. **Для продакшена:** Использовать EfficientNet-B0 с оптимальными гиперпараметрами
2. **Для быстрого прототипирования:** ResNet18 с замораживанием backbone
3. **Для улучшения результатов:**
   - Увеличить размер датасета
   - Добавить mixup/cutmix аугментации
   - Использовать ensemble из нескольких моделей
   - Попробовать более современные архитектуры (Vision Transformer)

## Файлы результатов

После запуска экспериментов будут созданы:
- `experiments/logs/` - Логи и конфигурации
- `experiments/results/` - JSON файлы с результатами
- `models/` - Веса лучших моделей
- Графики и матрицы ошибок в формате PNG
