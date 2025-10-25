# experiments/hyperparameter_tuning.py
"""
Расширенный подбор гиперпараметров для моделей ResNet18 и EfficientNet-B0
"""

import os
import json
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score

from datasets import ImageFolderSimple, get_transforms
from utils import set_seed


class HyperparameterTuner:
    """Класс для подбора гиперпараметров"""
    
    def __init__(self, data_dir: str, train_subdir: str, val_subdir: str, 
                 out_dir: str, logs_dir: str, device: str = "cuda"):
        self.data_dir = data_dir
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.out_dir = out_dir
        self.logs_dir = logs_dir
        self.device = device
        
        # Создание директорий
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    
    def freeze_backbone(self, model, freeze: bool = True):
        """Замораживание/размораживание backbone модели"""
        for name, param in model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True  # Классификатор всегда обучается
            else:
                param.requires_grad = not freeze
    
    def train_single_config(self, model_name: str, config: Dict, 
                           train_loader: DataLoader, val_loader: DataLoader,
                           epochs: int = 8) -> Dict:
        """Обучение модели с одной конфигурацией гиперпараметров"""
        
        # Создание модели
        model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=config['num_classes']
        )
        model.to(self.device)
        
        # Замораживание backbone
        if config['freeze_backbone']:
            self.freeze_backbone(model, freeze=True)
        
        # Оптимизатор
        if config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                momentum=0.9
            )
        
        # Планировщик
        if config['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config['step_size'], gamma=config['gamma']
            )
        elif config['scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=2, factor=0.5
            )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        history = {'val_acc': []}
        
        # Цикл обучения
        for epoch in range(epochs):
            # Обучение
            model.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Валидация
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = model(imgs)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Обновление планировщика
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Размораживание backbone
            if (config['freeze_backbone'] and 
                epoch == config.get('unfreeze_at_epoch', 3)):
                self.freeze_backbone(model, freeze=False)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
        
        return {
            'best_val_acc': best_val_acc,
            'history': history,
            'config': config
        }
    
    def grid_search(self, model_name: str, param_grid: Dict, 
                   train_loader: DataLoader, val_loader: DataLoader,
                   num_classes: int, epochs: int = 8) -> List[Dict]:
        """Поиск по сетке гиперпараметров"""
        
        # Генерация всех комбинаций параметров
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        results = []
        total_combinations = len(param_combinations)
        
        print(f"Начинаем поиск по сетке для {model_name}")
        print(f"Всего комбинаций: {total_combinations}")
        
        for i, combination in enumerate(param_combinations):
            config = dict(zip(param_names, combination))
            config['num_classes'] = num_classes
            
            print(f"\n[{i+1}/{total_combinations}] Тестируем конфигурацию:")
            for key, value in config.items():
                if key != 'num_classes':
                    print(f"  {key}: {value}")
            
            try:
                result = self.train_single_config(
                    model_name, config, train_loader, val_loader, epochs
                )
                result['model_name'] = model_name
                results.append(result)
                
                print(f"  Результат: {result['best_val_acc']:.4f}")
                
            except Exception as e:
                print(f"  Ошибка: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict], model_name: str):
        """Сохранение результатов поиска"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сортировка по точности
        results.sort(key=lambda x: x['best_val_acc'], reverse=True)
        
        # Сохранение в JSON
        results_path = os.path.join(self.logs_dir, f"hyperparameter_search_{model_name}_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Сохранение лучших результатов
        best_results = results[:5]  # Топ-5
        best_path = os.path.join(self.logs_dir, f"best_configs_{model_name}_{timestamp}.json")
        with open(best_path, 'w') as f:
            json.dump(best_results, f, indent=2)
        
        print(f"\nРезультаты сохранены:")
        print(f"  Все результаты: {results_path}")
        print(f"  Лучшие конфигурации: {best_path}")
        
        return results


def run_hyperparameter_tuning():
    """Запуск подбора гиперпараметров для обеих моделей"""
    
    # Настройки
    data_dir = "data/raw"
    train_subdir = "train"
    val_subdir = "val"
    out_dir = "models"
    logs_dir = "experiments/logs"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_seed(42)
    
    # Подготовка данных
    train_t, val_t = get_transforms(224)
    train_ds = ImageFolderSimple(os.path.join(data_dir, train_subdir), transform=train_t)
    val_ds = ImageFolderSimple(os.path.join(data_dir, val_subdir), 
                               transform=val_t, classes=list(train_ds.class_to_idx.keys()))
    
    num_classes = len(train_ds.class_to_idx)
    class_names = list(train_ds.class_to_idx.keys())
    
    print(f"Классы: {class_names}")
    print(f"Количество классов: {num_classes}")
    print(f"Тренировочных образцов: {len(train_ds)}")
    print(f"Валидационных образцов: {len(val_ds)}")
    
    # DataLoader'ы
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    # Создание тюнера
    tuner = HyperparameterTuner(data_dir, train_subdir, val_subdir, out_dir, logs_dir, device)
    
    # Сетки гиперпараметров для каждой модели
    param_grids = {
        'resnet18': {
            'lr': [1e-3, 5e-4, 1e-4],
            'weight_decay': [1e-4, 1e-3],
            'optimizer': ['AdamW', 'SGD'],
            'scheduler': ['StepLR', 'CosineAnnealingLR'],
            'freeze_backbone': [True, False],
            'step_size': [3, 5],
            'gamma': [0.1, 0.5]
        },
        'efficientnet_b0': {
            'lr': [1e-3, 5e-4, 1e-4],
            'weight_decay': [1e-4, 1e-3],
            'optimizer': ['AdamW', 'SGD'],
            'scheduler': ['StepLR', 'CosineAnnealingLR'],
            'freeze_backbone': [True, False],
            'step_size': [3, 5],
            'gamma': [0.1, 0.5]
        }
    }
    
    all_results = {}
    
    # Подбор гиперпараметров для каждой модели
    for model_name, param_grid in param_grids.items():
        print(f"\n{'='*60}")
        print(f"ПОДБОР ГИПЕРПАРАМЕТРОВ ДЛЯ {model_name.upper()}")
        print(f"{'='*60}")
        
        results = tuner.grid_search(
            model_name, param_grid, train_loader, val_loader, num_classes, epochs=6
        )
        
        tuner.save_results(results, model_name)
        all_results[model_name] = results
        
        # Вывод лучших результатов
        print(f"\n🏆 ТОП-5 КОНФИГУРАЦИЙ ДЛЯ {model_name.upper()}:")
        for i, result in enumerate(results[:5]):
            print(f"\n{i+1}. Точность: {result['best_val_acc']:.4f}")
            config = result['config']
            for key, value in config.items():
                if key != 'num_classes':
                    print(f"   {key}: {value}")
    
    # Сравнение лучших результатов
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ ЛУЧШИХ РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        if results:
            best_result = results[0]
            print(f"{model_name:15s}: {best_result['best_val_acc']:.4f}")
    
    return all_results


if __name__ == "__main__":
    results = run_hyperparameter_tuning()
