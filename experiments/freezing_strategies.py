# experiments/freezing_strategies.py
"""
Анализ различных стратегий замораживания/размораживания весов
"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from datasets import ImageFolderSimple, get_transforms
from utils import set_seed


class FreezingStrategyAnalyzer:
    """Анализатор стратегий замораживания"""
    
    def __init__(self, data_dir: str, train_subdir: str, val_subdir: str, 
                 out_dir: str, logs_dir: str, device: str = "cuda"):
        self.data_dir = data_dir
        self.train_subdir = train_subdir
        self.val_subdir = val_subdir
        self.out_dir = out_dir
        self.logs_dir = logs_dir
        self.device = device
        
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
    
    def freeze_backbone(self, model, freeze: bool = True):
        """Замораживание/размораживание backbone модели"""
        for name, param in model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True  # Классификатор всегда обучается
            else:
                param.requires_grad = not freeze
    
    def count_parameters(self, model):
        """Подсчет количества параметров"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def train_with_strategy(self, model_name: str, strategy: str, 
                           train_loader: DataLoader, val_loader: DataLoader,
                           num_classes: int, epochs: int = 12) -> Dict:
        """Обучение с определенной стратегией замораживания"""
        
        # Создание модели
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        model.to(self.device)
        
        # Подсчет параметров до замораживания
        total_params, initial_trainable = self.count_parameters(model)
        
        # Применение стратегии замораживания
        if strategy == "no_freeze":
            # Без замораживания - обучаем все параметры
            pass
        elif strategy == "freeze_always":
            # Всегда заморожен backbone
            self.freeze_backbone(model, freeze=True)
        elif strategy == "freeze_early":
            # Заморожен первые 5 эпох, потом разморожен
            self.freeze_backbone(model, freeze=True)
        elif strategy == "freeze_mid":
            # Заморожен первые 3 эпохи, потом разморожен
            self.freeze_backbone(model, freeze=True)
        elif strategy == "freeze_late":
            # Заморожен первые 8 эпох, потом разморожен
            self.freeze_backbone(model, freeze=True)
        
        # Подсчет параметров после замораживания
        _, frozen_trainable = self.count_parameters(model)
        
        # Оптимизатор
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        # История обучения
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'trainable_params': []
        }
        
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"\nСтратегия: {strategy}")
        print(f"Всего параметров: {total_params:,}")
        print(f"Обучаемых параметров: {frozen_trainable:,}")
        
        # Цикл обучения
        for epoch in range(1, epochs + 1):
            # Обучение
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Валидация
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Сохранение истории
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['trainable_params'].append(self.count_parameters(model)[1])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
            
            print(f"Epoch {epoch:2d}: Train Loss={train_loss:.4f} | "
                  f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")
            
            # Размораживание по стратегии
            if strategy == "freeze_early" and epoch == 5:
                self.freeze_backbone(model, freeze=False)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                print(f"🔓 Разморожен backbone на эпохе {epoch}")
            elif strategy == "freeze_mid" and epoch == 3:
                self.freeze_backbone(model, freeze=False)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                print(f"🔓 Разморожен backbone на эпохе {epoch}")
            elif strategy == "freeze_late" and epoch == 8:
                self.freeze_backbone(model, freeze=False)
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1
                print(f"🔓 Разморожен backbone на эпохе {epoch}")
            
            scheduler.step()
        
        return {
            'strategy': strategy,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_params': total_params,
            'initial_trainable': initial_trainable,
            'final_trainable': self.count_parameters(model)[1],
            'history': history
        }
    
    def compare_strategies(self, model_name: str, strategies: List[str],
                          train_loader: DataLoader, val_loader: DataLoader,
                          num_classes: int, epochs: int = 12) -> Dict:
        """Сравнение различных стратегий замораживания"""
        
        results = {}
        
        print(f"\n{'='*60}")
        print(f"СРАВНЕНИЕ СТРАТЕГИЙ ЗАМОРАЖИВАНИЯ ДЛЯ {model_name.upper()}")
        print(f"{'='*60}")
        
        for strategy in strategies:
            print(f"\n{'='*40}")
            print(f"Тестируем стратегию: {strategy}")
            print(f"{'='*40}")
            
            result = self.train_with_strategy(
                model_name, strategy, train_loader, val_loader, num_classes, epochs
            )
            results[strategy] = result
        
        return results
    
    def visualize_strategies(self, results: Dict, model_name: str):
        """Визуализация результатов сравнения стратегий"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # График точности
        for strategy, result in results.items():
            axes[0, 0].plot(result['history']['val_acc'], 
                           label=f"{strategy} (best: {result['best_val_acc']:.4f})", 
                           marker='o')
        axes[0, 0].set_title('Валидационная точность по стратегиям')
        axes[0, 0].set_xlabel('Эпоха')
        axes[0, 0].set_ylabel('Точность')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # График loss
        for strategy, result in results.items():
            axes[0, 1].plot(result['history']['train_loss'], 
                           label=f"{strategy} (train)", linestyle='--')
            axes[0, 1].plot(result['history']['val_loss'], 
                           label=f"{strategy} (val)", linestyle='-')
        axes[0, 1].set_title('Loss по стратегиям')
        axes[0, 1].set_xlabel('Эпоха')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Сравнение лучших результатов
        strategies = list(results.keys())
        best_accs = [results[s]['best_val_acc'] for s in strategies]
        axes[1, 0].bar(strategies, best_accs)
        axes[1, 0].set_title('Лучшая валидационная точность')
        axes[1, 0].set_ylabel('Точность')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Количество обучаемых параметров
        initial_trainable = [results[s]['initial_trainable'] for s in strategies]
        final_trainable = [results[s]['final_trainable'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, initial_trainable, width, label='Начальные обучаемые')
        axes[1, 1].bar(x + width/2, final_trainable, width, label='Финальные обучаемые')
        axes[1, 1].set_title('Количество обучаемых параметров')
        axes[1, 1].set_ylabel('Количество параметров')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(strategies, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Сохранение графика
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.logs_dir, f"freezing_strategies_{model_name}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"График сохранен: {plot_path}")
    
    def save_results(self, results: Dict, model_name: str):
        """Сохранение результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сортировка по точности
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
        
        results_path = os.path.join(self.logs_dir, f"freezing_strategies_{model_name}_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(dict(sorted_results), f, indent=2)
        
        print(f"\nРезультаты сохранены: {results_path}")
        
        # Вывод лучших стратегий
        print(f"\n🏆 РЕЙТИНГ СТРАТЕГИЙ ДЛЯ {model_name.upper()}:")
        for i, (strategy, result) in enumerate(sorted_results):
            print(f"{i+1}. {strategy:15s}: {result['best_val_acc']:.4f} "
                  f"(эпоха {result['best_epoch']})")


def run_freezing_analysis():
    """Запуск анализа стратегий замораживания"""
    
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
    
    # DataLoader'ы
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    # Создание анализатора
    analyzer = FreezingStrategyAnalyzer(data_dir, train_subdir, val_subdir, out_dir, logs_dir, device)
    
    # Стратегии для тестирования
    strategies = [
        "no_freeze",      # Без замораживания
        "freeze_always",  # Всегда заморожен
        "freeze_early",   # Разморозка на 5 эпохе
        "freeze_mid",     # Разморозка на 3 эпохе
        "freeze_late"     # Разморозка на 8 эпохе
    ]
    
    # Модели для тестирования
    models = ['resnet18', 'efficientnet_b0']
    
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"АНАЛИЗ СТРАТЕГИЙ ЗАМОРАЖИВАНИЯ ДЛЯ {model_name.upper()}")
        print(f"{'='*80}")
        
        results = analyzer.compare_strategies(
            model_name, strategies, train_loader, val_loader, num_classes, epochs=10
        )
        
        analyzer.visualize_strategies(results, model_name)
        analyzer.save_results(results, model_name)
        
        all_results[model_name] = results
    
    # Общий анализ
    print(f"\n{'='*80}")
    print("ОБЩИЙ АНАЛИЗ СТРАТЕГИЙ ЗАМОРАЖИВАНИЯ")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
        for strategy, result in sorted_results:
            print(f"  {strategy:15s}: {result['best_val_acc']:.4f}")
    
    return all_results


if __name__ == "__main__":
    results = run_freezing_analysis()
