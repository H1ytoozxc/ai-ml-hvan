# Usage Guide - Project EvoArchitect v3

Практическое руководство по использованию автономного ИИ-агента для эволюции архитектур.

## 🚀 Быстрый старт

### 1. Проверка установки

```bash
# Запустите quick start тест
python quick_start.py
```

Этот скрипт проверит:
- ✅ Генерацию архитектур
- ✅ Создание PyTorch моделей
- ✅ Работу базы знаний
- ✅ Операции мутации

### 2. Первый запуск (быстрый тест)

```bash
# Запуск с минимальной популяцией для теста
python main.py --quick-test --device cuda
```

Ожидаемое время: ~30-60 минут  
Результат: 10 финальных моделей

### 3. Полный запуск

```bash
# Запуск с полной популяцией
python main.py --population-size 3000 --device cuda --output-dir ./my_results
```

Ожидаемое время: ~12-24 часа на RTX 4060  
Результат: Top-10 Pareto-оптимальных моделей

## 📊 Мониторинг процесса

### Weights & Biases (рекомендуется)

1. Создайте аккаунт на [wandb.ai](https://wandb.ai)

2. Авторизуйтесь:
```bash
wandb login
```

3. Запустите с WandB:
```bash
python main.py --device cuda
```

4. Открыть dashboard: автоматическая ссылка в консоли

### Локальный мониторинг

Результаты сохраняются в `./evo_runs/`:
- `knowledge_base.db` - база всех архитектур
- `pareto_stage2.png` - Pareto frontier после Stage 2
- `pareto_final.png` - финальный Pareto frontier
- `top_model_X.json` - JSON описания топ моделей
- `summary.json` - итоговая статистика

## 🎯 Кейсы использования

### Кейс 1: Поиск SOTA для CIFAR-100

**Цель**: Максимальная точность на CIFAR-100

**Конфигурация** (`config_sota.yaml`):
```yaml
initial_population_size: 5000

stage2_config:
  selection_criteria:
    novelty_score_weight: 0.1  # Меньше фокуса на новизну
    diversity_weight: 0.1
    # Больше фокуса на точность

stage3_config:
  epochs: 150  # Дольше обучаем
```

**Запуск**:
```bash
python main.py --config config_sota.yaml --device cuda
```

### Кейс 2: Исследование novel архитектур

**Цель**: Найти максимально новые, необычные архитектуры

**Конфигурация** (`config_novelty.yaml`):
```yaml
initial_population_size: 4000

stage1_config:
  selection_criteria:
    novelty_score_weight: 0.5  # Высокий приоритет новизне
    diversity_weight: 0.4

meta_optimization:
  reward_signal: "novelty_gain * 2.0 + performance_improvement"
```

**Запуск**:
```bash
python main.py --config config_novelty.yaml
```

### Кейс 3: Эффективные модели (для edge devices)

**Цель**: Легкие модели с хорошей точностью

**Конфигурация** (`config_efficient.yaml`):
```yaml
search_space:
  max_depth: 10  # Ограничиваем глубину
  base_blocks:  # Только эффективные блоки
    - "Conv3x3"
    - "ResidualBlock"
    - "MLP_Mixer_Block"

stage3_config:
  selection_criteria:
    objectives:
      - "accuracy"
      - "compute_efficiency"  # Приоритет на эффективность
      - "parameter_count"
      - "inference_latency_ms"
```

**Запуск**:
```bash
python main.py --config config_efficient.yaml
```

### Кейс 4: Робастные модели

**Цель**: Модели устойчивые к искажениям

**Конфигурация** (`config_robust.yaml`):
```yaml
stage2_config:
  robustness_metric:
    severities: [1, 2, 3, 4, 5]  # Все уровни искажений
  
  selection_criteria:
    objectives:
      - "accuracy"
      - "robustness_score"  # Высокий вес робастности

stage3_config:
  use_augmentation_policy: "AutoAugment + RandAugment + Mixup"
```

## 🔧 Настройка под вашу систему

### RTX 4060 (8GB VRAM) + i5-12100F

**Оптимальная конфигурация**:
```yaml
initial_population_size: 1500

runtime:
  compute_resources:
    num_parallel_trials: 2  # 2 модели параллельно
    gpu_memory_limit_gb: 6  # Резерв для системы

stage1_config:
  input_candidates: 1500

stage2_config:
  epochs: 15  # Немного меньше для скорости
```

**Запуск**:
```bash
python main.py --population-size 1500 --device cuda
```

**Ожидаемое время**: ~8-12 часов

### CPU-only система

```bash
python main.py --device cpu --population-size 100 --quick-test
```

⚠️ **Внимание**: На CPU процесс будет очень медленным (дни вместо часов).

### Многопроходные GPU (Tesla V100, A100, etc.)

```yaml
initial_population_size: 10000

runtime:
  compute_resources:
    num_parallel_trials: 8
    gpu_memory_limit_gb: 30

stage3_config:
  epochs: 200
```

## 📈 Анализ результатов

### Просмотр топ моделей

```python
import json

# Загрузить топ модель
with open('./evo_runs/top_model_1.json', 'r') as f:
    model_config = json.load(f)

print(f"Best accuracy: {model_config['metrics']['val_accuracy']}")
print(f"Architecture: {len(model_config['blocks'])} blocks")
print(f"Optimizer: {model_config['optimizer']}")
```

### Экспорт в PyTorch

```python
from src.search_space.search_space import ArchitectureGenome
from src.search_space.architecture_generator import ArchitectureBuilder
import json

# Загрузить genome
with open('./evo_runs/top_model_1.json', 'r') as f:
    genome_dict = json.load(f)

genome = ArchitectureGenome.from_dict(genome_dict)

# Построить модель
model = ArchitectureBuilder.build(genome, num_classes=100)

# Сохранить веса (после обучения)
# torch.save(model.state_dict(), 'best_model.pth')
```

### Анализ базы знаний

```python
from src.knowledge_base.database import KnowledgeBase

with KnowledgeBase('./evo_runs/knowledge_base.db') as kb:
    # Статистика
    stats = kb.get_statistics()
    print(f"Total architectures: {stats['total_architectures']}")
    
    # Топ модели по точности
    top_models = kb.get_top_architectures(
        stage="Stage3_Full_Validation",
        metric="val_accuracy",
        limit=10
    )
    
    for i, genome in enumerate(top_models):
        print(f"{i+1}. Accuracy: {genome.metrics.get('val_accuracy', 0):.2f}%")
```

## 🐛 Решение проблем

### CUDA Out of Memory

**Решение 1**: Уменьшите parallel trials
```yaml
runtime:
  compute_resources:
    num_parallel_trials: 1
```

**Решение 2**: Уменьшите population size
```bash
python main.py --population-size 500
```

**Решение 3**: Уменьшите max depth
```yaml
search_space:
  max_depth: 12
```

### Слишком долгое выполнение

**Решение 1**: Уменьшите epochs
```yaml
stage2_config:
  epochs: 10
stage3_config:
  epochs: 50
```

**Решение 2**: Используйте subset данных
```yaml
stage2_config:
  datasets:
    - name: "CIFAR-100"
      subset_percent: 30  # Вместо 50%
```

### WandB authentication errors

```bash
# Отключите WandB
python main.py --config config_no_wandb.yaml
```

В `config_no_wandb.yaml`:
```yaml
logging:
  dashboard_provider: null  # Отключить WandB
```

### Import errors

```bash
# Переустановите зависимости
pip install -r requirements.txt --upgrade

# Проверьте импорты
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

## 📚 Дополнительные примеры

### Кастомный датасет

```python
# В src/data/data_loaders.py добавьте:

@staticmethod
def get_my_dataset(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    trainset = MyCustomDataset(root='./data', train=True, transform=transform)
    testset = MyCustomDataset(root='./data', train=False, transform=transform)
    
    return {
        "train": DataLoader(trainset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(testset, batch_size=batch_size, shuffle=False)
    }
```

### Кастомная метрика

```python
# В src/evaluation/metrics.py добавьте:

class MyCustomMetric:
    def evaluate(self, model, dataloader):
        # Ваша логика
        return {"my_metric": score}
```

### Кастомная мутация

```python
# В src/meta_optimization/mutations.py:

class MyCustomMutation(MutationOperator):
    def __call__(self, genome, search_space):
        mutated = genome.clone()
        # Ваша логика мутации
        return mutated

# Добавьте в ALL_MUTATIONS:
ALL_MUTATIONS["my_mutation"] = MyCustomMutation
```

## 🎓 Обучающие материалы

### Понимание Pareto frontier

Pareto frontier - это набор решений, где улучшение одной метрики приводит к ухудшению другой.

Пример:
- Модель A: 95% accuracy, 10M параметров
- Модель B: 93% accuracy, 2M параметров
- Обе на Pareto frontier (нельзя улучшить обе метрики одновременно)

### Novelty Search

Novelty search ищет архитектуры, которые отличаются от уже найденных:
- **Architectural novelty**: Разница в структуре графа
- **Behavioral novelty**: Разница в активациях на данных

Комбинация помогает избежать локальных оптимумов.

### Meta-Optimization

RL контроллер учится выбирать лучшие мутации:
- Наблюдает состояние популяции
- Выбирает тип мутации (action)
- Получает reward от улучшения популяции
- Обновляет policy для лучших выборов

## 💡 Советы

1. **Начните с quick-test** перед полным запуском
2. **Используйте WandB** для мониторинга
3. **Сохраняйте чекпоинты** регулярно
4. **Экспериментируйте с весами** novelty/accuracy
5. **Анализируйте базу знаний** после каждого запуска
6. **Комбинируйте топ модели** для ансамблей

## 🔗 Полезные ссылки

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [WandB Guides](https://docs.wandb.ai/)
- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- [NAS Survey](https://arxiv.org/abs/1808.05377)

---

**Нужна помощь?** Откройте Issue на GitHub!
