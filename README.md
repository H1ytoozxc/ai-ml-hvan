# Project EvoArchitect v3 🧬🏗️

**Автономный ИИ-агент для открытия, эволюции и валидации нового поколения нейронных архитектур**

Полностью автономная система, использующая адаптивный многоступенчатый пайплайн, мета-обучение, novelty-driven search и многокритериальный отбор на основе фронта Парето для обнаружения новых SOTA подходов.

## 🎯 Основные возможности

- **3-ступенчатый адаптивный пайплайн**: Массовый скрининг → Уточнение → Полная валидация
- **Novelty-Driven Search**: Приоритет на неизведанные, потенциально прорывные области
- **Pareto-оптимальность**: Поиск фронта недоминируемых решений по множеству критериев
- **Meta-обучение**: RL-контроллер для самосовершенствования стратегии поиска
- **Богатое пространство поиска**: CNN, Transformers, MLP-Mixer, Spiking Networks, Graph Convolutions
- **Многокритериальная оценка**: Точность, новизна, робастность, эффективность, обобщающая способность
- **База знаний**: Persistent storage всех оцененных архитектур для ускорения поиска

## 📋 Архитектура проекта

```
Project_EvoArchitect_v3/
├── main.py                          # Точка входа
├── requirements.txt                 # Зависимости
├── README.md                        # Документация
├── config.yaml                      # Конфигурация (опционально)
│
├── src/
│   ├── config.py                    # Конфигурация системы
│   ├── orchestrator.py              # Главный оркестратор
│   │
│   ├── search_space/
│   │   ├── blocks.py                # Строительные блоки архитектур
│   │   ├── search_space.py          # Определение пространства поиска
│   │   └── architecture_generator.py # Генерация PyTorch моделей
│   │
│   ├── training/
│   │   └── trainer.py               # Обучение и оценка моделей
│   │
│   ├── evaluation/
│   │   └── metrics.py               # Сбор метрик
│   │
│   ├── selection/
│   │   ├── novelty_metrics.py       # Метрики новизны
│   │   └── pareto_frontier.py       # Отбор по Парето
│   │
│   ├── meta_optimization/
│   │   ├── mutations.py             # Операции мутации
│   │   └── rl_controller.py         # RL-контроллер стратегии
│   │
│   ├── knowledge_base/
│   │   └── database.py              # База знаний (SQLite)
│   │
│   └── data/
│       └── data_loaders.py          # Загрузка датасетов
│
└── evo_runs/                        # Результаты экспериментов
    ├── knowledge_base.db            # База знаний
    ├── top_model_1.json             # Топ модели
    ├── pareto_final.png             # Визуализация фронта Парето
    └── summary.json                 # Итоговая статистика
```

## 🚀 Быстрый старт

### Установка

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/Project_EvoArchitect_v3.git
cd Project_EvoArchitect_v3

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Валидация системы

```bash
# Полная валидация (рекомендуется перед первым запуском)
python validate_system.py

# Быстрая проверка импортов
python test_imports.py

# Функциональный тест
python quick_start.py
```

### Базовый запуск

```bash
# Запуск с настройками по умолчанию
python main.py

# Запуск с кастомной конфигурацией
python main.py --config config.yaml

# Быстрый тест (маленькая популяция)
python main.py --quick-test

# Указание устройства
python main.py --device cuda

# Кастомный размер популяции
python main.py --population-size 1000
```

### Параметры командной строки

- `--config PATH`: Путь к YAML конфигурации
- `--population-size N`: Размер начальной популяции
- `--device {cuda,cpu}`: Устройство для обучения
- `--output-dir PATH`: Директория для результатов
- `--seed N`: Random seed для воспроизводимости
- `--quick-test`: Быстрый тест с минимальными настройками

## 🎓 Концепции и алгоритмы

### Трехступенчатый пайплайн

#### Stage 1: Proxy Evaluation (Массовый скрининг)
- **Кандидаты**: 3000+ архитектур
- **Данные**: CIFAR-10, 10% подвыборка
- **Обучение**: 1 эпоха с weight sharing
- **Метрики**: proxy_accuracy, parameter_count, estimated_FLOPs
- **Суррогатная модель**: GNN для предсказания финальной точности
- **Отбор**: Top 15% (250-600 кандидатов) на основе composite score

#### Stage 2: Refinement Training (Уточнение)
- **Кандидаты**: Выжившие из Stage 1
- **Данные**: CIFAR-100, 50% подвыборка
- **Обучение**: 20 эпох с early stopping
- **Метрики**: accuracy, robustness, novelty, learning_curve_slope
- **Новизна**: Architectural + Behavioral distance
- **Отбор**: Pareto frontier (100 кандидатов)

#### Stage 3: Full Validation (Финальная оценка)
- **Кандидаты**: Pareto front из Stage 2
- **Данные**: CIFAR-100, ImageNet subset, SVHN, TinyImageNet
- **Обучение**: 100 эпох, полная валидация
- **Метрики**: Все метрики + cross-dataset generalization
- **Отбор**: Final Pareto front (Top-10 моделей)

### Пространство поиска

**Блоки архитектур:**
- Conv3x3: Стандартные сверточные слои
- ResidualBlock: Residual connections (ResNet-style)
- TransformerEncoderBlock: Vision Transformer блоки
- MLP_Mixer_Block: MLP-Mixer архитектура
- SpikingNeuronLayer: Spiking Neural Networks (LIF neurons)
- GraphConv: Graph convolution layers
- HyperNetworkBlock: Meta-learning через HyperNetworks

**Гиперпараметры:**
- Activations: ReLU, GELU, Swish, Mish, SiLU, Tanh
- Normalizations: BatchNorm, LayerNorm, GroupNorm, InstanceNorm
- Optimizers: AdamW, LAMB, SGD+Momentum, RAdam, Adafactor
- LR Schedulers: CosineAnnealing, OneCycleLR, ReduceLROnPlateau
- Augmentations: AutoAugment, RandAugment, Mixup, CutMix, Cutout

### Novelty Metrics

**Architectural Novelty:**
- Graph Edit Distance между архитектурами
- Diversity score относительно популяции
- k-nearest neighbor distance

**Behavioral Novelty:**
- Activation profile extraction
- Cosine distance между behavioral signatures
- Feature map statistics

**Combined Score:**
```python
novelty = 0.4 * architectural_novelty + 0.6 * behavioral_novelty
```

### Pareto Frontier Selection

Multi-objective optimization по критериям:
- **Accuracy**: Точность классификации
- **Novelty**: Степень новизны архитектуры
- **Robustness**: Устойчивость к искажениям
- **Efficiency**: Параметры/FLOPs/Latency
- **Generalization**: Gap между train и validation

Использует NSGA-II алгоритм с crowding distance для поддержания разнообразия.

### Meta-Optimization (RL Controller)

**REINFORCE Controller:**
- Учится выбирать лучшие стратегии мутации
- State: population statistics (accuracy, diversity, novelty)
- Action: выбор типа мутации
- Reward: улучшение Pareto frontier + novelty gain

**Мутации:**
- add_block: Добавить новый блок
- remove_block: Удалить блок
- mutate_block_type: Изменить тип блока
- change_activation: Изменить функцию активации
- adjust_dropout: Скорректировать dropout
- swap_optimizer: Сменить оптимизатор
- adjust_lr_schedule: Изменить LR schedule
- meta_block_insert: Добавить meta-learning компонент
- augment_strategy_mutate: Изменить augmentation

## 📊 Метрики и оценка

### Performance Metrics
- **Accuracy**: Точность на валидации
- **Loss**: Cross-entropy или другая loss function
- **Convergence Speed**: Скорость сходимости (epochs)
- **Learning Curve Slope**: Тренд обучения

### Efficiency Metrics
- **Parameter Count**: Число параметров модели
- **FLOPs**: Floating point operations
- **Inference Latency**: Время inference (ms)
- **Memory Usage**: GPU memory consumption

### Robustness Metrics
- **Corruption Benchmark**: CIFAR-C style corruptions
  - Gaussian noise, shot noise, impulse noise
  - Defocus blur, motion blur, zoom blur
  - Brightness, contrast, JPEG compression
- **Adversarial Robustness** (опционально)

### Generalization Metrics
- **Generalization Gap**: train_acc - val_acc
- **Cross-dataset Transfer**: Performance на других датасетах

## 💾 База знаний

SQLite база данных хранит:
- **Architectures**: Все оцененные геномы
- **Evaluations**: Результаты на каждом stage
- **Novelty Scores**: Метрики новизны
- **Pareto Fronts**: История фронтов Парето
- **Meta Stats**: Статистика мутаций для обучения

Позволяет:
- Избегать переоценки идентичных архитектур
- Анализировать эволюцию популяции
- Transfer learning от предыдущих запусков
- Экспорт топ моделей

## 🔧 Конфигурация

Создайте `config.yaml` для кастомной настройки:

```yaml
project_name: "My_Custom_Evolution"
initial_population_size: 5000
random_seed: 42

search_space:
  base_blocks:
    - Conv3x3
    - ResidualBlock
    - TransformerEncoderBlock
  max_depth: 25
  min_depth: 5

runtime:
  compute_resources:
    device: "cuda"
    num_parallel_trials: 4

logging:
  wandb_project: "my-evo-project"
  artifact_save_path: "./my_results"
```

## 📈 Мониторинг и визуализация

### Weights & Biases Integration

Автоматический логинг:
- Population statistics по generation
- Pareto frontier evolution
- Best/mean/std метрики
- Архитектурные распределения

### Визуализации

Автоматически генерируются:
- `pareto_stage2.png`: Pareto frontier после Stage 2
- `pareto_final.png`: Final Pareto frontier
- WandB dashboards с интерактивными графиками

## 🎯 Примеры использования

### Поиск SOTA модели для CIFAR-100

```bash
python main.py \
  --population-size 3000 \
  --device cuda \
  --output-dir ./cifar100_sota
```

### Исследование novel архитектур с высокой новизной

Увеличьте `novelty_weight` в конфигурации:
```python
# В config.py или YAML
selection_criteria:
  novelty_weight: 0.5  # Больше фокуса на новизну
  diversity_weight: 0.4
```

### Transfer learning с предыдущих экспериментов

База знаний автоматически переиспользуется при повторных запусках.

## 🧪 Тестирование

```bash
# Быстрый тест функциональности
python main.py --quick-test

# Запуск unit тестов
pytest tests/

# Проверка импортов
python -c "from src.orchestrator import EvoArchitectOrchestrator; print('OK')"
```

## 🤝 Вклад и расширение

### Добавление нового типа блока

1. Реализуйте блок в `src/search_space/blocks.py`:
```python
class MyCustomBlock(nn.Module):
    def __init__(self, ...):
        # Ваша реализация
        pass
```

2. Добавьте в `get_block_by_name()`:
```python
blocks["MyCustomBlock"] = MyCustomBlock
```

3. Обновите конфигурацию:
```python
search_space.base_blocks.append("MyCustomBlock")
```

### Добавление новой метрики

1. Реализуйте в `src/evaluation/metrics.py`:
```python
class MyCustomMetric:
    def compute(self, model, dataloader):
        # Ваша логика
        return metric_value
```

2. Интегрируйте в `compute_all_metrics()`.

## 📖 Научная база

Проект основан на следующих концепциях:

- **Neural Architecture Search (NAS)**: ENAS, DARTS, NAS-Bench
- **Multi-objective Optimization**: NSGA-II, Pareto optimality
- **Novelty Search**: Quality-Diversity algorithms
- **Meta-Learning**: MAML, Learning to Learn
- **Evolutionary Algorithms**: Genetic algorithms, mutation strategies

## ⚙️ Системные требования

**Минимальные:**
- Python 3.8+
- 8GB RAM
- GPU с 4GB VRAM (или CPU)

**Рекомендуемые:**
- Python 3.10+
- 32GB RAM
- NVIDIA GPU с 8GB+ VRAM (RTX 3060+)
- 50GB свободного места на диске

**Для вашей системы (RTX 4060 + i5-12100F):**
- Оптимально: population_size = 1000-2000
- Parallel trials = 2-4
- Ожидаемое время: 12-24 часа для полного пайплайна

## 🐛 Troubleshooting

**CUDA Out of Memory:**
```python
# Уменьшите batch size или population size
config.initial_population_size = 1000
runtime.compute_resources.num_parallel_trials = 1
```

**Слишком долгое выполнение:**
```python
# Используйте quick test или уменьшите epochs
python main.py --quick-test
```

**Проблемы с WandB:**
```bash
# Отключите WandB или авторизуйтесь
wandb login
# или в коде:
config.logging.dashboard_provider = None
```

## 📄 Лицензия

MIT License - свободно используйте для исследований и коммерческих проектов.

## 🙏 Благодарности

Проект вдохновлен работами:
- AutoML и NAS community
- NSGA-II и multi-objective optimization
- Novelty search и quality-diversity
- Open-source ML frameworks (PyTorch, Ray)

## 📧 Контакты

Для вопросов, предложений и сотрудничества:
- GitHub Issues: [Project Issues](https://github.com/yourusername/Project_EvoArchitect_v3/issues)
- Email: your.email@example.com

---

**Happy Evolving! 🧬🚀**
