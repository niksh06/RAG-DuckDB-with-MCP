# Docker Build Options - Optimized with UV

## ⚡ Ultra-Fast Build System

Использует **UV package manager** с параллельной установкой для максимальной скорости:
- 🚀 **10-100x быстрее** стандартного pip
- 🔄 **$(nproc) parallel jobs** - использует все CPU ядра
- 💾 **Smart caching** - Docker layer + UV cache
- 🛡️ **Robust retries** - автоматические повторы при сбоях

## Быстрая сборка (CPU-only)

Для разработки и тестирования:

```bash
docker build --build-arg USE_CPU_ONLY=true -t python-rag-server:cpu .
```

**Преимущества:**
- ⚡ **5-10x быстрее** (без CUDA пакетов)
- 📦 **~2-3GB меньше** размер образа  
- 🔄 **100% надежность** (нет больших загрузок)
- 🎯 **PyTorch CPU-only** из специального index

## Полная сборка (с CUDA)

Для production с GPU ускорением:

```bash
docker build --build-arg USE_CPU_ONLY=false -t python-rag-server:cuda .
```

**или по умолчанию:**

```bash
docker build -t python-rag-server:cuda .
```

**Особенности:**
- 🚀 **Максимальная ML производительность**
- 🎯 **GPU ускорение** для трансформеров
- ⚡ **Параллельная загрузка** больших CUDA файлов
- 🔄 **Smart retries** (до 5 попыток)

## UV Performance Features

### Параллельная обработка:
```bash
UV_CONCURRENT_DOWNLOADS=$(nproc)  # Параллельные загрузки
UV_CONCURRENT_INSTALLS=$(nproc)   # Параллельные установки  
UV_HTTP_TIMEOUT=1200              # Расширенные таймауты
```

### Cache стратегия:
- **Docker mount cache** для `/root/.cache/uv` и `/root/.cache/pip`
- **Layer caching** для requirements файлов
- **Incremental builds** - пересборка только изменившихся слоев

## Устранение проблем

### Медленная сборка:
```bash
# Используйте BuildKit для лучшего кэширования
DOCKER_BUILDKIT=1 docker build --build-arg USE_CPU_ONLY=false .

# Verbose логи для диагностики
docker buildx build --build-arg USE_CPU_ONLY=false --progress=plain .
```

### Проблемы с сетью:
```bash
# CPU-only билд как fallback
docker build --build-arg USE_CPU_ONLY=true -t rag-dev .

# Увеличенные таймауты для слабой сети (уже включены)
# UV_HTTP_TIMEOUT=1200, --timeout 1200
```

## Структура Requirements (Optimized)

```
requirements-base.txt     # 🟢 Быстрые пакеты (FastAPI, DuckDB, etc.)
requirements-ml.txt       # 🔴 CUDA ML пакеты (большие, медленные)  
requirements-cpu.txt      # 🟡 CPU-only ML пакеты (без torch)
requirements.txt          # 📋 Полный список (справочный)
```

**Преимущества разделения:**
- ✅ **Независимое кэширование** каждой группы
- ✅ **Conditional installs** в зависимости от USE_CPU_ONLY
- ✅ **Better error isolation** - проблема с ML не влияет на базовые пакеты

## Performance Benchmarks

| Сценарий | Время сборки | Размер образа | CPU Cores Used |
|----------|-------------|---------------|-----------------|
| **CPU-only build** | ~3-5 мин | ~2.5GB | $(nproc) |
| **CUDA build** | ~8-15 мин | ~5-7GB | $(nproc) |
| **Incremental rebuild** | ~30 сек | - | $(nproc) |

## Рекомендуемые команды

### 🛠️ Разработка (быстро, надежно):
```bash
docker build --build-arg USE_CPU_ONLY=true -t rag-dev .
docker run -p 8000:8000 rag-dev
```

### 🚀 Production (полная производительность):
```bash
docker build --build-arg USE_CPU_ONLY=false -t rag-prod .
docker run --gpus all -p 8000:8000 rag-prod
```

### 🔍 Диагностика и debugging:
```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg USE_CPU_ONLY=false \
  --progress=plain \
  --no-cache \
  -t rag-debug .
``` 