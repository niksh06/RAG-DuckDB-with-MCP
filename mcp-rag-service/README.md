# MCP RAG Service

**MCP сервис для интеграции с Python RAG Server (DuckDB VSS) v2.0**

Улучшенный MCP сервис для работы с продвинутыми методами векторного поиска, основанный на принципе KISS (Keep It Simple, Stupid).

## 🚀 Возможности

### Основные функции:
- **`rag_upload_file`** - Загрузка файлов в RAG сервер
- **`rag_search`** - Продвинутый поиск по документам (гибридный, семантический, ключевой) с переранжированием и расширением запроса.
- **`rag_get_file_content`** - Получение полного содержимого файла из базы данных.
- **`rag_similar_documents`** - Поиск похожих документов
- **`rag_analyze_collection`** - Анализ коллекции (кластеры, выбросы, центральность)
- **`rag_get_collection_stats`** - Получение быстрой статистики по RAG коллекции.
- **`rag_query_direct`** - Прямые SQL запросы к DuckDB VSS

### Поддерживаемые форматы файлов:
- Текстовые: `.txt`, `.md`
- Документы: `.pdf`
- Программный код: `.py`, `.js`, `.java`, `.c`, `.cpp`
- Конфигурационные: `.json`, `.yaml`, `.yml`, `.ini`, `.toml`

## 📋 Требования

- Python 3.8+
- Работающий Python RAG Server с DuckDB VSS
- MCP совместимый клиент (например, Claude Desktop)

## 🛠️ Установка и запуск

```bash
# Клонирование репозитория
cd mcp-rag-service

# Установка в режиме разработки
pip install -e .

# Или с dev зависимостями
pip install -e ".[dev]"
```

### Запуск через Docker (Рекомендуется)
1. Сборка Docker образа:
```bash
docker build -t mcp-rag-service:latest .
```

2. Запуск контейнера:
```bash
docker run --rm -it \
  --network="host" \
  -v "/path/to/your/rag-server/data:/data" \
  mcp-rag-service:latest
```

Контейнеру нужен доступ к RAG серверу (по умолчанию http://localhost:8000) и к файлу базы данных rag.duckdb.

- `--network="host"`: Позволяет контейнеру обращаться к localhost хост-машины.
- `-v "/path/to/your/rag-server/data:/data"`: Монтирует папку с rag.duckdb внутрь контейнера.

### Настройка MCP клиента (например, Cursor или Claude Desktop)

Добавьте в конфигурационный файл вашего клиента:

```json
{
  "mcpServers": {
    "rag-vector-service": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "--network=host",
        "-v", "/path/to/your/rag-server/data:/data",
        "mcp-rag-service:latest"
      ],
      "env": {
        "RAG_SERVER_URL": "http://localhost:8000"
      }
    }
  }
}
```

Замените `/path/to/your/rag-server/data` на реальный путь к папке data вашего основного RAG сервера.

### Убедитесь, что основной RAG сервер запущен:
```bash
cd /path/to/your/python-rag-server
docker run -p 8000:8000 -v "$(pwd)/data:/app/data" ...
```

## 📖 Использование

### Загрузка файла
```python
# Загрузка документа
await mcp_client.call_tool(
    "rag_upload_file",
    {"file_path": "/path/to/document.pdf"}
)
```

### Продвинутый поиск
```python
# Гибридный поиск с переранжированием
await mcp_client.call_tool(
    "rag_search",
    {
        "query": "машинное обучение и нейронные сети",
        "top_k": 5,
        "search_type": "hybrid",
        "use_reranker": true,
        "expand_query": false
    }
)
```

### Анализ коллекции
```python
# Поиск похожих документов
await mcp_client.call_tool(
    "rag_similar_documents",
    {
        "reference_file": "main.py",
        "top_k": 3
    }
)

# Анализ кластеров
await mcp_client.call_tool(
    "rag_analyze_collection",
    {
        "analysis_type": "clusters",
        "top_k": 10
    }
)
```

### Получение статистики
```python
# Получить общую статистику по базе данных
await mcp_client.call_tool(
    "rag_get_collection_stats",
    {}
)
```

### Прямой SQL запрос
```python
# Для продвинутых пользователей
await mcp_client.call_tool(
    "rag_query_direct",
    {
        "sql_query": "SELECT file_name, COUNT(*) FROM chunks GROUP BY file_name"
    }
)
```

## 🔒 Безопасность

- ✅ Валидация путей к файлам
- ✅ Whitelist разрешенных расширений файлов
- ✅ SQL injection защита для прямых запросов
- ✅ Ограничение на команды только для чтения (SELECT, WITH, SHOW, etc.)
- ✅ Санитизация имен файлов
- ✅ Контейнеризация через Docker для простоты развертывания.

## 🏗️ Архитектура

```
mcp-rag-service/
├── Dockerfile                 # Docker-файл для сборки сервиса
├── src/                       # Исходный код
│   ├── rag_mcp_server.py     # Главный MCP сервер
│   ├── rag_client.py         # HTTP клиент для RAG сервера
│   ├── vector_operations.py  # Векторные операции и аналитика
│   └── utils.py              # Утилиты и валидация
├── examples/                  # Примеры использования
│   ├── upload_example.py     # Загрузка файлов
│   ├── search_example.py     # Семантический поиск
│   └── analysis_example.py   # Анализ коллекции
└── pyproject.toml            # Конфигурация проекта
```

## 🧠 Векторная аналитика

### Основные типы анализа:

**Кластеризация**: Поиск самых похожих пар документов для понимания тематических групп.

**Поиск выбросов**: Выявление уникального контента, который отличается от основной коллекции.

**Анализ центральности**: Определение "центральных" документов, которые связаны с большинством других.

**Матрица similarity**: Анализ связей между файлами для понимания структуры коллекции.

## 🚦 Статус проекта

- ✅ Основные MCP функции реализованы
- ✅ Интеграция с DuckDB VSS  
- ✅ Безопасность и валидация
- ✅ Векторная аналитика
- ✅ Продвинутые методы поиска
- ⏳ REST API интеграция (улучшение)
- ⏳ Тесты
- ⏳ Продвинутая обработка ошибок

## 📄 Лицензия

MIT License 