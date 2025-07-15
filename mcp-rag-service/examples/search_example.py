#!/usr/bin/env python3
"""
Демонстрация семантического поиска с embeddings
Показывает важность преобразования query в embeddings
"""

import asyncio
import logging
from pathlib import Path
import sys

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_client import RAGClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_semantic_search():
    """Демонстрация семантического поиска"""
    
    async with RAGClient() as client:
        
        # Тестовые запросы для демонстрации семантического поиска
        test_queries = [
            {
                "query": "машинное обучение",
                "description": "Прямое упоминание термина"
            },
            {
                "query": "ML algorithms neural networks",
                "description": "Английские синонимы"
            },
            {
                "query": "искусственный интеллект нейронные сети",
                "description": "Смежные термины"
            },
            {
                "query": "обучение модели на данных",
                "description": "Описательный запрос"
            },
            {
                "query": "векторное представление текста",
                "description": "Техническое описание embeddings"
            }
        ]
        
        logger.info("=== ДЕМОНСТРАЦИЯ СЕМАНТИЧЕСКОГО ПОИСКА ===")
        logger.info("Показываем как query преобразуется в embeddings и ищется по similarity\n")
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            logger.info(f"🔍 Запрос {i}: '{query}'")
            logger.info(f"   Тип: {description}")
            
            try:
                # Поиск с разными порогами similarity
                thresholds = [0.8, 0.6, 0.4]
                
                for threshold in thresholds:
                    logger.info(f"   └─ Порог similarity: {threshold}")
                    
                    results = await client.semantic_search(
                        query=query,
                        top_k=3,
                        similarity_threshold=threshold,
                        rag_server_url="http://localhost:8000"
                    )
                    
                    if results:
                        logger.info(f"      Найдено: {len(results)} результатов")
                        for j, result in enumerate(results[:2], 1):  # Показываем топ-2
                            logger.info(f"      {j}. {result['file_name']} (similarity: {result['similarity']})")
                            logger.info(f"         Превью: {result['content'][:100]}...")
                    else:
                        logger.info(f"      Результатов не найдено")
                
                logger.info("")  # Пустая строка между запросами
                
            except Exception as e:
                logger.error(f"Ошибка поиска для '{query}': {e}")

async def demonstrate_embedding_importance():
    """Демонстрация важности embeddings"""
    
    logger.info("=== ПОЧЕМУ EMBEDDINGS КРИТИЧЕСКИ ВАЖНЫ ===\n")
    
    # Объясняем процесс
    explanations = [
        "1. 📄 При загрузке документов:",
        "   - Текст разбивается на chunks",
        "   - Каждый chunk → model.encode() → embedding FLOAT[768]", 
        "   - Embeddings сохраняются в DuckDB",
        "",
        "2. 🔍 При поиске пользователя:",
        "   - Запрос 'машинное обучение' → model.encode() → query_embedding",
        "   - DuckDB: array_cosine_similarity(chunk_embedding, query_embedding)",
        "   - Результаты сортируются по similarity DESC",
        "",
        "3. 🎯 Семантическое понимание:",
        "   - 'машинное обучение' ≈ 'ML' ≈ 'neural networks'",
        "   - Embeddings кодируют семантический смысл, не только слова",
        "   - Одна модель для документов и запросов → единое векторное пространство",
        "",
        "4. ❌ Без embeddings запроса:",
        "   - Невозможно сравнить text с FLOAT[768]",
        "   - Нет семантического понимания",
        "   - Только точное совпадение слов (как grep)",
        "",
        "5. ✅ С embeddings:",
        "   - Семантический поиск по смыслу",
        "   - Находит синонимы и близкие концепции", 
        "   - Работает на разных языках",
        "   - Similarity score показывает релевантность"
    ]
    
    for explanation in explanations:
        logger.info(explanation)

async def demonstrate_similarity_thresholds():
    """Демонстрация порогов similarity"""
    
    logger.info("\n=== ПОРОГИ SIMILARITY (из user_rule.txt) ===\n")
    
    thresholds = [
        {"range": "0.9 - 1.0", "interpretation": "Практически идентичный", "use_case": "Детекция дубликатов"},
        {"range": "0.8 - 0.9", "interpretation": "Очень похожий", "use_case": "Поиск похожих документов"},
        {"range": "0.7 - 0.8", "interpretation": "Семантически связанный", "use_case": "Рекомендации"},
        {"range": "0.6 - 0.7", "interpretation": "Умеренно связанный", "use_case": "Релевантный поиск"},
        {"range": "0.5 - 0.6", "interpretation": "Слабая связь", "use_case": "Кластеризация"},
        {"range": "0.3 - 0.5", "interpretation": "Очень слабая связь", "use_case": "—"},
        {"range": "0.0 - 0.3", "interpretation": "Несвязанный", "use_case": "—"},
    ]
    
    logger.info("Интерпретация similarity scores:")
    for threshold in thresholds:
        logger.info(f"  {threshold['range']}: {threshold['interpretation']} ({threshold['use_case']})")

async def main():
    """Главная функция"""
    logger.info("🚀 Демонстрация семантического поиска с embeddings\n")
    
    try:
        # Объясняем теорию
        await demonstrate_embedding_importance()
        await demonstrate_similarity_thresholds()
        
        # Практическая демонстрация (если RAG сервер доступен)
        try:
            await demonstrate_semantic_search()
        except Exception as e:
            logger.warning(f"Не удалось подключиться к RAG серверу: {e}")
            logger.info("Убедитесь, что RAG сервер запущен на http://localhost:8000")
            logger.info("И что в базе есть документы для поиска")
        
        logger.info("✅ Демонстрация завершена!")
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 