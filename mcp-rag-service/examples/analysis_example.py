#!/usr/bin/env python3
"""
Пример векторного анализа через MCP RAG Service
Демонстрация всех типов анализа из user_rule.txt
"""

import asyncio
import logging
import json
from pathlib import Path
import sys

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_operations import VectorAnalytics

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_cluster_analysis():
    """Демонстрация кластерного анализа"""
    logger.info("=== КЛАСТЕРНЫЙ АНАЛИЗ ===")
    
    analytics = VectorAnalytics()
    
    try:
        result = await analytics.analyze_collection(
            analysis_type="clusters",
            top_k=10,
            db_path="data/rag.duckdb"
        )
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Интерпретация результатов
        clusters = result.get("results", [])
        if clusters:
            logger.info(f"Найдено {len(clusters)} похожих пар chunks")
            
            high_sim = [c for c in clusters if c["similarity"] >= 0.8]
            if high_sim:
                logger.info(f"Высокое сходство (≥0.8): {len(high_sim)} пар")
                for pair in high_sim[:3]:  # Показываем топ-3
                    logger.info(f"  {pair['file1']} ↔ {pair['file2']}: {pair['similarity']:.4f}")
        
    except Exception as e:
        logger.error(f"Ошибка кластерного анализа: {e}")

async def demonstrate_outlier_analysis():
    """Демонстрация анализа выбросов"""
    logger.info("\n=== АНАЛИЗ ВЫБРОСОВ ===")
    
    analytics = VectorAnalytics()
    
    try:
        result = await analytics.analyze_collection(
            analysis_type="outliers",
            top_k=5,
            db_path="data/rag.duckdb"
        )
        
        outliers = result.get("results", [])
        if outliers:
            logger.info(f"Найдено {len(outliers)} уникальных chunks")
            
            for outlier in outliers:
                logger.info(f"  {outlier['file_name']}: similarity={outlier['avg_similarity']:.4f}")
                logger.info(f"    Превью: {outlier['content_preview'][:100]}...")
        
        # Показываем интерпретацию
        interpretation = result.get("interpretation", {})
        for category, items in interpretation.items():
            if items:
                logger.info(f"  {category}: {len(items)} chunks")
        
    except Exception as e:
        logger.error(f"Ошибка анализа выбросов: {e}")

async def demonstrate_centrality_analysis():
    """Демонстрация анализа центральности"""
    logger.info("\n=== АНАЛИЗ ЦЕНТРАЛЬНОСТИ ===")
    
    analytics = VectorAnalytics()
    
    try:
        result = await analytics.analyze_collection(
            analysis_type="centrality",
            top_k=5,
            db_path="data/rag.duckdb"
        )
        
        central_chunks = result.get("results", [])
        if central_chunks:
            logger.info(f"Найдено {len(central_chunks)} центральных chunks")
            
            for chunk in central_chunks:
                logger.info(f"  {chunk['file_name']}: centrality={chunk['avg_similarity']:.4f}")
                logger.info(f"    Превью: {chunk['content_preview'][:100]}...")
        
    except Exception as e:
        logger.error(f"Ошибка анализа центральности: {e}")

async def demonstrate_similarity_matrix():
    """Демонстрация матрицы similarity"""
    logger.info("\n=== МАТРИЦА SIMILARITY ===")
    
    analytics = VectorAnalytics()
    
    try:
        result = await analytics.analyze_collection(
            analysis_type="similarity_matrix",
            top_k=20,
            db_path="data/rag.duckdb"
        )
        
        summary = result.get("summary", {})
        logger.info(f"Всего пар файлов: {summary.get('total_pairs', 0)}")
        logger.info(f"Высокое сходство (≥0.8): {summary.get('highly_similar', 0)}")
        logger.info(f"Умеренное сходство (0.6-0.8): {summary.get('moderately_similar', 0)}")
        logger.info(f"Слабое сходство (<0.6): {summary.get('weakly_similar', 0)}")
        
        # Показываем топ связей
        matrix = result.get("results", [])
        if matrix:
            logger.info("\nТоп-5 наиболее связанных пар файлов:")
            for pair in sorted(matrix, key=lambda x: x["similarity"], reverse=True)[:5]:
                logger.info(f"  {pair['file1']} ↔ {pair['file2']}: {pair['similarity']:.4f}")
        
    except Exception as e:
        logger.error(f"Ошибка создания матрицы similarity: {e}")

async def demonstrate_similar_documents():
    """Демонстрация поиска похожих документов"""
    logger.info("\n=== ПОИСК ПОХОЖИХ ДОКУМЕНТОВ ===")
    
    analytics = VectorAnalytics()
    
    try:
        # Сначала получаем список файлов
        stats = await analytics.get_collection_stats("data/rag.duckdb")
        files = stats.get("files_breakdown", [])
        
        if files:
            # Берем первый файл как референс
            reference_file = files[0]["file_name"]
            logger.info(f"Ищем документы, похожие на: {reference_file}")
            
            result = await analytics.find_similar_documents(
                reference_file=reference_file,
                top_k=5,
                db_path="data/rag.duckdb"
            )
            
            similar_docs = result.get("similar_documents", [])
            if similar_docs:
                logger.info(f"Найдено {len(similar_docs)} похожих документов:")
                for doc in similar_docs:
                    logger.info(f"  {doc['similar_file']}: similarity={doc['avg_similarity']:.4f}")
            else:
                logger.info("Похожие документы не найдены")
        else:
            logger.warning("Нет файлов в базе данных для анализа")
        
    except Exception as e:
        logger.error(f"Ошибка поиска похожих документов: {e}")

async def demonstrate_collection_stats():
    """Демонстрация статистики коллекции"""
    logger.info("\n=== СТАТИСТИКА КОЛЛЕКЦИИ ===")
    
    analytics = VectorAnalytics()
    
    try:
        stats = await analytics.get_collection_stats("data/rag.duckdb")
        
        logger.info(f"Всего chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"Всего файлов: {stats.get('total_files', 0)}")
        logger.info(f"Chunks с embeddings: {stats.get('chunks_with_embeddings', 0)}")
        logger.info(f"Средняя длина содержимого: {stats.get('avg_content_length', 0):.1f} символов")
        
        files_breakdown = stats.get("files_breakdown", [])
        if files_breakdown:
            logger.info("\nРаспределение по файлам:")
            for file_info in files_breakdown[:10]:  # Топ-10 файлов
                logger.info(f"  {file_info['file_name']}: {file_info['chunks_count']} chunks")
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")

async def demonstrate_direct_query():
    """Демонстрация прямого SQL запроса"""
    logger.info("\n=== ПРЯМОЙ SQL ЗАПРОС ===")
    
    analytics = VectorAnalytics()
    
    try:
        # Безопасный SQL запрос из user_rule.txt
        sql_query = """
        SELECT 
            file_name,
            COUNT(*) as chunks_count,
            AVG(LENGTH(content)) as avg_content_length
        FROM chunks 
        GROUP BY file_name 
        ORDER BY chunks_count DESC 
        LIMIT 5
        """
        
        result = await analytics.execute_direct_query(sql_query, "data/rag.duckdb")
        
        if result:
            logger.info("Топ-5 файлов по количеству chunks:")
            for row in result:
                logger.info(f"  {row['file_name']}: {row['chunks_count']} chunks, "
                          f"средняя длина: {row['avg_content_length']:.1f}")
        else:
            logger.info("Результаты не найдены")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения SQL запроса: {e}")

async def main():
    """Главная функция - запускает все примеры анализа"""
    logger.info("Начинаем демонстрацию векторного анализа RAG коллекции...")
    
    try:
        # Проверяем наличие базы данных
        db_path = Path("data/rag.duckdb")
        if not db_path.exists():
            logger.error(f"База данных не найдена: {db_path}")
            logger.info("Запустите сначала RAG сервер и загрузите некоторые документы")
            return
        
        # Запускаем все демонстрации
        await demonstrate_collection_stats()
        await demonstrate_cluster_analysis()
        await demonstrate_outlier_analysis()
        await demonstrate_centrality_analysis()
        await demonstrate_similarity_matrix()
        await demonstrate_similar_documents()
        await demonstrate_direct_query()
        
        logger.info("\n✅ Демонстрация векторного анализа завершена успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка в демонстрации: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 