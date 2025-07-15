"""
Векторные операции и аналитика для DuckDB VSS
Реализация паттернов из user_rule.txt
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
import duckdb
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorAnalytics:
    """Аналитика векторной базы данных"""
    
    def __init__(self):
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
    
    async def _get_connection(self, db_path: str) -> duckdb.DuckDBPyConnection:
        """Получить соединение с DuckDB"""
        if self.connection is None:
            db_file = Path(db_path)
            if not db_file.exists():
                raise FileNotFoundError(f"База данных не найдена: {db_path}")
            
            self.connection = duckdb.connect(database=db_path, read_only=True)
            
            # Загружаем VSS extension
            try:
                self.connection.execute("LOAD vss;")
            except Exception as e:
                logger.warning(f"Не удалось загрузить VSS extension: {e}")
        
        return self.connection
    
    async def find_similar_documents(
        self, 
        reference_file: str, 
        top_k: int = 5, 
        db_path: str = "/data/rag.duckdb"
    ) -> Dict[str, Any]:
        """
        Найти документы, похожие на указанный файл
        Паттерн из user_rule.txt: "Поиск похожих документов"
        """
        conn = await self._get_connection(db_path)
        
        try:
            # Кластеризация по файлам (из user_rule.txt)
            query = """
            SELECT a.file_name as file1, b.file_name as file2,
                   ROUND(AVG(array_cosine_similarity(a.embedding, b.embedding)), 4) as avg_similarity
            FROM chunks a, chunks b 
            WHERE a.file_name = ? AND b.file_name != a.file_name 
            GROUP BY a.file_name, b.file_name 
            ORDER BY avg_similarity DESC
            LIMIT ?;
            """
            
            results = conn.execute(query, (reference_file, top_k)).fetchall()
            
            similar_docs = [
                {
                    "reference_file": row[0],
                    "similar_file": row[1],
                    "avg_similarity": float(row[2])
                }
                for row in results
            ]
            
            return {
                "reference_file": reference_file,
                "similar_documents": similar_docs,
                "total_found": len(similar_docs)
            }
            
        except Exception as e:
            logger.error(f"Ошибка поиска похожих документов: {e}")
            raise
    
    async def analyze_collection(
        self, 
        analysis_type: str = "clusters", 
        top_k: int = 10, 
        db_path: str = "/data/rag.duckdb"
    ) -> Dict[str, Any]:
        """
        Анализ коллекции документов
        Реализует различные паттерны из user_rule.txt
        """
        conn = await self._get_connection(db_path)
        
        try:
            if analysis_type == "clusters":
                return await self._analyze_clusters(conn, top_k)
            elif analysis_type == "outliers":
                return await self._analyze_outliers(conn, top_k)
            elif analysis_type == "centrality":
                return await self._analyze_centrality(conn, top_k)
            elif analysis_type == "similarity_matrix":
                return await self._create_similarity_matrix(conn, top_k)
            else:
                raise ValueError(f"Неподдерживаемый тип анализа: {analysis_type}")
                
        except Exception as e:
            logger.error(f"Ошибка анализа коллекции: {e}")
            raise
    
    async def _analyze_clusters(self, conn: duckdb.DuckDBPyConnection, top_k: int) -> Dict[str, Any]:
        """
        Кластерный анализ - самые похожие пары чанков
        Паттерн из user_rule.txt: "Самые похожие пары чанков"
        """
        query = """
        SELECT a.id as id1, b.id as id2, 
               a.file_name as file1, b.file_name as file2,
               LEFT(a.content, 80) as content1, LEFT(b.content, 80) as content2,
               array_cosine_similarity(a.embedding, b.embedding) as similarity
        FROM chunks a, chunks b 
        WHERE a.id < b.id 
        ORDER BY similarity DESC 
        LIMIT ?;
        """
        
        results = conn.execute(query, (top_k,)).fetchall()
        
        clusters = [
            {
                "chunk1_id": row[0],
                "chunk2_id": row[1],
                "file1": row[2],
                "file2": row[3],
                "content1_preview": row[4],
                "content2_preview": row[5],
                "similarity": float(row[6])
            }
            for row in results
        ]
        
        return {
            "analysis_type": "clusters",
            "description": "Самые похожие пары chunks в коллекции",
            "results": clusters,
            "similarity_thresholds": {
                "very_high": [c for c in clusters if c["similarity"] >= 0.9],
                "high": [c for c in clusters if 0.8 <= c["similarity"] < 0.9],
                "moderate": [c for c in clusters if 0.7 <= c["similarity"] < 0.8]
            }
        }
    
    async def _analyze_outliers(self, conn: duckdb.DuckDBPyConnection, top_k: int) -> Dict[str, Any]:
        """
        Анализ выбросов - уникальный контент
        Паттерн из user_rule.txt: "Найти выбросы в коллекции"
        """
        query = """
        SELECT a.id, a.file_name, LEFT(a.content, 100) as content,
               AVG(array_cosine_similarity(a.embedding, b.embedding)) as avg_similarity
        FROM chunks a, chunks b WHERE a.id != b.id
        GROUP BY a.id, a.file_name, a.content
        ORDER BY avg_similarity ASC 
        LIMIT ?;
        """
        
        results = conn.execute(query, (top_k,)).fetchall()
        
        outliers = [
            {
                "chunk_id": row[0],
                "file_name": row[1],
                "content_preview": row[2],
                "avg_similarity": float(row[3])
            }
            for row in results
        ]
        
        return {
            "analysis_type": "outliers",
            "description": "Наиболее уникальные chunks (выбросы)",
            "results": outliers,
            "interpretation": {
                "very_unique": [o for o in outliers if o["avg_similarity"] < 0.3],
                "unique": [o for o in outliers if 0.3 <= o["avg_similarity"] < 0.5],
                "somewhat_unique": [o for o in outliers if 0.5 <= o["avg_similarity"] < 0.6]
            }
        }
    
    async def _analyze_centrality(self, conn: duckdb.DuckDBPyConnection, top_k: int) -> Dict[str, Any]:
        """
        Анализ центральности - "центральный" контент
        Паттерн из user_rule.txt: "Найти центральный контент"
        """
        query = """
        SELECT a.id, a.file_name, LEFT(a.content, 120) as content_preview,
               AVG(array_cosine_similarity(a.embedding, b.embedding)) as avg_similarity
        FROM chunks a, chunks b 
        WHERE a.id != b.id 
        GROUP BY a.id, a.file_name, a.content 
        ORDER BY avg_similarity DESC 
        LIMIT ?;
        """
        
        results = conn.execute(query, (top_k,)).fetchall()
        
        central_chunks = [
            {
                "chunk_id": row[0],
                "file_name": row[1],
                "content_preview": row[2],
                "avg_similarity": float(row[3])
            }
            for row in results
        ]
        
        return {
            "analysis_type": "centrality",
            "description": "Наиболее центральные chunks (связанные с остальными)",
            "results": central_chunks,
            "interpretation": {
                "highly_central": [c for c in central_chunks if c["avg_similarity"] >= 0.8],
                "moderately_central": [c for c in central_chunks if 0.7 <= c["avg_similarity"] < 0.8],
                "somewhat_central": [c for c in central_chunks if 0.6 <= c["avg_similarity"] < 0.7]
            }
        }
    
    async def _create_similarity_matrix(self, conn: duckdb.DuckDBPyConnection, top_k: int) -> Dict[str, Any]:
        """
        Создание матрицы similarity между файлами
        Паттерн из user_rule.txt: "Построй карту документов"
        """
        query = """
        SELECT a.file_name, b.file_name, 
               ROUND(AVG(array_cosine_similarity(a.embedding, b.embedding)), 3) as similarity
        FROM chunks a, chunks b 
        WHERE a.file_name <= b.file_name
        GROUP BY a.file_name, b.file_name 
        ORDER BY a.file_name, similarity DESC
        LIMIT ?;
        """
        
        results = conn.execute(query, (top_k,)).fetchall()
        
        matrix = [
            {
                "file1": row[0],
                "file2": row[1],
                "similarity": float(row[2])
            }
            for row in results
        ]
        
        # Группируем по файлам для удобства
        files = {}
        for item in matrix:
            file1 = item["file1"]
            if file1 not in files:
                files[file1] = []
            files[file1].append({
                "related_file": item["file2"],
                "similarity": item["similarity"]
            })
        
        return {
            "analysis_type": "similarity_matrix",
            "description": "Матрица similarity между файлами",
            "results": matrix,
            "files_relationships": files,
            "summary": {
                "total_pairs": len(matrix),
                "highly_similar": len([m for m in matrix if m["similarity"] >= 0.8]),
                "moderately_similar": len([m for m in matrix if 0.6 <= m["similarity"] < 0.8]),
                "weakly_similar": len([m for m in matrix if m["similarity"] < 0.6])
            }
        }
    
    async def execute_direct_query(self, sql_query: str, db_path: str) -> List[Dict[str, Any]]:
        """
        Выполнить прямой SQL запрос к DuckDB VSS
        Для экспертного использования
        """
        conn = await self._get_connection(db_path)
        
        try:
            results = conn.execute(sql_query).fetchall()
            
            # Получаем названия колонок
            columns = [desc[0] for desc in conn.description] if conn.description else []
            
            # Преобразуем в список словарей
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(columns, row)))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения SQL запроса: {e}")
            raise
    
    async def get_collection_stats(self, db_path: str) -> Dict[str, Any]:
        """
        Получить статистику коллекции документов
        """
        conn = await self._get_connection(db_path)
        
        try:
            # Основная статистика
            stats_query = """
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT file_name) as total_files,
                COUNT(embedding) as with_embeddings,
                AVG(LENGTH(content)) as avg_content_length
            FROM chunks;
            """
            
            stats = conn.execute(stats_query).fetchone()
            
            # Статистика по файлам
            files_query = """
            SELECT 
                file_name,
                COUNT(*) as chunks_count,
                AVG(LENGTH(content)) as avg_chunk_length
            FROM chunks
            GROUP BY file_name
            ORDER BY chunks_count DESC;
            """
            
            files_stats = conn.execute(files_query).fetchall()
            
            return {
                "total_chunks": stats[0],
                "total_files": stats[1],
                "chunks_with_embeddings": stats[2],
                "avg_content_length": round(stats[3], 2) if stats[3] else 0,
                "files_breakdown": [
                    {
                        "file_name": row[0],
                        "chunks_count": row[1],
                        "avg_chunk_length": round(row[2], 2)
                    }
                    for row in files_stats
                ]
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            raise
    
    async def get_file_content(self, file_name: str, db_path: str) -> Dict[str, Any]:
        """
        Получить полное содержимое файла из базы данных, собрав его из чанков.
        """
        conn = await self._get_connection(db_path)
        
        try:
            query = """
            SELECT content
            FROM chunks
            WHERE file_name = ?
            ORDER BY chunk_index ASC;
            """
            
            results = conn.execute(query, (file_name,)).fetchall()
            
            if not results:
                return {"file_name": file_name, "content": "", "status": "not_found"}
            
            full_content = "".join([row[0] for row in results])
            
            return {"file_name": file_name, "content": full_content, "status": "success", "chunks_count": len(results)}
            
        except Exception as e:
            logger.error(f"Ошибка получения содержимого файла {file_name}: {e}")
            raise

    async def get_chunk_by_id(self, chunk_id: int, db_path: str) -> Optional[Dict[str, Any]]:
        """
        Получить содержимое и метаданные чанка по его ID.
        """
        conn = await self._get_connection(db_path)

        try:
            query = "SELECT id, file_name, chunk_index, content, metadata FROM chunks WHERE id = ?;"
            result = conn.execute(query, (chunk_id,)).fetchone()

            if not result:
                return None

            return {
                "id": result[0],
                "file_name": result[1],
                "chunk_index": result[2],
                "content": result[3],
                "metadata": json.loads(result[4]) if result[4] else {}
            }

        except Exception as e:
            logger.error(f"Ошибка получения чанка по ID {chunk_id}: {e}")
            raise

    def close(self):
        """Закрыть соединение с базой данных"""
        if self.connection:
            self.connection.close()
            self.connection = None 