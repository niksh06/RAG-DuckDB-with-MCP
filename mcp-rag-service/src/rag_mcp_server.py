#!/usr/bin/env python3
"""
MCP Server для интеграции с Python RAG Server (DuckDB VSS)
Простая реализация по принципу KISS
"""

import json
import logging
import asyncio
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from mcp import types

# Local imports
from .rag_client import RAGClient
from .vector_operations import VectorAnalytics
from .utils import validate_file_path, safe_sql_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMCPServer:
    """MCP Server для RAG операций"""
    
    def __init__(self):
        self.server = Server("rag-vector-service")
        self.rag_client = RAGClient()
        self.vector_analytics = VectorAnalytics()
        
        # Регистрируем tools
        self._register_tools()
    
    def _register_tools(self):
        """Регистрация MCP tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """Список доступных tools"""
            return [
                Tool(
                    name="rag_upload_file",
                    description="Загрузить файл в RAG сервер для обработки и создания embeddings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Путь к файлу для загрузки"
                            },
                            "rag_server_url": {
                                "type": "string",
                                "description": "URL RAG сервера",
                                "default": "http://localhost:8000"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="rag_search",
                    description="Выполнить поиск в RAG. Поддерживает гибридный, семантический и ключевой поиск с опциональным переранжированием и расширением запроса.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Поисковый запрос"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Количество результатов",
                                "default": 5,
                                "minimum": 1
                            },
                            "search_type": {
                                "type": "string",
                                "description": "Тип поиска",
                                "enum": ["hybrid", "semantic", "keyword"],
                                "default": "hybrid"
                            },
                            "use_reranker": {
                                "type": "boolean",
                                "description": "Использовать переранжирование для повышения точности",
                                "default": True
                            },
                            "expand_query": {
                                "type": "boolean",
                                "description": "Автоматически расширять запрос ключевыми словами из релевантных документов",
                                "default": False
                            },
                            "rag_server_url": {
                                "type": "string",
                                "description": "URL RAG сервера",
                                "default": "http://localhost:8000"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="rag_get_file_content",
                    description="Получить полное содержимое файла из базы данных RAG.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_name": {
                                "type": "string",
                                "description": "Имя файла, содержимое которого нужно получить."
                            },
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": ["file_name"]
                    }
                ),
                Tool(
                    name="rag_get_chunk_by_id",
                    description="Получить содержимое и метаданные чанка по его ID.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "integer",
                                "description": "ID чанка, который нужно получить."
                            },
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": ["chunk_id"]
                    }
                ),
                Tool(
                    name="rag_similar_documents",
                    description="Найти документы, похожие на указанный файл",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reference_file": {
                                "type": "string",
                                "description": "Имя референсного файла"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Количество похожих документов",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": ["reference_file"]
                    }
                ),
                Tool(
                    name="rag_analyze_collection",
                    description="Анализ коллекции документов (кластеры, выбросы, центральность)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "description": "Тип анализа",
                                "enum": ["clusters", "outliers", "centrality", "similarity_matrix"],
                                "default": "clusters"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Количество результатов",
                                "default": 10,
                                "minimum": 5,
                                "maximum": 50
                            },
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="rag_get_collection_stats",
                    description="Получить статистику по коллекции документов в RAG.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="rag_query_direct",
                    description="Прямой SQL запрос к DuckDB VSS (для экспертов)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sql_query": {
                                "type": "string",
                                "description": "SQL запрос к DuckDB VSS"
                            },
                            "db_path": {
                                "type": "string",
                                "description": "Путь к DuckDB файлу",
                                "default": "/data/rag.duckdb"
                            }
                        },
                        "required": ["sql_query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Обработка вызовов tools"""
            
            try:
                if name == "rag_upload_file":
                    return await self._handle_upload_file(arguments)
                elif name == "rag_search":
                    return await self._handle_search(arguments)
                elif name == "rag_get_file_content":
                    return await self._handle_get_file_content(arguments)
                elif name == "rag_get_chunk_by_id":
                    return await self._handle_get_chunk_by_id(arguments)
                elif name == "rag_similar_documents":
                    return await self._handle_similar_documents(arguments)
                elif name == "rag_analyze_collection":
                    return await self._handle_analyze_collection(arguments)
                elif name == "rag_get_collection_stats":
                    return await self._handle_get_collection_stats(arguments)
                elif name == "rag_query_direct":
                    return await self._handle_query_direct(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Ошибка выполнения {name}: {str(e)}"
                )]

    async def _handle_upload_file(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка загрузки файла"""
        file_path = args.get("file_path")
        rag_server_url = args.get("rag_server_url", "http://host.docker.internal:8000")
        
        if not file_path:
            raise ValueError("Путь к файлу не может быть пустым")
        
        # Валидация пути остается важной
        if not validate_file_path(file_path):
            raise ValueError(f"Недействительный или небезопасный путь к файлу: {file_path}")
        
        try:
            # Путь может быть относительным, Path() справится с этим.
            # Логика нормализации пути в /data/ слишком привязана к Docker.
            # Будем считать, что путь доступен как есть (через volume mount).
            result = await self.rag_client.upload_file(file_path, rag_server_url)
            
        except FileNotFoundError:
            result = {
                "status": "error",
                "message": f"Файл не найден по пути: {file_path}",
                "suggestion": "Убедитесь, что файл существует и путь к нему доступен из контейнера mcp-rag-service. Возможно, требуется настроить volume mapping (-v /host/path:/container/path)."
            }
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {file_path}: {e}")
            result = {
                "status": "error",
                "message": f"Ошибка загрузки файла: {str(e)}"
            }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]

    async def _handle_search(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка поиска"""
        query = args.get("query")
        top_k = args.get("top_k", 5)
        search_type = args.get("search_type", "hybrid")
        
        # Конвертируем boolean параметры правильно
        use_reranker_value = args.get("use_reranker", "true")
        if isinstance(use_reranker_value, bool):
            use_reranker = use_reranker_value
        else:
            use_reranker = str(use_reranker_value).lower() in ["true", "1", "yes", "on"]
        
        expand_query_value = args.get("expand_query", "false")
        if isinstance(expand_query_value, bool):
            expand_query = expand_query_value
        else:
            expand_query = str(expand_query_value).lower() in ["true", "1", "yes", "on"]
        
        rag_server_url = args.get("rag_server_url", "http://host.docker.internal:8000")
        
        if not query or not query.strip():
            raise ValueError("Поисковый запрос не может быть пустым")
        
        results = await self.rag_client.search(
            query=query, 
            top_k=top_k, 
            search_type=search_type,
            use_reranker=use_reranker,
            expand_query=expand_query,
            rag_server_url=rag_server_url
        )
        
        # Результат уже приходит в нужном формате от клиента
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, ensure_ascii=False)
        )]

    async def _handle_get_file_content(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка получения содержимого файла"""
        file_name = args.get("file_name")
        db_path = args.get("db_path", "/data/rag.duckdb")

        if not file_name:
            raise ValueError("Имя файла не может быть пустым")

        try:
            result = await self.vector_analytics.get_file_content(file_name, db_path)
            
            # Если файл не найден, добавляем полезную информацию
            if result.get("status") == "not_found":
                # Попробуем найти похожие файлы
                stats = await self.vector_analytics.get_collection_stats(db_path)
                available_files = [f["file_name"] for f in stats.get("files_breakdown", [])]
                
                # Найдем похожие имена файлов
                similar_files = [f for f in available_files if file_name.lower() in f.lower() or f.lower() in file_name.lower()]
                
                result.update({
                    "available_files_count": len(available_files),
                    "similar_files": similar_files[:5] if similar_files else [],
                    "suggestion": f"Используйте точное имя файла из коллекции. Доступно {len(available_files)} файлов."
                })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
            
        except Exception as e:
            logger.error(f"Ошибка получения содержимого файла {file_name}: {e}")
            error_result = {
                "status": "error",
                "file_name": file_name,
                "message": str(e),
                "suggestion": "Проверьте что база данных доступна и файл был проиндексирован"
            }
            return [TextContent(
                type="text",
                text=json.dumps(error_result, indent=2, ensure_ascii=False)
            )]

    async def _handle_get_chunk_by_id(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка получения чанка по ID"""
        chunk_id = args.get("chunk_id")
        db_path = args.get("db_path", "/data/rag.duckdb")

        if not isinstance(chunk_id, int) or chunk_id <= 0:
            raise ValueError("chunk_id должен быть положительным целым числом")

        result = await self.vector_analytics.get_chunk_by_id(chunk_id, db_path)

        if result is None:
            return [TextContent(type="text", text=json.dumps({"status": "not_found", "chunk_id": chunk_id}, indent=2, ensure_ascii=False))]

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]

    async def _handle_get_collection_stats(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка получения статистики коллекции"""
        db_path = args.get("db_path", "/data/rag.duckdb")
        
        results = await self.vector_analytics.get_collection_stats(db_path)
        
        formatted_results = {
            "description": "Статистика по RAG коллекции",
            "stats": results
        }

        return [TextContent(
            type="text",
            text=json.dumps(formatted_results, indent=2, ensure_ascii=False)
        )]

    async def _handle_similar_documents(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка поиска похожих документов"""
        reference_file = args.get("reference_file")
        if not reference_file:
            raise ValueError("reference_file не может быть пустым")
        
        top_k = args.get("top_k", 5)
        db_path = args.get("db_path", "/data/rag.duckdb")
        
        results = await self.vector_analytics.find_similar_documents(
            reference_file, top_k, db_path
        )
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, ensure_ascii=False)
        )]

    async def _handle_analyze_collection(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка анализа коллекции"""
        analysis_type = args.get("analysis_type", "clusters")
        top_k = args.get("top_k", 10)
        db_path = args.get("db_path", "/data/rag.duckdb")
        
        results = await self.vector_analytics.analyze_collection(
            analysis_type, top_k, db_path
        )
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, ensure_ascii=False)
        )]

    async def _handle_query_direct(self, args: Dict[str, Any]) -> List[TextContent]:
        """Обработка прямого SQL запроса"""
        sql_query = args.get("sql_query")
        if not sql_query:
            raise ValueError("sql_query не может быть пустым")
        
        db_path = args.get("db_path", "/data/rag.duckdb")
        
        if not safe_sql_query(sql_query):
            raise ValueError("Небезопасный SQL запрос")
        
        results = await self.vector_analytics.execute_direct_query(sql_query, db_path)
        
        return [TextContent(
            type="text",
            text=json.dumps(results, indent=2, ensure_ascii=False)
        )]

    async def run(self):
        """Запуск MCP сервера"""
        logger.info("Запуск RAG MCP Server...")
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="rag-vector-service",
                    server_version="1.0.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability()
                    ),
                ),
            )

def main():
    """Entry point"""
    rag_server = RAGMCPServer()
    asyncio.run(rag_server.run())

if __name__ == "__main__":
    main() 