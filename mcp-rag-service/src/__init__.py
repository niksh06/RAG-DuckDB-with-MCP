"""
MCP RAG Service

MCP сервис для интеграции с Python RAG Server (DuckDB VSS)
Простая и эффективная интеграция векторного поиска по принципу KISS
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .rag_mcp_server import RAGMCPServer
from .rag_client import RAGClient
from .vector_operations import VectorAnalytics
from .utils import (
    validate_file_path,
    safe_sql_query,
    format_similarity_score,
    get_similarity_threshold_recommendations,
)

__all__ = [
    "RAGMCPServer",
    "RAGClient", 
    "VectorAnalytics",
    "validate_file_path",
    "safe_sql_query",
    "format_similarity_score",
    "get_similarity_threshold_recommendations",
] 