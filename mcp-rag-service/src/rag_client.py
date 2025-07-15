"""
Клиент для взаимодействия с Python RAG Server
Простая реализация HTTP клиента
"""

import aiohttp
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RAGClient:
    """HTTP клиент для RAG сервера"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получить HTTP сессию"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def upload_file(self, file_path: str, rag_server_url: str) -> Dict[str, Any]:
        """
        Загрузить файл в RAG сервер
        
        Args:
            file_path: Путь к файлу
            rag_server_url: URL RAG сервера
            
        Returns:
            Результат загрузки
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Указанный путь не является файлом: {file_path}")
        
        # Проверяем расширение файла
        allowed_extensions = {
            ".txt", ".md", ".pdf", ".py", ".js", ".java", 
            ".c", ".cpp", ".json", ".yaml", ".yml", ".ini", ".toml"
        }
        
        if file_path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Неподдерживаемый тип файла: {file_path.suffix}")
        
        session = await self._get_session()
        
        try:
            # Загружаем файл
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('files', f, filename=file_path.name)
                
                upload_url = f"{rag_server_url.rstrip('/')}/upload-files/"
                
                async with session.post(upload_url, data=data) as response:
                    if response.status != 200:
                        raise Exception(f"Ошибка загрузки файла: HTTP {response.status}")
                    
                    logger.info(f"Файл {file_path.name} успешно загружен")
            
            # Обрабатываем файл
            process_url = f"{rag_server_url.rstrip('/')}/process-files/"
            
            async with session.post(process_url) as response:
                if response.status != 200:
                    raise Exception(f"Ошибка обработки файла: HTTP {response.status}")
                
                logger.info(f"Файл {file_path.name} успешно обработан")
            
            return {
                "status": "success",
                "message": f"Файл {file_path.name} загружен и обработан",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Ошибка при работе с файлом {file_path}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "file_name": file_path.name
            }
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        search_type: str = "hybrid",
        use_reranker: bool = True,
        expand_query: bool = False,
        rag_server_url: str = "http://host.docker.internal:8000",
    ) -> List[Dict[str, Any]]:
        """
        Выполнить поиск через JSON API
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            search_type: Тип поиска ('hybrid', 'semantic', 'keyword')
            use_reranker: Использовать ли переранжирование
            expand_query: Использовать ли расширение запроса
            rag_server_url: URL RAG сервера
            
        Returns:
            Список результатов поиска
        """
        session = await self._get_session()
        try:
            # Используем новый JSON API endpoint
            search_url = f"{rag_server_url.rstrip('/')}/api/search"
            
            # Отправляем запрос с параметрами (boolean конвертируем в строки)
            params = {
                "query": query,
                "top_k": top_k,
                "search_type": search_type,
                "use_reranker": str(use_reranker).lower(),
                "expand_query": str(expand_query).lower(),
            }
            
            async with session.post(search_url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Ошибка поиска: HTTP {response.status}, {await response.text()}")
                
                result = await response.json()
                
                # Проверяем наличие ошибки в ответе
                if "error" in result:
                    raise Exception(f"Ошибка сервера: {result['error']}")
                
                logger.info(f"Поиск выполнен для запроса: '{query}', найдено: {result.get('total_results', 0)} результатов")
                
                return result.get("results", [])
                
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            raise
    
    async def close(self):
        """Закрыть HTTP сессию"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close() 