#!/usr/bin/env python3
"""
Пример загрузки файлов через MCP RAG Service
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

async def upload_file_example():
    """Пример загрузки файла в RAG сервер"""
    
    # Создаем клиент
    async with RAGClient() as client:
        
        # Примеры файлов для загрузки
        test_files = [
            "example.txt",
            "config.json", 
            "script.py",
            "README.md"
        ]
        
        for file_name in test_files:
            file_path = f"/tmp/{file_name}"
            
            # Создаем тестовый файл
            Path(file_path).write_text(f"""
            Это тестовый файл: {file_name}
            
            Содержимое для демонстрации работы RAG сервиса.
            Файл типа: {Path(file_name).suffix}
            
            Этот файл будет обработан и добавлен в векторную базу данных.
            """)
            
            try:
                logger.info(f"Загружаем файл: {file_name}")
                
                result = await client.upload_file(
                    file_path=file_path,
                    rag_server_url="http://localhost:8000"
                )
                
                logger.info(f"Результат загрузки {file_name}: {result}")
                
            except Exception as e:
                logger.error(f"Ошибка загрузки {file_name}: {e}")
            
            finally:
                # Удаляем тестовый файл
                try:
                    Path(file_path).unlink()
                except:
                    pass

async def main():
    """Главная функция"""
    logger.info("Начинаем пример загрузки файлов...")
    
    try:
        await upload_file_example()
        logger.info("Пример загрузки завершен успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка в примере: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 