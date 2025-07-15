"""
Утилиты для MCP RAG сервиса
Валидация, безопасность, хелперы
"""

import re
import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """
    Валидация пути к файлу
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        True если путь валидный
    """
    if not file_path or not isinstance(file_path, str):
        return False
    
    try:
        path = Path(file_path)
        
        # Проверяем, что путь не содержит опасные последовательности
        path_str = str(path.resolve())
        dangerous_patterns = ['..', '~', '$']
        
        for pattern in dangerous_patterns:
            if pattern in path_str:
                logger.warning(f"Опасный паттерн в пути: {pattern}")
                return False
        
        # Проверяем расширение файла
        allowed_extensions = {
            ".txt", ".md", ".pdf", ".py", ".js", ".java", 
            ".c", ".cpp", ".json", ".yaml", ".yml", ".ini", ".toml"
        }
        
        if path.suffix.lower() not in allowed_extensions:
            logger.warning(f"Неподдерживаемое расширение: {path.suffix}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка валидации пути {file_path}: {e}")
        return False

def safe_sql_query(sql_query: str) -> bool:
    """
    Проверка безопасности SQL запроса
    Простая валидация против SQL injection
    
    Args:
        sql_query: SQL запрос для проверки
        
    Returns:
        True если запрос считается безопасным
    """
    if not sql_query or not isinstance(sql_query, str):
        return False
    
    # Приводим к нижнему регистру для проверки
    query_lower = sql_query.lower().strip()
    
    # Разрешенные команды (только для чтения)
    allowed_commands = {
        'select', 'with', 'show', 'describe', 'explain'
    }
    
    # Запрещенные команды
    forbidden_commands = {
        'insert', 'update', 'delete', 'drop', 'create', 'alter',
        'truncate', 'replace', 'merge', 'exec', 'execute',
        'declare', 'set', 'use', 'grant', 'revoke'
    }
    
    # Проверяем первое слово (основную команду)
    first_word = query_lower.split()[0] if query_lower.split() else ""
    
    if first_word not in allowed_commands:
        logger.warning(f"Неразрешенная SQL команда: {first_word}")
        return False
    
    # Проверяем на наличие запрещенных команд в любом месте запроса
    for forbidden in forbidden_commands:
        if f" {forbidden} " in f" {query_lower} ":
            logger.warning(f"Обнаружена запрещенная команда: {forbidden}")
            return False
    
    # Проверяем на потенциально опасные паттерны
    dangerous_patterns = [
        r'--',  # SQL комментарии
        r'/\*',  # Многострочные комментарии  
        r'\*/',
        r';.*',  # Множественные команды
        r'union.*select',  # UNION injection
        r'or.*1=1',  # OR injection
        r'and.*1=1',  # AND injection
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower):
            logger.warning(f"Обнаружен опасный паттерн: {pattern}")
            return False
    
    # Проверяем длину запроса (защита от чрезмерно сложных запросов)
    if len(sql_query) > 2000:
        logger.warning(f"Слишком длинный SQL запрос: {len(sql_query)} символов")
        return False
    
    return True

def format_similarity_score(similarity: float) -> str:
    """
    Форматирование similarity score с интерпретацией
    Основано на thresholds из user_rule.txt
    
    Args:
        similarity: Значение similarity (0.0-1.0)
        
    Returns:
        Отформатированная строка с интерпретацией
    """
    if similarity >= 0.9:
        interpretation = "практически идентичный"
    elif similarity >= 0.8:
        interpretation = "очень похожий"
    elif similarity >= 0.7:
        interpretation = "семантически связанный"
    elif similarity >= 0.6:
        interpretation = "умеренно связанный"
    elif similarity >= 0.5:
        interpretation = "слабая связь"
    elif similarity >= 0.3:
        interpretation = "очень слабая связь"
    else:
        interpretation = "несвязанный"
    
    return f"{similarity:.4f} ({interpretation})"

def get_similarity_threshold_recommendations() -> dict:
    """
    Возвращает рекомендуемые пороги similarity для разных задач
    Из user_rule.txt
    """
    return {
        "search_relevant": 0.6,
        "find_similar": 0.7,
        "detect_duplicates": 0.9,
        "clustering": 0.5,
        "recommendations": 0.6
    }

def validate_mcp_arguments(args: dict, required_fields: List[str], optional_fields: List[str] = None) -> bool:
    """
    Валидация аргументов MCP функции
    
    Args:
        args: Словарь аргументов
        required_fields: Список обязательных полей
        optional_fields: Список опциональных полей
        
    Returns:
        True если аргументы валидны
    """
    if not isinstance(args, dict):
        logger.error("Аргументы должны быть словарем")
        return False
    
    # Проверяем обязательные поля
    for field in required_fields:
        if field not in args:
            logger.error(f"Отсутствует обязательное поле: {field}")
            return False
        
        if args[field] is None or (isinstance(args[field], str) and not args[field].strip()):
            logger.error(f"Пустое значение для обязательного поля: {field}")
            return False
    
    # Проверяем, что нет неожиданных полей
    all_allowed = set(required_fields + (optional_fields or []))
    unexpected = set(args.keys()) - all_allowed
    
    if unexpected:
        logger.warning(f"Неожиданные поля в аргументах: {unexpected}")
    
    return True

def sanitize_filename(filename: str) -> str:
    """
    Очистка имени файла от опасных символов
    
    Args:
        filename: Исходное имя файла
        
    Returns:
        Безопасное имя файла
    """
    if not filename:
        return "unknown"
    
    # Удаляем опасные символы
    dangerous_chars = '<>:"/\\|?*'
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Удаляем точки в начале (скрытые файлы)
    sanitized = sanitized.lstrip('.')
    
    # Ограничиваем длину
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized or "unknown"

def create_error_response(error_message: str, error_code: str = "GENERAL_ERROR") -> dict:
    """
    Создание стандартного ответа об ошибке
    
    Args:
        error_message: Сообщение об ошибке
        error_code: Код ошибки
        
    Returns:
        Словарь с информацией об ошибке
    """
    return {
        "status": "error",
        "error_code": error_code,
        "message": error_message,
        "timestamp": None  # TODO: добавить timestamp если нужен
    }

def create_success_response(data: any, message: str = "Операция выполнена успешно") -> dict:
    """
    Создание стандартного ответа об успехе
    
    Args:
        data: Данные результата
        message: Сообщение об успехе
        
    Returns:
        Словарь с результатом операции
    """
    return {
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": None  # TODO: добавить timestamp если нужен
    } 