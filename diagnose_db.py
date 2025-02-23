import os
import time
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import logging
import sqlite3

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

def test_db_connection():
    logger.info("Тестирование подключения к ChromaDB...")
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        logger.info("✓ Подключение к ChromaDB успешно")
        return client
    except Exception as e:
        logger.error(f"✗ Ошибка подключения к ChromaDB: {str(e)}")
        return None

def test_collection_access(client):
    logger.info("\nПроверка доступа к коллекции...")
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name=os.getenv('EMBEDDING_MODEL', "text-embedding-3-large")
        )
        
        collection = client.get_collection(
            name=os.getenv('CHROMA_COLLECTION_NAME', 'confluence_eh'),
            embedding_function=openai_ef
        )
        count = collection.count()
        logger.info(f"✓ Доступ к коллекции успешен")
        logger.info(f"✓ Количество документов: {count}")
        return collection
    except Exception as e:
        logger.error(f"✗ Ошибка доступа к коллекции: {str(e)}")
        return None

def test_query_performance(collection):
    logger.info("\nТестирование производительности запросов...")
    
    test_queries = [
        "Что такое VPN?",
        "Как получить доступ к системе?",
        "Какие есть бенефиты?",
        "Структура компании",
        "Процесс онбординга"
    ]
    
    total_time = 0
    query_count = 0
    
    for query in test_queries:
        try:
            logger.info(f"\nТестовый запрос: '{query}'")
            
            logger.info("Состояние коллекции:")
            logger.info(f"- Количество документов: {collection.count()}")
            logger.info(f"- Имя коллекции: {collection.name}")
            logger.info(f"- Метаданные коллекции: {collection.metadata}")
            
            embed_start = time.time()
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv('OPENAI_API_KEY'),
                model_name=os.getenv('EMBEDDING_MODEL', "text-embedding-3-large")
            )
            query_embedding = openai_ef([query])
            embed_time = time.time() - embed_start
            logger.info(f"✓ Время получения эмбеддинга: {embed_time:.2f} сек")
            
            prep_start = time.time()
            query_params = {
                "query_texts": [query],
                "n_results": 5,
                "include": ['documents', 'metadatas']
            }
            prep_time = time.time() - prep_start
            logger.info(f"✓ Время подготовки запроса: {prep_time:.2f} сек")
            
            query_start = time.time()
            results = collection.query(**query_params)
            query_time = time.time() - query_start
            
            total_time += query_time
            query_count += 1
            
            doc_count = len(results['documents'][0]) if results['documents'] else 0
            logger.info(f"✓ Найдено документов: {doc_count}")
            logger.info(f"✓ Время запроса к ChromaDB: {query_time:.2f} сек")
            logger.info(f"✓ Общее время выполнения: {embed_time + prep_time + query_time:.2f} сек")
            
            if doc_count > 0:
                logger.info("Первый документ:")
                logger.info(f"- ID: {results['ids'][0][0]}")
                logger.info(f"- Заголовок: {results['metadatas'][0][0].get('title', 'Нет заголовка')}")
                logger.info(f"- Размер контента: {len(results['documents'][0][0])} символов")
                logger.info(f"- Метаданные: {results['metadatas'][0][0]}")
        
        except Exception as e:
            logger.error(f"✗ Ошибка при выполнении запроса '{query}': {str(e)}")
            import traceback
            logger.error(f"Stacktrace: {traceback.format_exc()}")
    
    if query_count > 0:
        avg_time = total_time / query_count
        logger.info(f"\nСреднее время запроса к ChromaDB: {avg_time:.2f} сек")

def check_db_size():
    logger.info("\nПроверка размера базы данных...")
    try:
        db_path = "./chroma_db/chroma.sqlite3"
        size_bytes = os.path.getsize(db_path)
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"✓ Размер базы данных: {size_mb:.2f} MB")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        total_size = page_count * page_size / (1024 * 1024)
        
        logger.info(f"✓ Размер страницы SQLite: {page_size} bytes")
        logger.info(f"✓ Количество страниц: {page_count}")
        logger.info(f"✓ Расчетный размер: {total_size:.2f} MB")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute(f"SELECT count(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            logger.info(f"Таблица {table[0]}: {count} записей")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"✗ Ошибка при проверке размера базы: {str(e)}")
        import traceback
        logger.error(f"Stacktrace: {traceback.format_exc()}")

def check_embeddings(collection):
    logger.info("\nПроверка эмбеддингов...")
    try:
        start_time = time.time()
        results = collection.get(
            limit=1,
            include=['embeddings', 'documents', 'metadatas']
        )
        get_time = time.time() - start_time
        logger.info(f"✓ Время получения эмбеддинга: {get_time:.2f} сек")
        
        logger.info("\nСтруктура ответа ChromaDB:")
        logger.info(f"Ключи в ответе: {list(results.keys())}")
        logger.info(f"Количество документов: {len(results.get('documents', []))}")
        logger.info(f"Количество метаданных: {len(results.get('metadatas', []))}")
        logger.info(f"Тип embeddings: {type(results.get('embeddings'))}")
        
        embeddings = results.get('embeddings')
        if embeddings is not None:
            if hasattr(embeddings, 'shape'):
                logger.info(f"Размер массива embeddings: {embeddings.shape}")
                if embeddings.shape[0] > 0:
                    logger.info(f"Размерность первого эмбеддинга: {embeddings.shape[1]}")
                    logger.info(f"✓ Первые три значения: {embeddings[0][:3].tolist()}")
                    
                    logger.info("\nПроверка времени ответа OpenAI API...")
                    start_time = time.time()
                    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.getenv('OPENAI_API_KEY'),
                        model_name=os.getenv('EMBEDDING_MODEL', "text-embedding-3-large")
                    )
                    test_result = openai_ef(["тестовый запрос"])
                    api_time = time.time() - start_time
                    logger.info(f"✓ Время ответа API: {api_time:.2f} сек")
                    
                    if api_time > 5:
                        logger.warning("⚠️ Время ответа API превышает норму (>5 сек)")
                else:
                    logger.error("✗ Массив embeddings пуст")
            else:
                logger.error("✗ Embeddings не являются numpy массивом")
        else:
            logger.error("✗ Embeddings отсутствуют")
    
    except Exception as e:
        logger.error(f"✗ Ошибка при проверке эмбеддингов: {str(e)}")
        import traceback
        logger.error(f"Stacktrace: {traceback.format_exc()}")

def main():
    logger.info("=== Диагностика ChromaDB ===")
    logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    client = test_db_connection()
    if not client:
        return
    
    collection = test_collection_access(client)
    if not collection:
        return
    
    check_db_size()
    check_embeddings(collection)
    test_query_performance(collection)

if __name__ == "__main__":
    main() 