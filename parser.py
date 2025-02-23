import os
import requests
from requests.auth import HTTPBasicAuth
import urllib3
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
import json
import shutil
import time
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parser.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
    
    content = soup.find('div', class_='wiki-content')
    if not content:
        content = soup
        
    for link in content.find_all('a'):
        href = link.get('href', '')
        if href and not href.startswith(('#', 'javascript:')):
            text = link.get_text(strip=True)
            if text and text != href:
                link.replace_with(f" {text} ({href}) ")
            else:
                link.replace_with(f" {href} ")
    
    for tag in content.find_all(True):
        if tag.name in ['br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            tag.replace_with(f"\n{tag.get_text()}\n")
        else:
            tag.replace_with(f" {tag.get_text()} ")
    
    text = content.get_text()
    
    lines = []
    for line in text.split('\n'):
        line = ' '.join(word for word in line.split() if word)
        if line:
            lines.append(line)
    
    return '\n'.join(lines)

def extract_page_id_from_url(url):
    if not url:
        return None
    
    if 'pageId=' in url:
        return url.split('pageId=')[-1].split('&')[0]
    
    if '/pages/' in url:
        parts = url.split('/pages/')[-1].split('/')
        if parts and parts[0].isdigit():
            return parts[0]
    
    return None

def get_page_structure(session, confluence_url, page_id):
    response = session.get(
        f"{confluence_url}/rest/api/content/{page_id}",
        headers={'Accept': 'application/json'},
        params={
            'expand': 'ancestors,children.page,version'
        }
    )
    
    if response.status_code != 200:
        logger.error(f"Ошибка при получении структуры страницы {page_id}: {response.status_code}")
        return None
    
    data = response.json()
    
    parent_id = None
    if data.get('ancestors'):
        parent_id = data['ancestors'][-1]['id']
    
    child_ids = []
    if data.get('children') and data['children'].get('page'):
        child_ids = [child['id'] for child in data['children']['page']['results']]
    
    page_url = f"{confluence_url}/pages/viewpage.action?pageId={page_id}"
    page_response = session.get(page_url)
    
    linked_ids = set()
    if page_response.status_code == 200:
        soup = BeautifulSoup(page_response.text, 'html.parser')
        main_content = soup.find('div', class_='wiki-content')
        
        if main_content:
            for link in main_content.find_all('a'):
                href = link.get('href', '')
                linked_id = extract_page_id_from_url(href)
                if linked_id:
                    linked_ids.add(linked_id)
    
    return {
        'parent_id': parent_id,
        'child_ids': child_ids,
        'linked_ids': list(linked_ids - {page_id}),
        'version': data['version']['number']
    }

def get_page_content(session, confluence_url, page_id, space_info):
    structure = get_page_structure(session, confluence_url, page_id)
    if not structure:
        return None
    
    page_url = f"{confluence_url}/pages/viewpage.action?pageId={page_id}"
    page_response = session.get(page_url)
    
    if page_response.status_code != 200:
        logger.error(f"Ошибка при получении страницы {page_id}: {page_response.status_code}")
        return None
    
    soup = BeautifulSoup(page_response.text, 'html.parser')
    main_content = soup.find('div', class_='wiki-content')
    
    if not main_content:
        logger.error(f"Не найден основной контент на странице {page_id}")
        return None
    
    title_elem = soup.find('title')
    title = title_elem.get_text().split(' - ')[0] if title_elem else 'Untitled'
    
    return {
        'id': page_id,
        'title': title,
        'url': f"{confluence_url}/spaces/{space_info['key']}/pages/{page_id}",
        'version': structure['version'],
        'raw_content': clean_html_content(str(main_content)),
        'space_key': space_info['key'],
        'space_name': space_info['name'],
        'space_url': f"{confluence_url}/spaces/{space_info['key']}",
        'parent_id': structure['parent_id'],
        'child_ids': structure['child_ids'],
        'linked_ids': structure['linked_ids']
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def add_to_collection(collection, content, metadata, doc_id):
    if not content or len(content.strip()) == 0:
        logger.warning(f"Пропуск пустой страницы: {metadata['title']}")
        return
        
    try:
        processed_metadata = {
            'title': metadata['title'],
            'url': metadata['url'],
            'version': str(metadata['version']),
            'space_key': metadata['space_key'],
            'space_name': metadata['space_name'],
            'space_url': metadata['space_url'],
            'parent_id': str(metadata.get('parent_id', '')),
            'child_ids': ','.join(metadata.get('child_ids', [])) or '',
            'linked_ids': ','.join(metadata.get('linked_ids', [])) or '',
        }
        
        collection.add(
            documents=[content],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        logger.info(f"Сохранена страница: {metadata['title']}")
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении страницы {metadata['title']}: {str(e)}")
        if "APIStatusError" in str(e):
            return
        raise

def get_child_pages(session, confluence_url, page_id, space_info, chroma_collection, processed_pages=None):
    if processed_pages is None:
        processed_pages = set()
        
    if page_id in processed_pages:
        return
        
    processed_pages.add(page_id)
    
    try:
        page_data = get_page_content(session, confluence_url, page_id, space_info)
        if page_data:
            metadata = {
                'title': page_data['title'],
                'url': page_data['url'],
                'version': str(page_data['version']),
                'space_key': page_data['space_key'],
                'space_name': page_data['space_name'],
                'space_url': page_data['space_url'],
                'parent_id': page_data['parent_id'],
                'child_ids': page_data['child_ids'],
                'linked_ids': page_data['linked_ids']
            }
            
            add_to_collection(
                chroma_collection,
                page_data['raw_content'],
                metadata,
                page_data['id']
            )
            
            for child_id in page_data['child_ids']:
                get_child_pages(session, confluence_url, child_id, space_info, chroma_collection, processed_pages)
            
            time.sleep(0.5)
            
    except Exception as e:
        logger.error(f"Ошибка при обработке страницы {page_id}: {str(e)}")

def get_confluence_pages():
    logger.info("Начало парсинга Confluence")
    
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        logger.info("Удалена старая база ChromaDB")
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-large"
    )
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.create_collection(
        name="confluence_eh",
        embedding_function=openai_ef
    )
    logger.info("Создана новая коллекция в ChromaDB")
    
    basic_auth = HTTPBasicAuth(
        os.getenv('BASIC_AUTH_USERNAME'),
        os.getenv('BASIC_AUTH_PASSWORD')
    )
    
    confluence_url = 'https://confluence-rp.teamslc.net'
    
    session = requests.Session()
    session.auth = basic_auth
    session.verify = False
    
    try:
        login_response = session.get(f"{confluence_url}/login.action")
        logger.info(f"Статус входа: {login_response.status_code}")
        
        login_data = {
            'os_username': os.getenv('CONFLUENCE_USERNAME'),
            'os_password': os.getenv('CONFLUENCE_PASSWORD'),
            'os_destination': '/index.action',
            'login': 'Log in'
        }
        
        login_result = session.post(
            f"{confluence_url}/dologin.action",
            data=login_data
        )
        
        if login_result.status_code == 200:
            logger.info("Успешный вход в систему")
            
            spaces_response = session.get(
                f"{confluence_url}/rest/api/space",
                headers={'Accept': 'application/json'},
                params={'expand': 'homepage'}
            )
            
            if spaces_response.status_code == 200:
                spaces = spaces_response.json()['results']
                logger.info(f"\nНайдено пространств: {len(spaces)}")
                
                for space in spaces:
                    homepage = space.get('homepage', {})
                    if homepage:
                        logger.info(f"\nОбработка пространства: {space['name']} ({space['key']})")
                        
                        space_info = {
                            'key': space['key'],
                            'name': space['name'],
                            'type': space.get('type', 'global')
                        }
                        
                        get_child_pages(session, confluence_url, homepage['id'], space_info, collection)
                        
                        logger.info(f"Завершено пространство: {space['key']}")
                
                logger.info("\nПарсинг всех пространств завершен")
            else:
                logger.error(f"Ошибка при получении списка пространств: {spaces_response.status_code}")
        else:
            logger.error("Ошибка входа в систему")
            
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        logger.exception("Детали ошибки:")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Начало выполнения скрипта: {start_time}")
    
    try:
        get_confluence_pages()
    except Exception as e:
        logger.error("Критическая ошибка при выполнении скрипта")
        logger.exception(str(e))
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Завершение выполнения скрипта: {end_time}")
        logger.info(f"Общее время выполнения: {duration}") 