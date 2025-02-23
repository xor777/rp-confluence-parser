import os
import logging
import chromadb
from datetime import datetime
from typing import Dict, Optional, List, Set
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

class Constants:
    DEFAULT_SYSTEM_PROMPT = "Я помогу найти нужную информацию в базе знаний компании."
    DEFAULT_USER_PROMPT = "Контекст:\n{context}\n\nВопрос: {question}\n\nНапиши ответ."
    MAX_RESPONSE_TOKENS = int(os.getenv('GPT_MAX_TOKENS', '500'))
    GPT_TEMPERATURE = float(os.getenv('GPT_TEMPERATURE', '0.0'))
    GPT_CONTEXT_RESERVE = int(os.getenv('GPT_CONTEXT_RESERVE', '1000'))
    
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
    DEFAULT_CHAT_MODEL = "gpt-4o"
    
    DEFAULT_COLLECTION_NAME = "confluence_eh"
    
    DEFAULT_RELEVANCE_THRESHOLD = 0.3
    DEFAULT_MAX_DOCS = 15
    DEFAULT_DOC_CHUNK_SIZE = 5000
    DEFAULT_BRANCH_MAX_DEPTH = 2
    
    TYPING_UPDATE_INTERVAL = int(os.getenv('TYPING_UPDATE_INTERVAL', '4'))
    
    NO_DOCUMENTS_MESSAGE = (
        "К сожалению, в базе знаний не найдено информации по вашему запросу.\n\n"
        "Попробуйте, пожалуйста:\n"
        "• Переформулировать вопрос\n"
        "• Использовать другие ключевые слова\n"
        "• Уточнить детали запроса"
    )
    TECHNICAL_ERROR_MESSAGE = (
        "😔 Произошла техническая ошибка при обработке запроса. "
        "Повторите запрос позже."
    )
    
    WELCOME_MESSAGE = (
        "🤖 База знаний компании\n\n"
        "Я помогу вам быстро найти нужную информацию в корпоративной базе знаний.\n\n"
        "Задайте свой вопрос, и я постараюсь помочь!\n"
    )
    
    API_TIMEOUT_MESSAGE = (
        "😔 В данный момент сервис работает медленнее обычного. Так бывает когда медленно отвечает OpenAI.\n"
        "Повторите запрос через некоторое время."
    )
    API_ERROR_MESSAGE = (
        "🔧 Извините, возникли технические проблемы при обработке запроса.\n"
        "Попробуйте еще раз позже."
    )
    
    API_TIMEOUT_SECONDS = 30.0
    QUERY_TIMEOUT_SECONDS = 30.0

class Branch:
    def __init__(self, main_doc_id: str, docs: Dict[str, Dict], scores: Dict[str, float]):
        self.main_doc_id = main_doc_id
        self.docs = docs
        self.scores = scores
        
    @property
    def branch_score(self) -> float:
        if not self.scores:
            return 0.0
        main_score = self.scores.get(self.main_doc_id, 0.0) * 1.5
        other_scores = [score for doc_id, score in self.scores.items() if doc_id != self.main_doc_id]
        if not other_scores:
            return main_score
        return (main_score + sum(other_scores)) / (1 + len(other_scores))
    
    def merge(self, other: 'Branch') -> 'Branch':
        merged_docs = {**self.docs, **other.docs}
        merged_scores = {**self.scores}
        for doc_id, score in other.scores.items():
            if doc_id in merged_scores:
                merged_scores[doc_id] = max(merged_scores[doc_id], score)
            else:
                merged_scores[doc_id] = score
        main_doc_id = self.main_doc_id
        if other.scores.get(other.main_doc_id, 0.0) > self.scores.get(self.main_doc_id, 0.0):
            main_doc_id = other.main_doc_id
        return Branch(main_doc_id, merged_docs, merged_scores)
    
    def has_common_docs(self, other: 'Branch') -> bool:
        return bool(set(self.docs.keys()) & set(other.docs.keys()))
    
    @property
    def size(self) -> int:
        return len(self.docs)

class KnowledgeBase:
    def __init__(self, db_path="./chroma_db"):
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name=os.getenv('EMBEDDING_MODEL', Constants.DEFAULT_EMBEDDING_MODEL)
        )
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(
            name=os.getenv('CHROMA_COLLECTION_NAME', Constants.DEFAULT_COLLECTION_NAME),
            embedding_function=self.openai_ef
        )
        
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.chat_model = os.getenv('CHAT_MODEL', Constants.DEFAULT_CHAT_MODEL)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        
        self.relevance_threshold = float(os.getenv('RELEVANCE_THRESHOLD', str(Constants.DEFAULT_RELEVANCE_THRESHOLD)))
        self.max_docs = int(os.getenv('MAX_DOCS', str(Constants.DEFAULT_MAX_DOCS)))
        self.doc_chunk_size = int(os.getenv('DOC_CHUNK_SIZE', str(Constants.DEFAULT_DOC_CHUNK_SIZE)))
        self.branch_max_depth = int(os.getenv('BRANCH_MAX_DEPTH', str(Constants.DEFAULT_BRANCH_MAX_DEPTH)))
        
        try:
            with open('system_prompt.txt', 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except Exception as e:
            logger.error(f"ошибка при загрузке system_prompt.txt: {str(e)}")
            self.system_prompt = Constants.DEFAULT_SYSTEM_PROMPT
            
        try:
            with open('user_prompt.txt', 'r', encoding='utf-8') as f:
                self.user_prompt_template = f.read().strip()
        except Exception as e:
            logger.error(f"ошибка при загрузке user_prompt.txt: {str(e)}")
            self.user_prompt_template = Constants.DEFAULT_USER_PROMPT
        
        self.system_tokens = len(self.tokenizer.encode(self.system_prompt))
        self.max_context_tokens = 128000 - self.system_tokens - Constants.GPT_CONTEXT_RESERVE

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        norm1 = sum(x * x for x in vec1) ** 0.5
        norm2 = sum(x * x for x in vec2) ** 0.5
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0
    
    def compute_similarity(self, query: str, text: str) -> float:
        query_embedding = self.openai_ef([query])[0]
        text_embedding = self.openai_ef([text])[0]
        return self.compute_cosine_similarity(query_embedding, text_embedding)
    
    def filter_related_docs(self, doc_ids: List[str], question: str) -> List[Dict]:
        if not doc_ids:
            return []
            
        logger.info(f"фильтрация {len(doc_ids)} связанных документов...")
        
        query_embedding = self.openai_ef([question])[0]
        
        results = self.collection.get(
            ids=doc_ids,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        filtered_docs = []
        for i, doc_id in enumerate(results['ids']):
            content = results['documents'][i][:self.doc_chunk_size] if results['documents'][i] else ""
            doc_embedding = results['embeddings'][i]
            
            score = self.compute_cosine_similarity(query_embedding, doc_embedding)
            
            logger.info(f"документ {doc_id}: релевантность {score:.3f}")
            
            if score >= self.relevance_threshold:
                filtered_docs.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': results['metadatas'][i]
                })
                logger.info(f"добавлен релевантный документ: {doc_id}")
        
        logger.info(f"отфильтровано документов: {len(filtered_docs)} из {len(doc_ids)}")
        return filtered_docs
    
    def expand_context(self, initial_docs: List[Dict], question: str) -> List[Dict]:
        logger.info("расширение контекста...")
        all_docs = []
        processed_ids = set()
        
        for doc in initial_docs:
            doc['content'] = doc['content'][:self.doc_chunk_size] if doc['content'] else ""
            all_docs.append(doc)
            processed_ids.add(doc['id'])
        
        for doc in initial_docs:
            if len(all_docs) >= self.max_docs:
                break
                
            m = doc['metadata']
            related_ids = []
            
            if m.get('parent_id'):
                related_ids.append(m['parent_id'].strip())
            
            if m.get('child_ids'):
                child_list = (m['child_ids'].split(',') if isinstance(m['child_ids'], str) 
                            else m['child_ids'] if isinstance(m['child_ids'], list) else [])
                related_ids.extend([c.strip() for c in child_list if c.strip()])
            
            if m.get('linked_ids'):
                linked_list = (m['linked_ids'].split(',') if isinstance(m['linked_ids'], str)
                             else m['linked_ids'] if isinstance(m['linked_ids'], list) else [])
                related_ids.extend([l.strip() for l in linked_list if l.strip()])
            
            related_ids = [rid for rid in related_ids if rid not in processed_ids]
            if related_ids:
                filtered_docs = self.filter_related_docs(related_ids, question)
                
                for rd in filtered_docs:
                    if len(all_docs) < self.max_docs and rd['id'] not in processed_ids:
                        all_docs.append(rd)
                        processed_ids.add(rd['id'])
        
        logger.info(f"итоговое количество документов: {len(all_docs)}")
        return all_docs
    
    def format_confluence_url(self, url: str) -> str:
        if not url:
            return ""
        
        if "/spaces/" in url:
            parts = url.split("/")
            if len(parts) >= 7:
                page_id = parts[-1]
                base_url = "/".join(parts[0:3])
                return f"{base_url}/pages/viewpage.action?pageId={page_id}"
        return url
    
    def escape_markdown(self, text: str) -> str:
        escape_chars = ['[', ']', '(', ')', '_', '*', '`']
        for char in escape_chars:
            text = text.replace(char, '\\' + char)
        return text
    
    def format_context(self, docs: List[Dict]) -> str:
        context_parts = []
        total_tokens = 0
        
        for doc in docs:
            title = self.escape_markdown(doc['metadata'].get('title', ''))
            url = self.format_confluence_url(doc['metadata'].get('url', ''))
            content = self.escape_markdown(doc['content'])
            
            doc_part = f"=== {title} ===\n{content}"
            if url:
                doc_part += f"\nСсылка: {url}"
            doc_part += "\n" + "-" * 50 + "\n"
            
            doc_tokens = self.count_tokens(doc_part)
            
            if total_tokens + doc_tokens > self.max_context_tokens:
                logger.info(f"достигнут лимит токенов контекста ({total_tokens})")
                break
            
            context_parts.append(doc_part)
            total_tokens += doc_tokens
        
        logger.info(f"всего токенов в контексте: {total_tokens}")
        return "\n".join(context_parts)
    
    def parse_id_list(self, id_field: str | List[str]) -> List[str]:
        if not id_field:
            return []
        if isinstance(id_field, str):
            return [id.strip() for id in id_field.split(',') if id.strip()]
        if isinstance(id_field, list):
            return [str(id).strip() for id in id_field if str(id).strip()]
        return []

    def get_related_ids(self, metadata: Dict) -> List[str]:
        related_ids = []
        
        if metadata.get('parent_id'):
            parent_id = metadata['parent_id'].strip()
            if parent_id:
                related_ids.append(parent_id)
        
        child_ids = self.parse_id_list(metadata.get('child_ids', []))
        related_ids.extend(child_ids)
        
        linked_ids = self.parse_id_list(metadata.get('linked_ids', []))
        related_ids.extend(linked_ids)
        
        return list(set(related_ids))

    def build_branch(self, main_doc: Dict, question_embedding: List[float]) -> Branch:
        docs = {main_doc['id']: main_doc}
        scores = {}
        processed_ids = {main_doc['id']}
        docs_to_process = [(main_doc, 0)]
        
        while docs_to_process:
            current_doc, depth = docs_to_process.pop(0)
            if depth >= self.branch_max_depth:
                continue
            
            related_ids = self.get_related_ids(current_doc['metadata'])
            related_ids = [rid for rid in related_ids if rid not in processed_ids]
            
            if related_ids:
                results = self.collection.get(
                    ids=related_ids,
                    include=['documents', 'metadatas', 'embeddings']
                )
                
                for i, doc_id in enumerate(results['ids']):
                    if doc_id in processed_ids:
                        continue
                        
                    doc_embedding = results['embeddings'][i]
                    score = self.compute_similarity_with_embeddings(question_embedding, doc_embedding)
                    
                    if score >= self.relevance_threshold:
                        doc_data = {
                            'id': doc_id,
                            'content': results['documents'][i][:self.doc_chunk_size],
                            'metadata': results['metadatas'][i]
                        }
                        docs[doc_id] = doc_data
                        scores[doc_id] = score
                        processed_ids.add(doc_id)
                        docs_to_process.append((doc_data, depth + 1))
        
        main_doc_embedding = self.collection.get(
            ids=[main_doc['id']],
            include=['embeddings']
        )['embeddings'][0]
        scores[main_doc['id']] = self.compute_similarity_with_embeddings(
            question_embedding, main_doc_embedding
        )
        
        return Branch(main_doc['id'], docs, scores)
    
    def merge_overlapping_branches(self, branches: List[Branch]) -> List[Branch]:
        if not branches:
            return []
            
        result = []
        processed = set()
        
        for i, branch in enumerate(branches):
            if i in processed:
                continue
                
            current_branch = branch
            processed.add(i)
            
            for j, other_branch in enumerate(branches):
                if j in processed:
                    continue
                    
                if current_branch.has_common_docs(other_branch):
                    current_branch = current_branch.merge(other_branch)
                    processed.add(j)
            
            result.append(current_branch)
        
        return result
    
    def compute_similarity_with_embeddings(self, emb1: List[float], emb2: List[float]) -> float:
        return self.compute_cosine_similarity(emb1, emb2)
    
    def get_answer(self, question: str) -> str:
        logger.info("поиск релевантных документов...")
        
        try:
            start_time = datetime.now()
            
            logger.info("получение эмбеддинга для вопроса...")
            try:
                question_embedding = self.openai_ef([question])[0]
                logger.info("эмбеддинг получен")
            except Exception as e:
                logger.error(f"ошибка получения эмбеддинга: {str(e)}")
                return Constants.API_ERROR_MESSAGE
            
            logger.info("запрос к chromadb...")
            try:
                results = self.collection.query(
                    query_embeddings=[question_embedding],
                    n_results=5,
                    include=['documents', 'metadatas', 'embeddings']
                )
            except Exception as e:
                logger.error(f"ошибка запроса к chromadb: {str(e)}")
                return Constants.API_ERROR_MESSAGE
            
            query_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"запрос к chromadb выполнен за {query_time:.2f} секунд")
            
            if not results.get('documents') or not results['documents'][0]:
                logger.info("не найдено релевантных документов")
                return Constants.NO_DOCUMENTS_MESSAGE
            
            logger.info(f"найдено {len(results['documents'][0])} начальных документов")
            logger.info("======= документы =======")
            for i, doc_id in enumerate(results['ids'][0]):
                title = results['metadatas'][0][i].get('title', 'Без заголовка')
                doc_embedding = results['embeddings'][0][i]
                score = self.compute_similarity_with_embeddings(question_embedding, doc_embedding)
                logger.info(f"{i+1}. ID: {doc_id}, Title: {title}, Relevance: {score:.3f}")
            logger.info("=========================")
            
            branches = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i][:self.doc_chunk_size],
                    'metadata': results['metadatas'][0][i]
                }
                branch = self.build_branch(doc, question_embedding)
                if branch.size > 1:
                    branches.append(branch)
                    logger.info(f"построена ветка из {branch.size} документов для {doc['id']}")
            
            merged_branches = self.merge_overlapping_branches(branches)
            logger.info(f"после объединения: {len(merged_branches)} веток")
            
            merged_branches.sort(key=lambda x: x.branch_score, reverse=True)
            
            all_docs = []
            seen_ids = set()
            docs_needed = self.max_docs
            
            for branch in merged_branches:
                branch_docs = sorted(
                    [(doc_id, doc, branch.scores[doc_id]) 
                     for doc_id, doc in branch.docs.items()],
                    key=lambda x: x[2],
                    reverse=True
                )
                
                for doc_id, doc, score in branch_docs:
                    if doc_id not in seen_ids and len(all_docs) < docs_needed:
                        all_docs.append(doc)
                        seen_ids.add(doc_id)
                        logger.info(f"добавлен документ {doc_id} со скором {score:.3f}")
            
            logger.info(f"итоговое количество документов: {len(all_docs)}")
            
            if not all_docs:
                return Constants.NO_DOCUMENTS_MESSAGE
            
            context = self.format_context(all_docs)
            
            try:
                user_content = self.user_prompt_template.format(
                    context=context,
                    question=self.escape_markdown(question)
                )
                
                logger.info("отправка запроса к GPT...")
                start_time = datetime.now()
                response = self.openai_client.chat.completions.create(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=Constants.GPT_TEMPERATURE,
                    max_tokens=Constants.MAX_RESPONSE_TOKENS,
                    timeout=Constants.API_TIMEOUT_SECONDS
                )
                
                api_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"ответ от GPT получен за {api_time:.2f} секунд")
                answer = response.choices[0].message.content
                return answer
                
            except Exception as e:
                logger.error(f"ошибка при запросе к GPT: {str(e)}")
                return Constants.API_ERROR_MESSAGE
            
        except Exception as e:
            logger.error(f"общая ошибка: {str(e)}")
            return Constants.TECHNICAL_ERROR_MESSAGE 