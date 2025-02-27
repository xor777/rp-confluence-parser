Инструмент для создания базы знаний из Confluence с использованием графовой структуры и RAG 

## Принцип работы

Парсер обходит все пространства и страницы Confluence, сохраняя не только контент, но и связи между страницами. Это позволяет строить (квази-)граф знаний, где:

- Узлы - отдельные страницы Confluence
- Ребра - связи между страницами (родитель-потомок, прямые ссылки)

## Структура данных

Каждая страница сохраняется в ChromaDB со следующими данными:

```python
{
    'id': 'page_id',
    'title': 'Заголовок страницы',
    'content': 'Очищенный текст страницы',
    'metadata': {
        'space_key': 'Ключ пространства',
        'space_name': 'Название пространства',
        'parent_id': 'ID родительской страницы',
        'child_ids': ['ID дочерних страниц'],
        'linked_ids': ['ID страниц, на которые есть ссылки']
    }
}
```

### Особенности реализации

1. **Graph RAG**: При поиске ответа на вопрос система:
   - Находит наиболее релевантные страницы через векторный поиск
   - Расширяет контекст, используя связанные страницы (родители, потомки, ссылки)
   - Оценивает релевантность связанных страниц
   - Формирует оптимальный набор документов для ответа

2. **Хранение данных**:
   - Используется ChromaDB с Sqlite3 в качестве бэкенда
   - Векторные эмбеддинги создаются с помощью OpenAI (text-embedding-3-large)
   - Метаданные и связи хранятся вместе с документами
