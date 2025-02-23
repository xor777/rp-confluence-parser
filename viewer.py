from flask import Flask, render_template_string, request, jsonify
from markupsafe import Markup
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import json
import requests
from requests.auth import HTTPBasicAuth
import urllib3

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Confluence Pages Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .page-card {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .page-title {
            color: #0052CC;
            margin-top: 0;
        }
        .page-content {
            color: #333;
            margin: 10px 0;
            line-height: 1.5;
            white-space: pre-line;
        }
        .page-content.collapsed {
            max-height: 200px;
            overflow: hidden;
            position: relative;
        }
        .page-content.collapsed::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 50px;
            background: linear-gradient(transparent, white);
        }
        .expand-btn {
            color: #0052CC;
            background: none;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            font-size: 0.9em;
            margin-top: 10px;
            border: 1px solid #0052CC;
            border-radius: 3px;
        }
        .expand-btn:hover {
            background: #0052CC;
            color: white;
        }
        .page-meta {
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }
        .meta-item {
            margin: 5px 0;
        }
        .meta-label {
            font-weight: bold;
            color: #555;
        }
        .pagination {
            margin: 20px 0;
            text-align: center;
        }
        .pagination a {
            display: inline-block;
            padding: 8px 16px;
            text-decoration: none;
            color: #0052CC;
            background: white;
            border: 1px solid #ddd;
            margin: 0 4px;
            border-radius: 3px;
        }
        .pagination a:hover {
            background: #0052CC;
            color: white;
        }
        .current-page {
            font-weight: bold;
            background: #0052CC !important;
            color: white !important;
        }
        .space-info {
            color: #666;
            margin-bottom: 5px;
            background: #f8f9fa;
            padding: 5px 10px;
            border-radius: 3px;
        }
        .page-link {
            color: #0052CC;
            text-decoration: none;
        }
        .page-link:hover {
            text-decoration: underline;
        }
        .metadata-section {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .metadata-item {
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }
        .debug-link {
            margin-left: 10px;
            color: #666;
            text-decoration: none;
            font-size: 0.9em;
            padding: 2px 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
        .debug-link:hover {
            background: #f0f0f0;
            text-decoration: none;
        }
    </style>
    <script>
        function toggleContent(id) {
            const content = document.getElementById('content-' + id);
            const btn = document.getElementById('btn-' + id);
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                btn.textContent = 'Свернуть';
            } else {
                content.classList.add('collapsed');
                btn.textContent = 'Показать полностью';
            }
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>View DB</h1>
        
        {% for item in items %}
        <div class="page-card">
            <div class="space-info">
                Space: {{ item.metadata.space_name }} ({{ item.metadata.space_key }})
            </div>
            <h2 class="page-title">
                <a href="{{ item.confluence_url }}" target="_blank" class="page-link">
                    {{ item.metadata.title }}
                </a>
            </h2>
            <div id="content-{{ item.id }}" class="page-content collapsed">
                {{ item.document | safe }}
            </div>
            <button class="expand-btn" id="btn-{{ item.id }}" onclick="toggleContent('{{ item.id }}')">
                Показать полностью
            </button>
            <div class="page-meta">
                <div class="metadata-section">
                    {% for key, value in item.metadata.items() %}
                    <div class="metadata-item">
                        <span class="meta-label">{{ key }}:</span>
                        <span class="meta-value">{{ value }}</span>
                    </div>
                    {% endfor %}
                    <div class="metadata-item">
                        <span class="meta-label">Document ID:</span>
                        <span class="meta-value">
                            {{ item.id }}
                            <a href="/debug/{{ item.id }}" target="_blank" class="debug-link">[Debug]</a>
                        </span>
                    </div>
                </div>
                
                {% if item.structure %}
                <div class="structure-section">
                    <h3>Page Structure</h3>
                    <div class="metadata-section">
                        {% for label, value in item.structure %}
                        <div class="metadata-item">
                            <span class="meta-label">{{ label }}:</span>
                            <span class="meta-value">{{ value }}</span>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
        
        <div class="pagination">
            {% if page > 1 %}
                <a href="?page={{ page - 1 }}&per_page={{ per_page }}">Previous</a>
            {% endif %}
            
            {% set start = page - 2 if page > 2 else 1 %}
            {% set end = page + 2 if page + 2 <= total_pages else total_pages %}
            
            {% for p in range(start, end + 1) %}
                <a href="?page={{ p }}&per_page={{ per_page }}" 
                   {% if p == page %}class="current-page"{% endif %}>
                    {{ p }}
                </a>
            {% endfor %}
            
            {% if page < total_pages %}
                <a href="?page={{ page + 1 }}&per_page={{ per_page }}">Next</a>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

def get_confluence_session():
    session = requests.Session()
    session.auth = HTTPBasicAuth(
        os.getenv('BASIC_AUTH_USERNAME'),
        os.getenv('BASIC_AUTH_PASSWORD')
    )
    session.verify = False
    
    confluence_url = 'https://confluence-rp.teamslc.net'
    login_data = {
        'os_username': os.getenv('CONFLUENCE_USERNAME'),
        'os_password': os.getenv('CONFLUENCE_PASSWORD'),
        'os_destination': '/index.action',
        'login': 'Log in'
    }
    
    session.post(
        f"{confluence_url}/dologin.action",
        data=login_data
    )
    
    return session

def get_chroma_client():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-large"
    )
    
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection(
        name="confluence_eh",
        embedding_function=openai_ef
    )

def format_confluence_url(page_id):
    return f"https://confluence-rp.teamslc.net/pages/viewpage.action?pageId={page_id}"

def pretty_print_document(doc):
    return {
        'document_decoded': doc,
        'document_pretty': {
            'text': doc,
            'length': len(doc),
            'lines': doc.split('\n'),
            'lines_count': doc.count('\n') + 1,
            'preview_decoded': doc[:200]
        }
    }

def format_page_structure(metadata):
    structure = []
    
    if metadata.get('parent_id'):
        structure.append(('Parent Page ID', metadata['parent_id']))
    
    if metadata.get('child_ids'):
        children = metadata['child_ids'].split(',')
        if children and children[0]:
            structure.append(('Child Pages', f"{len(children)} pages"))
            structure.append(('Child IDs', ', '.join(children)))
    
    if metadata.get('linked_ids'):
        links = metadata['linked_ids'].split(',')
        if links and links[0]:
            structure.append(('Linked Pages', f"{len(links)} pages"))
            structure.append(('Linked IDs', ', '.join(links)))
    
    return structure

def get_filtered_metadata(metadata):
    return {k: v for k, v in metadata.items() 
            if k not in ['parent_id', 'child_ids', 'linked_ids']}

@app.route('/debug/<doc_id>')
def debug_document(doc_id):
    collection = get_chroma_client()
    
    result = collection.get(
        ids=[doc_id],
        include=['documents', 'metadatas']
    )
    
    if not result['documents']:
        return jsonify({'error': 'Document not found'}), 404
    
    doc = result['documents'][0]
    response = {
        'id': doc_id,
        'document_info': pretty_print_document(doc),
        'metadata': result['metadatas'][0],
    }
    
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )

@app.route('/')
def index():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    collection = get_chroma_client()
    
    total_count = collection.count()
    total_pages = (total_count + per_page - 1) // per_page
    
    offset = (page - 1) * per_page
    
    result = collection.get(
        limit=per_page,
        offset=offset,
        include=['documents', 'metadatas']
    )
    
    session = get_confluence_session()
    
    items = []
    for i in range(len(result['ids'])):
        page_url = format_confluence_url(result['ids'][i])
        
        structure = format_page_structure(result['metadatas'][i])
        
        filtered_metadata = get_filtered_metadata(result['metadatas'][i])
        
        items.append({
            'id': result['ids'][i],
            'document': Markup(result['documents'][i].replace('\n', '<br>')),
            'metadata': filtered_metadata,
            'structure': structure,
            'confluence_url': page_url
        })
    
    confluence_url = 'https://confluence-rp.teamslc.net'
    
    return render_template_string(
        PAGE_TEMPLATE,
        items=items,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        confluence_url=confluence_url
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)