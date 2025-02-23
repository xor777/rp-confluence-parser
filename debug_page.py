import os
import requests
from requests.auth import HTTPBasicAuth
import urllib3
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_page_content(page_id):
    confluence_url = 'https://confluence-rp.teamslc.net'
    
    session = requests.Session()
    session.auth = HTTPBasicAuth(
        os.getenv('BASIC_AUTH_USERNAME'),
        os.getenv('BASIC_AUTH_PASSWORD')
    )
    session.verify = False
    
    login_response = session.get(f"{confluence_url}/login.action")
    print(f"Login page status: {login_response.status_code}")
    
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
    print(f"Login status: {login_result.status_code}")
    
    content_url = f"{confluence_url}/rest/api/content/{page_id}"
    print(f"\nFetching content from: {content_url}")
    
    response = session.get(
        content_url,
        headers={'Accept': 'application/json'},
        params={
            'expand': 'body.storage,version,space'
        }
    )
    
    print(f"Content API status: {response.status_code}")
    
    if response.status_code == 200:
        content = response.json()
        
        print("\nAPI Response Structure:")
        print("-" * 50)
        print("Content keys:", content.keys())
        print("Body keys:", content.get('body', {}).keys())
        print("Storage keys:", content.get('body', {}).get('storage', {}).keys())
        
        html_content = content.get('body', {}).get('storage', {}).get('value', '')
        
        print("\nHTML Content from API:")
        print("-" * 50)
        print(html_content)
        print("-" * 50)
        
        direct_url = f"{confluence_url}/pages/viewpage.action?pageId={page_id}"
        print(f"\nFetching page directly from: {direct_url}")
        
        direct_response = session.get(direct_url)
        print(f"Direct page status: {direct_response.status_code}")
        
        if direct_response.status_code == 200:
            soup = BeautifulSoup(direct_response.text, 'html.parser')
            main_content = soup.find('div', class_='wiki-content')
            
            if main_content:
                print("\nDirect page content structure:")
                print("-" * 50)
                print("Lists found:", len(main_content.find_all(['ul', 'ol'])))
                print("Links found:", len(main_content.find_all('a')))
                print("Paragraphs found:", len(main_content.find_all('p')))
                
                print("\nLinks in direct content:")
                for link in main_content.find_all('a'):
                    print(f"Link text: {link.get_text(strip=True)}, href: {link.get('href', '')}")
                
                print("\nDirect content HTML:")
                print("-" * 50)
                print(main_content.prettify())
                
            else:
                print("Could not find wiki-content div in direct page")
        
        else:
            print("Failed to fetch direct page")
    
    else:
        print("Failed to fetch content through API")

if __name__ == "__main__":
    get_page_content("236635571")