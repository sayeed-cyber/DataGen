import time
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
import PyPDF2
import docx
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from textblob import TextBlob
import cohere
import nltk
from requests.exceptions import Timeout
import csv
from dotenv import load_dotenv
import threading
from uuid import uuid4

load_dotenv()

# --- NLTK SETUP (Self-Healing) ---
local_nltk_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(local_nltk_path):
    os.makedirs(local_nltk_path)
nltk.data.path.insert(0, local_nltk_path)

try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, Exception):
    print("Downloading NLTK 'punkt' data to local folder...")
    try:
        nltk.download('punkt', download_dir=local_nltk_path)
        nltk.download('punkt_tab', download_dir=local_nltk_path) 
    except Exception as e:
        print(f"Warning during NLTK download: {e}")

# Initialize the Cohere client
cohere_api_key = os.getenv('API_KEY')
co = cohere.Client(cohere_api_key)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

app.secret_key = os.getenv('FLASK_SECRET_KEY', str(uuid4()))

# Per-session storage
user_data_store = {}

def get_user_id():
    if 'user_id' not in session:
        session['user_id'] = str(uuid4())
    return session['user_id']

def get_user_state():
    user_id = get_user_id()
    if user_id not in user_data_store:
        user_data_store[user_id] = {
            'Global_data': '',
            'process_log': [],
            'qa_pairs_json': '[]',
            'llm_progress': {"current": 0, "total": 0, "status": "idle"},
            'abort_event': threading.Event()
        }
    return user_data_store[user_id]

def log_function_call(func_name, status="Started", data=None):
    user_state = get_user_state()
    entry = {"function": func_name, "status": status}
    if data:
        entry["data"] = data
    user_state['process_log'].append(entry)

# --- File Extraction Functions ---
def extract_text_from_pdf(file):
    log_function_call("extract_text_from_pdf")
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
        log_function_call("extract_text_from_pdf", "Completed")
        return pdf_text
    except Exception as e:
        log_function_call("extract_text_from_pdf", f"Error: {e}")
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file):
    log_function_call("extract_text_from_docx")
    try:
        doc = docx.Document(file)
        doc_text = "\n".join([para.text for para in doc.paragraphs])
        log_function_call("extract_text_from_docx", "Completed")
        return doc_text
    except Exception as e:
        log_function_call("extract_text_from_docx", f"Error: {e}")
        return f"Error reading DOCX: {e}"

def extract_text_from_txt(file):
    log_function_call("extract_text_from_txt")
    try:
        text = file.read().decode('utf-8')
        log_function_call("extract_text_from_txt", "Completed")
        return text
    except Exception as e:
        log_function_call("extract_text_from_txt", f"Error: {e}")
        return f"Error reading TXT: {e}"

# --- Routes ---
@app.route('/')
def index():
    backend_url = request.host_url.rstrip('/')
    return render_template('index.html', backend_url=backend_url)

@app.route('/process')
def process():
    return render_template('process.html')

# --- COHERE API LOGIC ---
def generate_response(prompt, max_retries=3, backoff_factor=2):
    attempt = 0
    while attempt < max_retries:
        try:
            start_time = time.time()
            
            # Using 'command-nightly' (Large context, often available on trial)
            response = co.chat(
                model="command-nightly", 
                message=prompt,
                temperature=0.7,
            )
            
            elapsed_time = time.time() - start_time
            if elapsed_time > 60: 
                raise TimeoutError("Response took too long.")
            
            return response.text.strip()

        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if "not found" in error_msg or "removed" in error_msg:
                print("CRITICAL: Model name invalid. Stopping retries to save credits.")
                return None

            if "429" in error_msg: # Rate limit handling
                time.sleep(20) 
        
        attempt += 1
        time.sleep(backoff_factor ** attempt)

    print("Max retries reached.")
    return None

def send_chunk_to_LLM(chunk):
    prompt = (
        f"You are an AI assistant. Generate detailed question-answer pairs based on the text below. "
        "Respond ONLY in valid JSON format: "
        "[{'question': '...', 'answer': '...'}, ...]. "
        "Do not include Markdown formatting (```json). "
        f"\n\nText: {chunk}"
    )
    
    content = generate_response(prompt)
    if content is None:
        return []

    content = content.replace("```json", "").replace("```", "").strip()

    try:
        new_qa_pairs = json.loads(content)
        if isinstance(new_qa_pairs, dict):
            new_qa_pairs = [new_qa_pairs]
        return new_qa_pairs
    except json.JSONDecodeError:
        try:
            fixed_content = content.strip().rstrip(",}") + "}"
            new_qa_pairs = json.loads(fixed_content)
            if isinstance(new_qa_pairs, list):
                return new_qa_pairs
        except:
            pass
        return []

# --- CORE PIPELINE (OPTIMIZED FOR RAM & LARGE FILES) ---
def process_pipeline_after_extraction(user_id, Global_data, chunk_size):
    user_state = user_data_store.get(user_id)
    if not user_state: return

    user_state['abort_event'].clear()
    
    # 1. Create CSV file immediately
    csv_filename = f"qa_pairs_{user_id}.csv"
    csv_headers = ['question', 'answer']
    
    # Overwrite any existing file
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    try:
        user_state['process_log'].append({"function": "NLP_processing", "status": "Started"})
        blob = TextBlob(Global_data)
        sentences = blob.sentences
        
        chunks = [' '.join(str(sentence) for sentence in sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
        
        # Clear huge blob from memory
        del blob
        del sentences

        user_state['process_log'].append({"function": "NLP_processing", "status": "Completed"})
        user_state['process_log'].append({"function": "sending_chunk_to_LLM", "status": "Started"})
        
        user_state['llm_progress']["current"] = 0
        user_state['llm_progress']["total"] = len(chunks)
        user_state['llm_progress']["status"] = "running"
        
        # Buffer for frontend display (Max 20 items)
        latest_pairs_for_display = [] 

        for i, chunk in enumerate(chunks, start=1):
            if user_state['abort_event'].is_set():
                user_state['llm_progress']["status"] = "aborted"
                user_state['process_log'].append({"function": "sending_chunk_to_LLM", "status": "Aborted"})
                return

            print(f"Processing Chunk {i}/{len(chunks)}")
            
            pairs = send_chunk_to_LLM(chunk)
            
            if pairs:
                # 1. Write to CSV IMMEDIATELY (Disk)
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    for p in pairs:
                        q_safe = p.get('question', 'No Question')
                        a_safe = p.get('answer', 'No Answer')
                        writer.writerow({'question': q_safe, 'answer': a_safe})
                
                # 2. Update Frontend Buffer (RAM)
                latest_pairs_for_display.extend(pairs)
                # Keep only last 20 pairs to keep JSON small and fast
                if len(latest_pairs_for_display) > 20:
                    latest_pairs_for_display = latest_pairs_for_display[-20:]
                
                user_state['qa_pairs_json'] = json.dumps(latest_pairs_for_display, indent=2)
                
                # 3. Clean up
                del pairs
            else:
                 print(f"Chunk {i} produced no pairs.")

            user_state['llm_progress']["current"] = i
            time.sleep(1) 

        user_state['llm_progress']["status"] = "done"
        user_state['process_log'].append({"function": "sending_chunk_to_LLM", "status": "Completed"})

    except Exception as e:
        print(f"Pipeline Error: {e}")
        user_state['llm_progress']["status"] = "error"
        user_state['process_log'].append({"function": "pipeline_error", "status": f"Error: {e}"})

# --- FILE UPLOAD ---
@app.route('/upload_file', methods=['POST'])
def upload_file():
    user_state = get_user_state()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension == '.pdf':
        user_state['Global_data'] = extract_text_from_pdf(file)
    elif file_extension == '.docx':
        user_state['Global_data'] = extract_text_from_docx(file)
    elif file_extension == '.txt':
        user_state['Global_data'] = extract_text_from_txt(file)
    else:
        return jsonify({"error": "Unsupported file format"}), 400
    
    if user_state['Global_data'].startswith("Error"):
        return jsonify({"error": user_state['Global_data']}), 500
    
    log_function_call("text_extraction", "Completed")
    # Chunk size 60 for fewer API calls
    threading.Thread(target=process_pipeline_after_extraction, args=(get_user_id(), user_state['Global_data'], 60)).start()
    return jsonify({"stage": "extracted", "message": "Text extraction completed."}), 200

# --- STATUS ENDPOINTS ---
@app.route('/get_process_log', methods=['GET'])
def get_process_log():
    user_state = get_user_state()
    log_data = jsonify(user_state['process_log'])
    user_state['process_log'].clear()
    return log_data, 200

@app.route('/get_qa_pairs', methods=['GET'])
def get_qa_pairs():
    user_state = get_user_state()
    return jsonify({"qa_pairs": user_state['qa_pairs_json']}), 200

@app.route('/get_llm_progress', methods=['GET'])
def get_llm_progress():
    user_state = get_user_state()
    return jsonify(user_state['llm_progress']), 200

# --- WEBPAGE EXTRACTION ---
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme in ['http', 'https'])

def extract_data_from_webpage(url):
    log_function_call("extract_data_from_webpage")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True) if is_valid_url(urljoin(url, link['href']))]
        log_function_call("extract_data_from_webpage", "Completed")
        return text_content, links
    except Exception as e:
        log_function_call("extract_data_from_webpage", f"Error: {e}")
        return None, []

@app.route('/extract_webpage', methods=['POST'])
def extract_webpage():
    user_state = get_user_state()
    data = request.json
    url = data.get('url')
    if not url or not is_valid_url(url):
        return jsonify({"error": "Valid URL is required"}), 400
    
    user_state['Global_data'], links = extract_data_from_webpage(url)
    if not user_state['Global_data']:
        return jsonify({"error": "Unable to retrieve webpage content"}), 500
    
    log_function_call("text_extraction", "Completed")
    # Chunk size 60
    threading.Thread(target=process_pipeline_after_extraction, args=(get_user_id(), user_state['Global_data'], 60)).start()
    return jsonify({"stage": "extracted", "message": "Webpage extraction completed."}), 200

# --- CRAWLING ---
def crawl_website(start_url, max_pages=50):
    log_function_call("crawl_website")
    visited = set()
    to_visit = [start_url] 
    crawled_content = ""
    crawled_links = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited: continue

        text_content, links = extract_data_from_webpage(current_url)
        if text_content:
            crawled_content += text_content + "\n\n"

        for link in links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        visited.add(current_url)
        crawled_links.append(current_url)

    log_function_call("crawl_website", "Completed")
    return crawled_content, crawled_links

@app.route('/crawl_webpage', methods=['POST'])
def crawl_webpage():
    user_id = get_user_id()
    user_state = get_user_state()
    user_state['process_log'].append({"function": "crawl_webpage", "status": "Started"})
    
    try:
        data = request.json
        start_url = data.get('url')
        max_pages = int(data.get('max_pages', 20))
        
        if not start_url or not is_valid_url(start_url):
            return jsonify({"error": "Valid URL required"}), 400
            
        user_state['Global_data'], links = crawl_website(start_url, max_pages)
        # Chunk size 60
        threading.Thread(target=process_pipeline_after_extraction, args=(user_id, user_state['Global_data'], 60)).start()
        return jsonify({"stage": "extracted", "message": "Crawling completed."}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# --- DOWNLOAD CSV ---
@app.route('/download_csv', methods=['GET'])
def download_csv():
    user_id = get_user_id()
    csv_filename = f"qa_pairs_{user_id}.csv"
    
    if os.path.exists(csv_filename):
        return send_file(csv_filename, as_attachment=True)
    else:
        return jsonify({"error": "CSV file not found (Processing might still be starting)"}), 404

@app.route('/abort_process', methods=['POST'])
def abort_process():
    user_state = get_user_state()
    user_state['abort_event'].set()
    user_state['llm_progress'] = {"current": 0, "total": 0, "status": "idle"}
    return jsonify({"message": "Process aborted."}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)