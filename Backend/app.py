import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv

# --- PATHS SETTING ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'Frontend')

load_dotenv()
app = Flask(__name__, template_folder=FRONTEND_DIR)
CORS(app)

# --- API KEYS & SETUP ---
# Safety check: agar key na mile to error print kare
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found!")

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("pak-stay-index")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- GEMINI EMBEDDING FUNCTION ---
def get_embedding(text):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Sahi model call
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini API Error details: {e}")
        return None # None return karein taake hum handle kar sakein

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_hotels():
    data = request.json
    city = data.get('city', '')
    travel_type = data.get('travel_type', '')
    budget = data.get('budget', '20000')
    min_stars = data.get('min_stars', '1')

    # Embedding logic
    query_vector = get_embedding(f"Hotel in {city} for {travel_type}")
    
    if query_vector is None:
        return jsonify({"error": "Embedding generation failed"}), 500

    results = pinecone_index.query(vector=query_vector, top_k=5, include_metadata=True)

    hotel_context = "\n".join([f"- {m['metadata'].get('name')} | Rating: {m['metadata'].get('rating')} | Price: {m['metadata'].get('price')} PKR" for m in results['matches']])

    if not hotel_context:
        return jsonify([{"name": "No Results", "comment": "No data found."}])

    prompt = f"Travel agent context: {hotel_context}. User wants {min_stars}-star hotel in {city} for {travel_type} budget {budget}. JSON format: {{\"hotels\": [...]}}"

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        return jsonify(json.loads(response.choices[0].message.content).get("hotels", []))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()