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
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pinecone_index = pc.Index("pak-stay-index")
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# --- GEMINI EMBEDDING FUNCTION (Stable & Fast) ---
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []

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

    search_query = f"Looking for a hotel in {city} suitable for {travel_type}."
    
    # Embedding using Gemini (Stable)
    query_vector = get_embedding(search_query)

    if not query_vector:
        return jsonify({"error": "Failed to generate embeddings"}), 500

    results = pinecone_index.query(
        vector=query_vector,
        top_k=8, 
        include_metadata=True
    )

    hotel_context = ""
    for match in results['matches']:
        m = match['metadata']
        hotel_context += f"- {m.get('name')} | Rating: {m.get('rating')} | Price: {m.get('price')} PKR | Amenities: {m.get('amenities')}\n"

    if not hotel_context:
        return jsonify([{"name": "No Database Match", "comment": f"No data for {city}."}])

    prompt = f"""You are an AI travel agent. Database Context: {hotel_context}.
    User wants: {min_stars}-star hotel in {city} for {travel_type} around Rs.{budget}.
    Output STRICTLY in JSON format: {{"hotels": [{"name": "...", "rating": 0, "price": 0, "amenities": [], "comment": "...", "top": true}]}}"""

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