import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

# --- PATHS SETTING ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'Frontend')

load_dotenv()
app = Flask(__name__, template_folder=FRONTEND_DIR)
CORS(app)

# --- API KEYS SETUP ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') # Naya addition

print("Connecting to databases and AI... Please wait.")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("pak-stay-index")
groq_client = Groq(api_key=GROQ_API_KEY)

# Gemini configure karein
genai.configure(api_key=GEMINI_API_KEY)

# --- GEMINI EMBEDDING FUNCTION (Zero RAM Cost) ---
def get_embedding(text):
    try:
        # Using Gemini's embedding model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []

# --- HOME PAGE ROUTE ---
@app.route('/')
def home():
    return render_template('index.html')

# --- AI SEARCH ROUTE ---
@app.route('/api/search', methods=['POST'])
def search_hotels():
    data = request.json
    city = data.get('city', '')
    travel_type = data.get('travel_type', '')
    budget = data.get('budget', '20000')
    min_stars = data.get('min_stars', '1')

    # Semantic Query
    search_query = f"Looking for a hotel in {city} suitable for {travel_type}."
    
    # Generate vector using Gemini
    query_vector = get_embedding(search_query)

    if not query_vector:
        return jsonify({"error": "Failed to connect to Embedding API"}), 500

    # Pinecone se vector search
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
        return jsonify([{
            "name": "No Database Match",
            "rating": 0,
            "price": 0,
            "amenities": [],
            "comment": f"We currently don't have any data for {city}."
        }])

    # --- SMART AI PROMPT (Using Groq) ---
    prompt = f"""
    You are an intelligent AI travel agent for Pakistan. 
    User wants: A {min_stars}-star (or higher) hotel in {city} for {travel_type} with a target budget around Rs.{budget}.
    
    Database Context:
    {hotel_context}
    
    RULES:
    1. Try to find the best match based on the user's criteria.
    2. SMART FALLBACK (STARS): If a {min_stars}-star hotel is NOT available, find the closest higher rating.
    3. SMART FALLBACK (BUDGET): If hotels under Rs.{budget} are NOT available, give the closest available price.
    4. NEVER return empty if there is context. Pick the best 3 alternatives.
    5. In the "comment", briefly explain your choice.
    
    Output STRICTLY in this JSON format:
    {{
      "hotels": [
        {{"name": "Hotel X", "rating": 4.5, "price": 10000, "amenities": ["WiFi"], "comment": "Your reasoning here", "top": true}}
      ]
    }}
    """

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        llm_response = response.choices[0].message.content
        parsed = json.loads(llm_response)
        
        hotel_json = parsed.get("hotels", [])
        return jsonify(hotel_json)

    except Exception as e:
        print(f"LLM Error: {e}")
        return jsonify({"error": "Failed to generate AI response"}), 500

if __name__ == '__main__':
    app.run(debug=True)