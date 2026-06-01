import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Yahan humne .csv kar diya hai
DATA_PATH = "pak_stay.csv"
INDEX_NAME = "pak-stay-index"
BATCH_SIZE = 100 

def load_and_preprocess_excel(file_path):
    print(f"Loading data from {file_path}...")
    # Yahan humne read_csv kar diya hai
    df = pd.read_csv(file_path)
    
    df = df.dropna(subset=['Hotel_Name', 'City']) 
    
    chunks = []
    metadata_list = []
    
    for index, row in df.iterrows():
        text_chunk = (
            f"{row.get('Hotel_Name', 'This hotel')} is located in {row.get('City', 'Unknown City')}, "
            f"{row.get('Province', '')}. It has an average customer rating of {row.get('Customer_Rating', 'N/A')} "
            f"out of 5. The average room rate is Rs. {row.get('Room_Rate', 'N/A')}. "
            f"Available amenities include: {row.get('Amenities', 'Not specified')}. "
            f"It is highly popular among {row.get('Customer_Type', 'various')} travelers."
        )
        
        metadata = {
            "hotel_name": str(row.get('Hotel_Name', '')),
            "city": str(row.get('City', '')),
            "room_rate": float(row.get('Room_Rate', 0)) if pd.notnull(row.get('Room_Rate')) else 0.0,
            "text": text_chunk 
        }
        
        chunks.append(text_chunk)
        metadata_list.append(metadata)
        
    print(f"Successfully processed {len(chunks)} hotel records into text chunks.")
    return chunks, metadata_list

def initialize_pinecone():
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    dimension = 384 
    
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" 
            )
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
        
    return pc.Index(INDEX_NAME)

def process_and_upload_to_pinecone(chunks, metadata_list, pinecone_index):
    print("Loading open-source embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    total_chunks = len(chunks)
    print(f"Generating embeddings and uploading in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_metadata = metadata_list[i : i + BATCH_SIZE]
        
        embeddings = model.encode(batch_chunks).tolist()
        
        ids = [f"hotel_{j}" for j in range(i, i + len(batch_chunks))]
        records = zip(ids, embeddings, batch_metadata)
        
        pinecone_index.upsert(vectors=records)
        print(f"Uploaded batch {i // BATCH_SIZE + 1} (Records {i} to {i + len(batch_chunks)})...")

    print("Data successfully uploaded to Pinecone!")

def main():
    chunks, metadata = load_and_preprocess_excel(DATA_PATH)
    index = initialize_pinecone()
    process_and_upload_to_pinecone(chunks, metadata, index)

if __name__ == "__main__":
    main()