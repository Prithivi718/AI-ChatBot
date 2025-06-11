from qdrant_client import QdrantClient
from config import QDRANT_API_URL, QDRANT_API_KEY

import streamlit as st
#from fastapi import FastAPI, Response

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_API_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# Initiating FastAPI, Response
# check_fapi= FastAPI()
# res_check= Response()

def check_existing_collection(filepath):
    import os
    filename = os.path.basename(filepath)  # e.g., "physics12.pdf"
    doc_name = filename.split('.')[0].lower()  # "physics12"
    collection_name = "RAG_" + doc_name  # Final name: RAG_physics12

    try:
        # For the Response-Content Type
        #res_check.headers["Content-Type"] = "application/json"

        collections = qdrant_client.get_collections().collections # To retrieve all the collections in the Qdrant Server
        existing = [c.name for c in collections] # Returns the matching collection

        if collection_name in existing: # Checks for Existing Collection
            st.success(f"📁 Collection '{collection_name}' already exists in Qdrant.")
            return collection_name
        else: # Returns back to main.py for Uploading a New Data Chunk.
            st.text("📁 No existing collection found. Upload required.")
            return None
    except Exception as e:
        print(f"❌ Error checking collections: {e}")
        return None