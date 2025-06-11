from qdrant_client import QdrantClient # For storing Chunks to Qdrant Server
from docling.chunking import HybridChunker # Used for Large data Chunking
from docling.datamodel.base_models import InputFormat # Used for proper formatting
from docling.document_converter import DocumentConverter # To validate Documents
from config import QDRANT_API_URL, QDRANT_API_KEY

import os
import streamlit as st
# from fastapi import FastAPI,Response

# Initialization Process
qdrant_client = QdrantClient(url=QDRANT_API_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
qdrant_client.set_model("sentence-transformers/all-MiniLM-L6-v2") # Set Embedding Model
qdrant_client.set_sparse_model("Qdrant/bm25") # Qdrant Model to store Data Chunks


converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX, InputFormat.PPTX]
)

def upload_document(filepath):
    try:
        filename = os.path.basename(filepath)  # Gets just 'bitcoin.pdf'
        doc_name = filename.split('.')[0].lower()  # Gets 'bitcoin' (without extension)
        collection_name = "RAG_" + doc_name  # Final collection name like 'RAG_bitcoin'

        result = converter.convert(filepath) # Converts the file
        docs, metadatas = [], [] # docs--> to store the data chunk of docs; metadatas--> To store info about the data chunks

        for i, chunk in enumerate(HybridChunker().chunk(result.document)): # Chunking Process
            docs.append(chunk.text) # Append data to Docs
            metadatas.append(chunk.meta.export_json_dict()) # Append Data of Data in JSON format
            st.markdown(f"✅ Chunked part {i + 1}")

        qdrant_client.add(
            collection_name=collection_name, #To mention the Folder_Name for Storing and Retrieving
            documents=docs, # To store data chunks
            metadata=metadatas, # To store info of chunks
            batch_size=64, # How many vectors will be uploaded in a single request
        )
        # For the Response-Content Type->res_fapi.headers["Content-Type"] = "application/json"

        st.success(f"\n✅ Upload complete! {len(docs)} chunks added to collection '{collection_name}'")
        return collection_name
    except Exception as e:
        st.text(f"❌ Error during upload: {e}")
        return None
