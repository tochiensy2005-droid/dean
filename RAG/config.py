import os
from dotenv import load_dotenv

load_dotenv()

# ==================== PDF PATHS ====================
PDF_FILE_1 = r"E:\Sỹ\archive (1)\Đề án.pdf"  # 536 trang
PDF_FILE_2 = r"E:\Sỹ\archive (1)\vietnam_tourism_data.pdf"  # 2807 trang

# ==================== CHUNK SETTINGS ====================
CHUNK_SIZE = 1500  # ký tự
CHUNK_OVERLAP = 150  # 10% của 1500
SEMANTIC_CHUNKING = True

# ==================== EMBEDDING SETTINGS ====================
# Using publicly available model from sentence-transformers
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768

# ==================== VECTOR DB SETTINGS ====================
# Store index in a simple ASCII path to avoid encoding issues
FAISS_INDEX_PATH = r"e:\faiss_index\index.faiss"
FAISS_METADATA_PATH = r"e:\faiss_metadata.pkl"

# ==================== SEMANTIC SEARCH SETTINGS ====================
TOP_K = 5  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.45  # Balanced threshold to filter out irrelevant results
USE_SIMILARITY_THRESHOLD = False  # Boolean bật/tắt

# ==================== GEMINI SETTINGS ====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "models/gemini-2.5-flash"  # updated to available model (<list_models>)
GEMINI_TEMPERATURE = 0.3
GEMINI_MAX_TOKENS = 2048

# ==================== RAG SETTINGS ====================
CHAIN_TYPE = "stuff"  # Q&A chain type
VERBOSE = True
