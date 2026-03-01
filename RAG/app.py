"""
RAG CHATBOT - Ứng dụng interactive
Dùng Gemini + FAISS + Semantic Search
"""

import logging
import sys
from pathlib import Path
from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from embedding_service import EmbeddingService
from vector_store import FAISSVectorStore
from gemini_rag import GeminiRAG

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_vector_store():
    """Kiểm tra xem vector store đã được training chưa"""
    if not Path(FAISS_INDEX_PATH).exists() or not Path(FAISS_METADATA_PATH).exists():
        logger.error("\n❌ FAISS vector store chưa được tạo!")
        logger.info("\n💡 Hãy chạy lệnh sau trước:")
        logger.info("   python train_rag.py")
        sys.exit(1)

def main():
    logger.info("\n" + "="*60)
    logger.info("💬 RAG CHATBOT - DU LỊCH VIỆT NAM")
    logger.info("="*60)
    
    try:
        # Kiểm tra vector store
        check_vector_store()
        
        # Load services
        logger.info("\n🔄 Khởi tạo hệ thống...")
        
        logger.info("  1️⃣  Load embedding service...")
        embedding_service = EmbeddingService()
        
        logger.info("  2️⃣  Load vector store...")
        vector_store = FAISSVectorStore()
        vector_store.load()
        
        logger.info("  3️⃣  Khởi tạo RAG chain...")
        rag = GeminiRAG(vector_store, embedding_service)
        
        logger.info("\n✅ Hệ thống sẵn sàng!\n")
        
        # Interactive chat
        rag.interactive_chat()
    
    except Exception as e:
        logger.error(f"❌ Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
