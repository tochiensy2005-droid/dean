"""
TRAIN RAG SYSTEM - Script huấn luyện hệ thống
Semantic chunking + Google Embeddings + FAISS + Gemini
"""

import logging
from pdf_loader import load_all_pdfs
from semantic_chunker import semantic_chunk
from embedding_service import EmbeddingService
from vector_store import FAISSVectorStore
from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from pathlib import Path

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("\n" + "🚀"*30)
    logger.info("KHỞI TẠO HỆ THỐNG RAG CHO DỮ LIỆU DU LỊCH VIỆT NAM")
    logger.info("🚀"*30 + "\n")
    
    try:
        # ========== BƯỚC 1: LOAD PDF ==========
        logger.info("📖 BƯỚC 1: LOAD DỮ LIỆU TỪ PDF")
        logger.info("-" * 60)
        documents = load_all_pdfs()
        
        # ========== BƯỚC 2: SEMANTIC CHUNKING ==========
        logger.info("📖 BƯỚC 2: SEMANTIC CHUNKING")
        logger.info("-" * 60)
        chunks = semantic_chunk(documents)
        
        # ========== BƯỚC 3: EMBEDDING ==========
        logger.info("📖 BƯỚC 3: EMBEDDING CHUNKS")
        logger.info("-" * 60)
        embedding_service = EmbeddingService()
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)
        
        # ========== BƯỚC 4: TẠO VECTOR STORE ==========
        logger.info("📖 BƯỚC 4: TẠO FAISS VECTOR STORE")
        logger.info("-" * 60)
        
        metadata = [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        
        vector_store = FAISSVectorStore()
        vector_store.create_index(embeddings, metadata)
        
        # ========== BƯỚC 5: LƯU INDEX ==========
        logger.info("📖 BƯỚC 5: LƯU INDEX")
        logger.info("-" * 60)
        vector_store.save()
        
        # ========== HOÀN TẤT ==========
        logger.info("\n" + "✅"*30)
        logger.info("HOÀN TẤT HUẤN LUYỆN HỆ THỐNG RAG!")
        logger.info("✅"*30)
        
        logger.info("\n📊 THỐNG KÊ:")
        logger.info(f"  • Tổng PDF pages: {len(documents)}")
        logger.info(f"  • Tổng chunks: {len(chunks)}")
        logger.info(f"  • Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  • Vector store size: {vector_store.index.ntotal}")
        logger.info(f"\n💾 Lưu tại:")
        logger.info(f"  • Index: {FAISS_INDEX_PATH}")
        logger.info(f"  • Metadata: {FAISS_METADATA_PATH}")
        
        return True
    
    except Exception as e:
        logger.error(f"\n❌ LỖI HUẤN LUYỆN: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
