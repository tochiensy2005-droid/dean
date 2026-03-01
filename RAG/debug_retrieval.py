import logging
from embedding_service import EmbeddingService
from vector_store import FAISSVectorStore
from config import TOP_K
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load services
logger.info("🔧 Khởi tạo embedding service...")
embedding_service = EmbeddingService()

logger.info("\n📂 Load vector store...")
vector_store = FAISSVectorStore()
vector_store.load()

# Test queries
queries = [
    "Phú Quốc du lịch",
    "Hà Nội tham quan",
    "Du lịch Việt Nam",
    "Các điểm đến nổi tiếng"
]

for query in queries:
    logger.info(f"\n{'='*70}")
    logger.info(f"🔍 Query: '{query}'")
    logger.info(f"{'='*70}\n")
    
    query_embedding = embedding_service.embed_query(query)
    
    # Search (lấy top 10 để debug)
    import faiss
    distances, indices = vector_store.index.search(
        query_embedding.astype('float32').reshape(1, -1), 
        k=10
    )
    
    logger.info(f"Top 10 Results:\n")
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        similarity = 1 / (1 + distance)
        content = vector_store.metadata[idx]["content"][:150]
        source = vector_store.metadata[idx]["metadata"]["source"]
        page = vector_store.metadata[idx]["metadata"]["page"]
        
        threshold_status = "✅ PASS" if similarity >= 0.6 else "❌ FILTERED"
        
        logger.info(f"[{i+1}] {threshold_status} | Similarity: {similarity:.4f}")
        logger.info(f"     Source: {source} - Page {page}")
        logger.info(f"     Content: {content}...\n")

logger.info("\n" + "="*70)
logger.info("📊 TÓML TẮT:")
logger.info("="*70)
logger.info("• Nếu Similarity < 0.6 và được đánh ❌ FILTERED")
logger.info("  → Cần GIẢM THRESHOLD từ 0.6 xuống (ví dụ 0.3-0.4)")
logger.info("• Nếu không tìm thấy relevant results ở vị trí cao")
logger.info("  → Cần TĂNG TOP_K hoặc TỐI ƯU CHUNKING")
logger.info("="*70)
