from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info("🧠 KHỞI TẠO EMBEDDING SERVICE")
        logger.info(f"   Model: {EMBEDDING_MODEL}")
        
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"   Dimension: {EMBEDDING_DIMENSION}")
            logger.info("✅ Load model thành công\n")
        except Exception as e:
            logger.error(f"❌ Lỗi load model: {str(e)}")
            raise
    
    def embed_documents(self, texts: list) -> np.ndarray:
        """Embedding batch documents"""
        logger.info(f"📊 Embedding {len(texts)} chunks...")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            logger.info(f"✅ Embedding thành công: shape {embeddings.shape}\n")
            return embeddings
        
        except Exception as e:
            logger.error(f"❌ Lỗi embedding: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embedding một query"""
        try:
            embedding = self.model.encode(query, convert_to_numpy=True,normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"❌ Lỗi embedding query: {str(e)}")
            raise
