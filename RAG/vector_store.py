import faiss
import pickle
import numpy as np
from pathlib import Path
from config import (
    FAISS_INDEX_PATH, 
    FAISS_METADATA_PATH,
    TOP_K,
    USE_SIMILARITY_THRESHOLD,
    SIMILARITY_THRESHOLD
)
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_dimension = None
    
    def create_index(self, embeddings: np.ndarray, metadata: list):
        """
        Tạo FAISS index từ embeddings
        """
        logger.info("🔨 TẠO FAISS INDEX")
        logger.info(f"   Embeddings shape: {embeddings.shape}")
        
        try:
            # Đảm bảo embeddings là float32
            embeddings = embeddings.astype(np.float32)
            self.embedding_dimension = embeddings.shape[1]
            
            # Tạo index (L2 distance - cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            self.index.add(embeddings)
            
            self.metadata = metadata
            
            logger.info(f"✅ Index tạo thành công: {self.index.ntotal} vectors\n")
        
        except Exception as e:
            logger.error(f"❌ Lỗi tạo index: {str(e)}")
            raise
    
    def save(self):
        """Lưu index và metadata"""
        logger.info("💾 LƯU FAISS INDEX")
        
        try:
            Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(self.index, FAISS_INDEX_PATH)
            with open(FAISS_METADATA_PATH, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"   Index: {FAISS_INDEX_PATH}")
            logger.info(f"   Metadata: {FAISS_METADATA_PATH}")
            logger.info("✅ Lưu thành công\n")
        
        except Exception as e:
            logger.error(f"❌ Lỗi lưu: {str(e)}")
            raise
    
    def load(self):
        """Load index từ disk"""
        logger.info("📂 LOAD FAISS INDEX")
        
        try:
            if not Path(FAISS_INDEX_PATH).exists():
                raise FileNotFoundError(f"Index không tồn tại: {FAISS_INDEX_PATH}")
            
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.embedding_dimension = self.index.d
            
            logger.info(f"   Vectors: {self.index.ntotal}")
            logger.info(f"   Dimension: {self.embedding_dimension}")
            logger.info("✅ Load thành công\n")
        
        except Exception as e:
            logger.error(f"❌ Lỗi load: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = TOP_K) -> list:
        """
        Semantic search với FAISS
        Trả về top K results với similarity score
        """
        try:
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # FAISS tính L2 distance, ta chuyển sang cosine similarity
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                similarity = float(distance)   
                
                # Bật/tắt threshold
                if USE_SIMILARITY_THRESHOLD and similarity < SIMILARITY_THRESHOLD:
                    continue
                
                results.append({
                    "rank": i + 1,
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"],
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4)
                })
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Lỗi search: {str(e)}")
            raise
