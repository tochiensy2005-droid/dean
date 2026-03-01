from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

def semantic_chunk(documents: list) -> list:
    """
    Tách documents thành chunks theo ngữ nghĩa (semantic).
    Sử dụng RecursiveCharacterTextSplitter với separators phù hợp cho Tiếng Việt.
    """
    logger.info("🔪 BẮT ĐẦU SEMANTIC CHUNKING")
    logger.info(f"   Chunk size: {CHUNK_SIZE} ký tự")
    logger.info(f"   Chunk overlap: {CHUNK_OVERLAP} ký tự ({int(CHUNK_OVERLAP/CHUNK_SIZE*100)}%)")
    
    # Separators theo thứ tự ưu tiên (semantic coherence)
    separators = [
        "\n\n",      # Ngắt đoạn văn (mạnh nhất)
        "\n",        # Ngắt dòng
        "。",        # Dấu chấm Trung Quốc (nếu có)
        "！",        # Dấu chấm than
        "？",        # Dấu chấm hỏi
        ".",         # Dấu chấm English
        " ",         # Khoảng trắng
        ""           # Fallback: chia từ
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len
    )
    
    chunks = []
    for i, doc in enumerate(documents):
        try:
            split_docs = text_splitter.split_documents([doc])
            chunks.extend(split_docs)
            
            if (i + 1) % 50 == 0:
                logger.info(f"   Đã xử lý: {i + 1}/{len(documents)} trang")
        
        except Exception as e:
            logger.warning(f"   ⚠️ Lỗi trang {doc.metadata.get('page')}: {str(e)}")
            continue
    
    logger.info(f"✅ Tạo thành công: {len(chunks)} chunks")
    logger.info("=" * 60 + "\n")
    
    return chunks
