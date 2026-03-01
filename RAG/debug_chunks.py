import logging
from pdf_loader import load_all_pdfs
from semantic_chunker import semantic_chunk

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("📚 Load toàn bộ PDF documents...")
docs = load_all_pdfs()

logger.info("✂️  Bắt đầu semantic chunking...")
chunks = semantic_chunk(docs)

logger.info(f"\n✅ Tạo {len(chunks)} chunks\n")

# Test keywords
keywords = ["Phú Quốc", "Hà Nội", "Du lịch", "Việt Nam", "Đà Lạt"]

for keyword in keywords:
    matching_chunks = [
        (i, chunk.page_content[:250], chunk.metadata) 
        for i, chunk in enumerate(chunks) 
        if keyword.lower() in chunk.page_content.lower()
    ]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔎 Tìm chunks chứa từ khóa: '{keyword}'")
    logger.info(f"{'='*70}")
    logger.info(f"Tìm thấy: {len(matching_chunks)} chunks\n")
    
    for idx, content, metadata in matching_chunks[:3]:  # Hiển thị top 3
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 0)
        logger.info(f"[Chunk {idx}] Source: {source} - Page {page}")
        logger.info(f"Content: {content}...\n")
        logger.info("-" * 70 + "\n")

logger.info("\n" + "="*70)
logger.info("📊 PHÂN TÍCH:")
logger.info("="*70)
logger.info("• Nếu từ khóa KHÔNG được tìm thấy")
logger.info("  → PDF có thể không chứa thông tin đó")
logger.info("  → Hoặc dữ liệu ở định dạng khác (ảnh, bảng, v.v.)")
logger.info("• Nếu chunks có vẻ bị cắt giữa chừng")
logger.info("  → Cần điều chỉnh CHUNK_SIZE hoặc separators")
logger.info("="*70)
