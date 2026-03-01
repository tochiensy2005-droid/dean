from pypdf import PdfReader
from langchain.schema import Document
from config import PDF_FILE_1, PDF_FILE_2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pdf_as_text(pdf_path: str) -> list:
    """
    Load PDF và trích xuất text (bỏ qua hình ảnh)
    """
    try:
        logger.info(f"📖 Đang load PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"   Tổng trang: {total_pages}")
        
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text.strip():  # Chỉ lấy trang có nội dung
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.split("\\")[-1],
                            "page": page_num + 1,
                            "total_pages": total_pages
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"   ⚠️ Lỗi trang {page_num + 1}: {str(e)}")
                continue
        
        logger.info(f"✅ Load thành công: {len(documents)} trang")
        return documents
    
    except Exception as e:
        logger.error(f"❌ Lỗi load PDF: {str(e)}")
        return []

def load_all_pdfs() -> list:
    """Load cả 2 file PDF"""
    logger.info("=" * 60)
    logger.info("🚀 ĐANG LOAD DỮ LIỆU TỪ PDF")
    logger.info("=" * 60)
    
    all_docs = []
    
    # Load file 1
    docs_1 = load_pdf_as_text(PDF_FILE_1)
    all_docs.extend(docs_1)
    
    # Load file 2
    docs_2 = load_pdf_as_text(PDF_FILE_2)
    all_docs.extend(docs_2)
    
    logger.info(f"\n📊 Tổng cộng: {len(all_docs)} trang text")
    logger.info("=" * 60 + "\n")
    
    return all_docs
