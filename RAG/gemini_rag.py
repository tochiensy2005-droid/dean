import re
import google.generativeai as genai
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
    TOP_K
)
import logging

logger = logging.getLogger(__name__)

class GeminiRAG:
    def __init__(self, vector_store, embedding_service):
        logger.info("🚀 KHỞI TẠO GEMINI RAG")
        logger.info(f"   Model: {GEMINI_MODEL}")
        
        if not GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY không được set!")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        logger.info("✅ Khởi tạo thành công\n")
    
    def _build_context(self, retrieved_docs: list) -> str:
        """Xây dựng context từ retrieved documents"""
        context = "=== THÔNG TIN TỪ TÀI LIỆU ===\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"[{i}] ({doc['metadata']['source']} - Trang {doc['metadata']['page']})\n"
            context += f"Độ tương tự: {doc['similarity']}\n"
            # include full content so model can see all relevant text
            context += f"Nội dung: {doc['content']}\n\n"
        
        return context
    
    def _build_prompt(self, context: str, query: str) -> str:
        """Xây dựng prompt cho Gemini"""
        prompt = f"""ROLE: Bạn là chuyên gia du lịch Việt Nam, trải dài trả lời dựa TOÀN BỘ trên dữ liệu được cung cấp.

DỮ LIỆU THAM KHẢO:
{context}

Câu hỏi: {query}

YÊU CẦU:
- PHẢI trả lời dựa trên dữ liệu trên. KHÔNG được từ chối vì "không tìm thấy" nếu dữ liệu có sẵn.
- Nếu dữ liệu đề cập, hãy tóm tắt chi tiết và trích dẫn nguồn.
- Nếu thật sự KHÔNG có thông tin liên quan, mới nói rõ.
- Trả lời bằng tiếng Việt, ngắn gọn.
"""
        return prompt
    
    def query(self, question: str) -> dict:
        """
        Full Q&A RAG chain
        """
        logger.info("\n" + "="*60)
        logger.info("❓ CÂU HỎI: " + question)
        logger.info("="*60)
        
        try:
            # 1. Embedding query
            logger.info("🔍 Embedding query...")
            query_embedding = self.embedding_service.embed_query(question)
            
            # 2. Semantic search
            logger.info(f"🔎 Semantic search (Top-{TOP_K})...")
            retrieved_docs = self.vector_store.search(query_embedding)
            
            if not retrieved_docs:
                logger.warning("⚠️ Không tìm thấy tài liệu phù hợp!")
                return {
                    "status": "no_results",
                    "question": question,
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn.",
                    "retrieved_docs": []
                }
            
            logger.info(f"✅ Tìm thấy {len(retrieved_docs)} documents:")
            for doc in retrieved_docs:
                logger.info(f"   - {doc['metadata']['source']} (Trang {doc['metadata']['page']}) - Sim: {doc['similarity']}")
            
            # 3. Xây dựng context
            context = self._build_context(retrieved_docs)
            
            # 4. Xây dựng prompt
            prompt = self._build_prompt(context, question)
            logger.info(f"📝 Prompt length: {len(prompt)} chars")
            
            # 5. Gọi Gemini
            logger.info("🤖 Gọi Gemini để sinh câu trả lời...")
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
            )
            
            answer = response.text
            # Loại bỏ thông tin trích dẫn dạng (file.pdf - Trang X) trong câu trả lời
            # Loại bỏ (file - Trang X) hoặc (file.pdf - Trang X)
            answer = re.sub(r'\s*\([^)]*-\s*Trang\s*\d+\)', '', answer)
            answer = re.sub(r'  +', ' ', answer).strip()  # Chuẩn hóa khoảng trắng thừa
            logger.info("✅ Sinh câu trả lời thành công\n")
            
            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "num_results": len(retrieved_docs)
            }
        
        except Exception as e:
            logger.error(f"❌ Lỗi trong Q&A chain: {str(e)}")
            return {
                "status": "error",
                "question": question,
                "error": str(e)
            }
    
    def interactive_chat(self):
        """Interactive chatbot mode"""
        logger.info("\n" + "="*60)
        logger.info("💬 CHẾ ĐỘ TƯƠNG TÁC")
        logger.info("(Nhập 'exit' để thoát)")
        logger.info("="*60 + "\n")
        
        while True:
            try:
                question = input("👤 Bạn: ").strip()
                
                if question.lower() == 'exit':
                    logger.info("👋 Tạm biệt!")
                    break
                
                if not question:
                    continue
                
                result = self.query(question)
                
                print("\n" + "="*60)
                print("🤖 Chatbot:")
                print("="*60)
                print(result["answer"])
                print("\n" + "-"*60)
                print(f"📊 Số tài liệu tìm được: {result.get('num_results', 0)}")
                
                if result.get("retrieved_docs"):
                    print("\n📚 Nguồn tham khảo:")
                    for doc in result["retrieved_docs"]:
                        print(f"  • {doc['metadata']['source']} - Trang {doc['metadata']['page']} (Tương tự: {doc['similarity']})")
                
                print("="*60 + "\n")
            
            except KeyboardInterrupt:
                logger.info("\n👋 Tạm biệt!")
                break
            except Exception as e:
                logger.error(f"❌ Lỗi: {str(e)}\n")
