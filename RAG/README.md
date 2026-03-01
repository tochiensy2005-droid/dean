# 🌏 RAG CHATBOT - HỆ THỐNG TRỢ LÝ DU LỊCH VIỆT NAM

## 📚 Công Nghệ Sử Dụng

| Thành Phần | Công Nghệ |
|-----------|-----------|
| **PDF Processing** | PyPDF - Trích xuất text từ PDF |
| **Semantic Chunking** | LangChain (1500 ký tự, 10% overlap) |
| **Embeddings** | Google Sentence Transformers (multilingual) |
| **Vector DB** | FAISS (local, fast search) |
| **Semantic Search** | L2 distance → Cosine similarity |
| **Threshold Filter** | Boolean configurable |
| **LLM** | Google Gemini Pro |
| **Q&A Chain** | Full RAG with retrieval + generation |

## 🚀 Quick Start

### 1. Cài Đặt
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API Key
Tạo `.env`:
```
GEMINI_API_KEY=your_gemini_key_here
```

### 3. Training
```bash
python train_rag.py
```

### 4. Run
```bash
python app.py
```

---
**Made with ❤️ for Vietnam Tourism (Feb 2026)** 
