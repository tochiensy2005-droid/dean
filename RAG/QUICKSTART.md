# 🚀 QUICKSTART - RAG CHATBOT DU LỊCH VIỆT NAM

## 📋 Yêu Cầu
- Python 3.9+
- API key từ Google Gemini (miễn phí)

## ⚡ Cài Đặt Nhanh (5 phút)

### 1. Setup môi trường
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cấu hình API key
Tạo/Edit file `.env`:
```
GEMINI_API_KEY=AIzaSyB00zfzkFvuJZb2SyBxz1p92bmGP_BqtHY
```

Lấy API key tại: https://ai.google.dev/

### 3. Chuẩn Bị File PDF
Đặt 2 file PDF vào thư mục `e:\Sỹ\`:
- `file1.pdf` (536 trang) Đean
- `file2.pdf` (2807 trang) vietnam_tourism

### 4. Training (lần đầu ~5-10 phút)
```bash
python train_rag.py
```

Quá trình này sẽ:
- Load 2 file PDF (3343 trang)
- Semantic chunking (1500 ký tự/chunk, 10% overlap)
- Embedding bằng Google multilingual model
- Tạo FAISS index local
- Lưu vector store (~1-2 GB)

### 5. Chạy Chatbot
```bash
python app.py
```

Sau đó nhập câu hỏi:
```
👤 Bạn: Những điểm du lịch nổi tiếng nhất ở Việt Nam là gì?
🤖 Chatbot: [Trả lời dựa trên tài liệu]
```

## ⚙️ Cấu Hình Tuỳ Chỉnh

Edit `config.py`:

```python
# Similarity threshold - bật/tắt bằng boolean
USE_SIMILARITY_THRESHOLD = True      # True: bật, False: tắt
SIMILARITY_THRESHOLD = 0.6           # Giá trị (0-1)

# Số results trả về
TOP_K = 5                            # Tăng/giảm số kết quả

# Chunk settings
CHUNK_SIZE = 1500                    # ký tự mỗi chunk
CHUNK_OVERLAP = 150                  # 10% của chunk_size

# Gemini settings
GEMINI_TEMPERATURE = 0.3             # 0-1 (thấp = chính xác, cao = sáng tạo)
GEMINI_MAX_TOKENS = 2048             # Max length của response
```

## 🎯 Ví Dụ Câu Hỏi

```
👤 Bạn: Hà Nội có những điểm gì đáng tham quan?
👤 Bạn: Du lịch Phú Quốc cần bao nhiêu tiền?
👤 Bạn: Mùa nào tốt nhất để đi du lịch Việt Nam?
👤 Bạn: Đà Nẵng có các hoạt động gì?
👤 Bạn: Cách đi từ Sài Gòn lên Hà Nội?
```

## 🔧 Lệnh Hữu Ích

```bash
# Xem logs chi tiết
python train_rag.py > train.log 2>&1

# Xóa vector store cũ (để training lại)
rmdir /s faiss_index
del faiss_metadata.pkl

# Kiểm tra size vector store
dir faiss_*
```

## ❓ FAQ

**Q: Vector store bao nhiêu GB?**  
A: ~500 MB - 2 GB tùy thuộc vào số chunks

**Q: Truy vấn mất bao lâu?**  
A: ~1-2 giây (search) + ~3-5 giây (Gemini generate)

**Q: Chạy offline được không?**  
A: Không, vì dùng Gemini API online

**Q: Có thể thêm/xóa tài liệu không?**  
A: Có, edit/thêm PDF và chạy `python train_rag.py` lại

**Q: Threshold là cái gì?**  
A: Mức độ tương tự tối thiểu. Nếu document có similarity < threshold sẽ bị loại bỏ

**Q: Làm sao tuỳ chỉnh threshold?**  
A: Edit `config.py`:
```python
USE_SIMILARITY_THRESHOLD = True      # Bật filter
SIMILARITY_THRESHOLD = 0.7           # Chỉ lấy sim > 0.7
```

## 🆘 Troubleshooting

**Error: "GEMINI_API_KEY not found"**
```
→ Tạo file .env và add GEMINI_API_KEY
→ Đảm bảo .env nằm cùng thư mục với app.py
```

**Error: "FAISS index not found"**
```
→ Chạy python train_rag.py trước
→ Đặt PDF file1.pdf và file2.pdf vào thư mục e:\Sỹ\
```

**Slow search?**
```
→ FAISS search nên <100ms. Nếu chậm, check CPU
→ Giảm TOP_K để tìm ít kết quả hơn
```

**Out of memory?**
```
→ Giảm CHUNK_SIZE hoặc CHUNK_OVERLAP
→ Hay dùng model embedding nhỏ hơn
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| PDF pages | 3343 |
| Chunks | ~2200 |
| Embedding dim | 384 |
| Search time | <100ms |
| Generate time | 3-5s |
| Total response | 3-6s |

## 📝 License
MIT

---
**Chúc bạn sử dụng vui vẻ! 🎉**
