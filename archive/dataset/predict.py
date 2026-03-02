import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from PIL import Image
import os
import sys
from pathlib import Path

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# LOAD DATASET (ĐỂ LẤY CLASS ĐÚNG THỨ TỰ)
# =========================
train_dataset = datasets.ImageFolder("train")
classes = train_dataset.classes
num_classes = len(classes)

print("Number of classes:", num_classes)
print("Classes:", classes)

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load("tourism_model.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# =========================
# TRANSFORM (PHẢI GIỐNG LÚC TRAIN)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# =========================
# PREDICT ALL IMAGES IN TEST FOLDER
# =========================
test_folder = "Test"

if not os.path.exists(test_folder):
    print("Test folder not found!")
    exit()

for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(test_folder, filename)
        
        print(f"\nProcessing: {filename}")
        
        try:
            img = Image.open(image_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            
            # =========================
            # PREDICTION
            # =========================
            with torch.no_grad():
                outputs = model(img)
                probs = F.softmax(outputs, dim=1)
            
            # Top 1
            _, predicted = torch.max(probs, 1)
            top1_class = classes[predicted.item()]
            top1_conf = float(probs[0][predicted.item()])
            
            print("TOP 1 Prediction:")
            print("Class:", top1_class)
            print("Confidence:", round(top1_conf, 4))
            
            # Top 5
            print("Top 5 Predictions:")
            top5 = torch.topk(probs, 5)
            
            for i in range(5):
                class_name = classes[top5.indices[0][i]]
                confidence = float(top5.values[0][i])
                print(f"{i+1}. {class_name} - {confidence:.4f}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nAll images processed!")

# =========================
# CHẾ ĐỘ Q&A - ĐẶT CÂU HỎI VỀ DU LỊCH
# =========================
rag_dir = Path(__file__).resolve().parent.parent.parent / "RAG"
sys.path.insert(0, str(rag_dir))
os.chdir(rag_dir)

print("\n" + "="*60)
print("Bạn có thể đặt câu hỏi về du lịch Việt Nam.")
print("Nhập 'exit' để thoát.")
print("="*60 + "\n")

try:
    from embedding_service import EmbeddingService
    from vector_store import FAISSVectorStore
    from gemini_rag import GeminiRAG

    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore()
    vector_store.load()
    rag = GeminiRAG(vector_store, embedding_service)
    rag.interactive_chat()
except FileNotFoundError as e:
    print("❌ FAISS vector store chưa được tạo!")
    print("💡 Hãy chạy lệnh sau trước: python train_rag.py")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo Q&A: {e}")