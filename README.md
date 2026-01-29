# Face Verification & Recognition System

Hệ thống xác minh và nhận dạng khuôn mặt sử dụng mô hình **GhostFaceNet** với **AdaFace fine-tuning** và cơ sở dữ liệu vector **FAISS**.

## Tổng quan

Dự án này cung cấp một hệ thống sinh trắc học khuôn mặt hoàn chỉnh với khả năng:
- **Xác minh khuôn mặt (Face Verification)**: So sánh hai ảnh khuôn mặt để xác định có phải cùng một người hay không
- **Nhận dạng khuôn mặt (Face Recognition)**: Tìm kiếm và nhận dạng khuôn mặt trong cơ sở dữ liệu
- **Quản lý cơ sở dữ liệu**: Đăng ký, xóa, và quản lý thông tin người dùng

## Cấu trúc dự án

```
Biometrics/
├── Model/                              # Thư mục chứa model
│   ├── ghostfacenet_fixed.h5          # Model chính được sử dụng
│   ├── ghostfacenet_asian_adaface_backbone.keras
│   └── ghostfacenet_tfkeras_export.keras
├── Notebook/                           # Jupyter notebooks cho training/testing
│   ├── GhostFaceNest_Webface.ipynb
│   └── Test_VNceleb.ipynb
├── Src/
│   ├── backend/                        # Backend API (FastAPI)
│   │   ├── app/
│   │   │   ├── main.py                # Entry point và API endpoints
│   │   │   └── models/
│   │   │       └── api_models.py      # Pydantic models
│   │   ├── services/
│   │   │   ├── face_verification_service.py  # Service xác minh khuôn mặt
│   │   │   └── vector_store/
│   │   │       ├── face_vector_store.py      # FAISS vector store
│   │   │       └── vector_store_service.py   # Service quản lý vector DB
│   │   ├── vector_db/                 # Dữ liệu vector database (đã chuyển vào backend)
│   │   │   ├── faiss_index.bin        # FAISS index file
│   │   │   ├── embeddings.npy         # Face embeddings
│   │   │   └── metadata.json          # Metadata người dùng
│   │   ├── requirements.txt           # Python dependencies
│   │   └── venv/                      # Python virtual environment
│   ├── frontend/                       # Frontend (React + Vite)
│   │   ├── src/
│   │   │   ├── app/
│   │   │   │   ├── App.tsx
│   │   │   │   └── components/
│   │   │   ├── main.tsx
│   │   │   └── styles/
│   │   ├── package.json
│   │   └── vite.config.ts
└── README.md
```

## Công nghệ sử dụng

### Backend
- **FastAPI** - Web framework
- **TensorFlow 2.15** - Deep learning framework
- **GhostFaceNet + AdaFace** - Model nhận dạng khuôn mặt
- **FAISS** - Vector similarity search
- **OpenCV** - Xử lý ảnh
- **NumPy** - Tính toán số học

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **TailwindCSS 4** - CSS framework
- **Radix UI** - UI components
- **Material UI** - UI components
- **Framer Motion** - Animations

## Cài đặt

### Yêu cầu
- Python 3.9+
- Node.js 18+
- npm hoặc pnpm

### Backend

```bash
cd Src/backend
py -3.9 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd Src/frontend
npm install
```

## Chạy ứng dụng

### Backend

```bash
cd Src/backend
.\venv\Scripts\activate 
uvicorn app.main:app --reload
# hoặc
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend sẽ chạy tại: `http://localhost:8000`

### Frontend

```bash
cd Src/frontend
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:3000`

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/verify_faces` | Xác minh hai ảnh khuôn mặt |
| `POST` | `/register_face` | Đăng ký khuôn mặt mới vào database |
| `POST` | `/recognize_face` | Nhận dạng khuôn mặt từ database |
| `GET` | `/persons` | Lấy danh sách người đã đăng ký |
| `GET` | `/persons/{id}` | Lấy thông tin một người |
| `DELETE` | `/persons/{id}` | Xóa người khỏi database |
| `GET` | `/database/stats` | Lấy thống kê database |
| `POST` | `/verify_faces_with_db` | Xác minh khuôn mặt kết hợp kiểm tra database |
| `GET` | `/health` | Kiểm tra trạng thái hệ thống |
