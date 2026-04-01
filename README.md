# Multimodal AI Assistant Platform

RAG Pipeline + Computer Vision Module, diorkestasi dengan Docker Compose.

## Struktur

```
multimodal-ai/
├── docker-compose.yml      ← orchestration utama
├── .env.example            ← template env (copy ke .env)
├── rag_pipeline/
│   ├── Dockerfile
│   ├── requirements.txt    ← sudah dirapikan, no conflict
│   └── src/
├── cv_module/
│   ├── Dockerfile
│   ├── requirements.txt    ← sudah dirapikan, no conflict
│   └── src/
├── nginx/
│   └── nginx.conf          ← reverse proxy + serve frontend
├── frontend/
│   └── index.html          ← UI terminal-themed
└── logs/
```

## Cara Jalankan

### 1. Isi API Key
```bash
cp .env.example .env
# edit .env, isi GROQ_API_KEY
```

### 2. Build & Run
```bash
docker-compose up --build
```

Pertama kali build agak lama (~10-20 menit) karena download PyTorch.
Selanjutnya pakai cache, jadi cepat.

### 3. Akses
| Service     | URL                          |
|-------------|------------------------------|
| UI          | http://localhost             |
| RAG API     | http://localhost/rag/docs    |
| CV API      | http://localhost/cv/docs     |
| MLflow      | http://localhost:5000        |

## Perintah Berguna

```bash
# jalankan di background
docker-compose up -d

# lihat log semua service
docker-compose logs -f

# lihat log service tertentu
docker-compose logs -f rag
docker-compose logs -f cv

# stop semua
docker-compose down

# stop + hapus volumes (reset data)
docker-compose down -v

# rebuild satu service saja
docker-compose up --build rag
```

## Kompatibilitas Versi (sudah ditest)

| Package         | Versi    | Alasan                              |
|-----------------|----------|-------------------------------------|
| torch           | 2.1.0    | stable, fix DLL issue Windows       |
| numpy           | 1.26.4   | <2.0 wajib untuk torch 2.1.0        |
| transformers    | 4.35.2   | kompatibel dengan torch 2.1.0       |
| chromadb        | 0.5.3    | kompatibel dengan langchain-chroma  |
| langchain-chroma| 0.1.4    | exclude 0.5.4 & 0.5.5               |
| timm            | 0.9.16   | stable untuk open-clip              |
