# 2api.ai - Quick Start Guide

## 🚀 Chạy Ngay Trong 5 Phút

### Option 1: Docker (Recommended)

```bash
# 1. Unzip và vào thư mục
unzip 2api-ai.zip
cd 2api-ai

# 2. Tạo file .env với API keys
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-key
EOF

# 3. Chạy với Docker Compose
docker-compose up -d

# 4. Test
curl http://localhost:8000/health
```

### Option 2: Python Local

```bash
# 1. Unzip và vào thư mục
unzip 2api-ai.zip
cd 2api-ai

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API keys
export OPENAI_API_KEY=sk-your-openai-key
export ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
export GOOGLE_API_KEY=your-google-key

# 5. Chạy server (từ repo root, KHÔNG cd vào src/)
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000

# 6. Test (terminal khác)
curl http://localhost:8000/health
```

---

## 📋 Chạy Contract Tests

```bash
cd 2api-ai
pip install pytest
pytest tests/test_contracts.py -v
```

Expected output:
```
tests/test_contracts.py::TestStreamingContract::test_chunk_has_required_fields PASSED
tests/test_contracts.py::TestStreamingContract::test_stream_ends_with_done PASSED
tests/test_contracts.py::TestToolCallingContract::test_tool_call_has_required_fields PASSED
...
```

---

## 🔧 Test API

### Health Check
```bash
curl http://localhost:8000/health
```

### List Models
```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer 2api_test_key"
```

### Chat Completion
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer 2api_test_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming Chat
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer 2api_test_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

---

## 📁 Project Structure

```
2api-ai/
├── docs/                    # 📚 Specifications
│   ├── SPEC_INDEX.md        # Start here - links to all specs
│   ├── ARCHITECTURE.md
│   ├── openapi.yaml
│   ├── STREAMING_SPEC.md
│   ├── TOOL_CALLING_SPEC.md
│   ├── RETRY_FALLBACK_POLICY.md
│   ├── ERROR_TAXONOMY.md
│   └── MULTI_TENANT_DESIGN.md
├── src/                     # 💻 Source Code
│   ├── core/                # Data models & errors
│   ├── adapters/            # Provider adapters
│   ├── routing/             # Intelligent router
│   ├── sdk/                 # Python & JS SDKs
│   └── server.py            # FastAPI server
├── tests/                   # 🧪 Tests
│   └── test_contracts.py    # Contract tests
├── .github/workflows/       # 🔄 CI/CD
│   └── ci.yml
├── Dockerfile
├── docker-compose.yaml
└── requirements.txt
```

---

## 📖 Documentation

Bắt đầu với [docs/SPEC_INDEX.md](docs/SPEC_INDEX.md) - trang master link tới tất cả specs.

---

## ⚠️ Notes

1. **API Keys**: Ít nhất cần 1 provider key (OpenAI, Anthropic, hoặc Google)
2. **Test Key**: Dùng bất kỳ key nào bắt đầu bằng `2api_` để test
3. **Database**: Chưa có database integration - usage tracking là placeholder
4. **Production**: Cần thêm Redis, PostgreSQL cho rate limiting và billing

---

## 🆘 Troubleshooting

### Import Error
```bash
# QUAN TRỌNG: Chạy từ repo root, KHÔNG cd vào src/
cd 2api-ai
python -m uvicorn src.server:app --port 8000
```

### Port Already in Use
```bash
# Dùng port khác
python -m uvicorn src.server:app --port 8001
```

### Missing Dependencies
```bash
pip install fastapi uvicorn httpx pydantic
```
