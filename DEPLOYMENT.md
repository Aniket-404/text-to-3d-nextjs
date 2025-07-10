# 3Dify Production Deployment Guide

This guide covers deploying 3Dify in production environments with full NeRF capabilities.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.9+ (for local development)
- GPU support recommended for optimal NeRF performance

### One-Command Deploy
```bash
# Clone and deploy in one go
git clone <your-repo-url>
cd text-to-3d-nextjs
npm run deploy:production
```

## 🛠 Manual Production Setup

### 1. Environment Configuration
```bash
# Copy the example environment file
cp .env.production.example .env.production

# Edit with your actual values
nano .env.production
```

**Required Environment Variables:**
```env
# Firebase (required for auth)
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
# ... other Firebase vars

# AI APIs (at least one required)
HUGGINGFACE_API_KEY=your_hf_token
GEMINI_API_KEY=your_gemini_key

# Storage (required)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### 2. Docker Deployment (Recommended)
```bash
# Build images
npm run docker:build
npm run docker:build-api

# Start services
npm run docker:up

# Check health
curl http://localhost:3000/api/health
curl http://localhost:5000/health
```

### 3. Manual Deployment

#### Frontend
```bash
# Install and build
npm ci --only=production
npm run build:production

# Start
npm run start:production
```

#### Backend API
```bash
# Install dependencies
cd python-api
pip install -r requirements.txt

# Start with gunicorn (production)
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app

# Or start with Flask (development)
python app.py
```

## 🏗 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js App   │    │   Python API    │    │   External      │
│   (Frontend)    │────│   (NeRF Engine)  │────│   Services      │
│   Port 3000     │    │   Port 5000     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐              ┌─────────┐          ┌─────────────┐
    │ Firebase│              │ PyTorch │          │ Cloudinary  │
    │  Auth   │              │  NeRF   │          │  Storage    │
    └─────────┘              └─────────┘          └─────────────┘
```

## 🔧 Production Optimizations

### Performance
- **GPU Acceleration**: CUDA support for NeRF training
- **Model Caching**: Pre-downloaded Stable Diffusion models
- **Image Optimization**: WebP/AVIF formats with Next.js
- **Code Splitting**: Optimized JavaScript bundles
- **CDN Integration**: Cloudinary for global asset delivery

### Security
- **HTTPS Enforcement**: SSL certificates configured
- **CORS Protection**: Proper origins configured
- **Input Validation**: Sanitized user inputs
- **Rate Limiting**: API request throttling
- **Security Headers**: XSS, CSRF protection

### Scalability
- **Horizontal Scaling**: Load balancer ready
- **Job Queue**: Background NeRF processing
- **Resource Limits**: Memory and CPU constraints
- **Health Monitoring**: Built-in health checks

## 📊 Monitoring & Logging

### Health Checks
```bash
# Frontend health
curl http://localhost:3000/api/health

# Backend health  
curl http://localhost:5000/health

# Docker health
docker-compose ps
```

### Logs
```bash
# View frontend logs
docker logs 3dify-frontend

# View API logs
docker logs 3dify-api

# Follow logs in real-time
docker-compose logs -f
```

### Metrics
The application provides metrics at:
- Frontend: `/api/health`
- Backend: `/health`
- Disk usage, memory, active jobs

## 🚨 Troubleshooting

### Common Issues

#### "NeRF training failed"
- Check GPU availability: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check disk space: `df -h`

#### "Failed to generate image"
- Verify API keys in `.env.production`
- Check Hugging Face token: `python python-api/test_hf_auth.py`
- Test Gemini API: `python python-api/test_gemini.py`

#### "Upload failed"
- Check Cloudinary credentials
- Verify network connectivity
- Check file size limits

#### High Memory Usage
- Adjust Docker memory limits in `docker-compose.yml`
- Reduce concurrent jobs: `MAX_CONCURRENT_JOBS=5`
- Enable model caching: `TORCH_HOME=/app/model_cache`

### Performance Tuning

#### For CPU-Only Deployment
```env
CUDA_VISIBLE_DEVICES=""
TORCH_NUM_THREADS=4
```

#### For GPU Deployment
```env
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0"
```

#### Memory Optimization
```env
MAX_WORKERS=2
NERF_TRAINING_TIMEOUT=300
MAX_CONCURRENT_JOBS=3
```

## 🔄 Updates & Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
npm run deploy:production

# Or with Docker
docker-compose down
docker-compose pull
docker-compose up -d
```

### Database Maintenance
- Firebase Firestore: Automatic scaling
- Clean up old models: Built-in cleanup jobs
- Monitor storage usage in Firebase console

### Model Cache Management
```bash
# Clear model cache
rm -rf python-api/model_cache/*

# Prewarm cache (optional)
python -c "from transformers import pipeline; pipeline('text-to-image')"
```

## 🛡 Security Best Practices

1. **API Keys**: Store in environment variables, never commit
2. **HTTPS**: Use SSL certificates in production
3. **Firewall**: Restrict access to ports 3000, 5000
4. **Updates**: Keep dependencies updated
5. **Monitoring**: Set up alerts for unusual activity

## 📈 Scaling

### Horizontal Scaling
- Load balancer (Nginx/HAProxy)
- Multiple frontend instances
- Redis for session management
- Database read replicas

### Vertical Scaling
- GPU instances for NeRF training
- More RAM for concurrent jobs
- SSD storage for faster I/O

## 🆘 Support

- **Documentation**: [NERF_README.md](./NERF_README.md)
- **API Reference**: [API.md](./API.md)
- **GitHub Issues**: Report bugs and feature requests
- **Discord**: Community support channel

## 📋 Production Checklist

- [ ] Environment variables configured
- [ ] Firebase project set up
- [ ] Cloudinary account configured
- [ ] AI API keys validated
- [ ] SSL certificates installed
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Performance testing completed
- [ ] Security audit passed

---

**Ready for Production!** 🎉

Your 3Dify application is now running with full NeRF capabilities, ready to generate high-quality 3D models from text prompts and images.
