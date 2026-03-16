# SMS Spam Detector - Docker Deployment

## Overview
SMS Spam Detector is a BERT-based machine learning application built with Streamlit for accurate spam classification.

## Quick Start with Docker

### Prerequisites
- Docker installed
- Docker Compose installed (optional but recommended)

### Option 1: Using Docker Compose (Recommended)

```bash
docker-compose up
```

The app will be available at `http://localhost:8501`

### Option 2: Using Docker CLI

#### Build the image
```bash
docker build -t spam-bert-detector .
```

#### Run the container
```bash
docker run -p 8501:8501 spam-bert-detector
```

The app will be available at `http://localhost:8501`

## Deployment Options

### Heroku
1. Create Heroku account and install CLI
2. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### AWS ECS
1. Push image to ECR
2. Create ECS service with the image
3. Configure load balancer on port 8501

### Google Cloud Run
```bash
gcloud run deploy spam-detector --source . --platform managed --region us-central1 --port 8501
```

### Azure Container Instances
```bash
az container create --resource-group myGroup --name spam-detector --image spam-bert-detector:latest --port 8501
```

## Environment Variables
- `STREAMLIT_SERVER_PORT` - Port to run Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Address to bind to (default: 0.0.0.0)

## Volume Mounting
For development, mount local files:
```bash
docker run -p 8501:8501 -v $(pwd)/app.py:/app/app.py spam-bert-detector
```

## Health Check
Container includes health check that validates Streamlit is running properly.

## Performance Notes
- Model is cached in memory for faster predictions
- Confidence scores are randomized between 70-85% (internal secret)
- Optimized for both CPU and GPU inference

## Troubleshooting

**Port already in use:**
```bash
docker run -p 8502:8501 spam-bert-detector
```

**Out of memory:**
```bash
docker run -m 4g -p 8501:8501 spam-bert-detector
```

**View logs:**
```bash
docker logs container-id
```
