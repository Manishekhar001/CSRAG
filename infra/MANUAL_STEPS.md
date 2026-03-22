# CSRAG — EC2 Deployment Manual Steps

Follows the exact same pattern as BasicRAG project.
One EC2 instance. Docker Hub as the registry. Plain docker run.

---

## Architecture

```
Your machine
     ↓  git push to main
GitHub Actions
  ├── JOB 1: Run all pytest tests
  ├── JOB 2: Build Docker image + smoke test
  └── JOB 3: Push to Docker Hub → SSH into EC2 → docker pull + run
                                        ↓
                              EC2 Instance t3.small (Ubuntu)
                                ├── csrag container     (port 8000)
                                ├── postgres container  (internal)
                                └── Ollama native       (port 11434)
```

Public URL: http://YOUR_EC2_PUBLIC_IP:8000

---

## Embedding Model Choice

This project uses **nomic-embed-text** (768 dimensions, 274MB, ~900MB RAM) instead of
mxbai-embed-large (1024 dimensions, 670MB, ~2.5GB RAM).

Reason: AWS free tier only offers t3.micro (1GB) and t3.small (2GB). mxbai-embed-large
requires 2.5GB RAM minimum and will OOM-crash on both. nomic-embed-text fits comfortably
on t3.small with swap space added.

nomic-embed-text produces equally good embeddings for RAG — the dimension difference
(768 vs 1024) has negligible effect on retrieval quality.

---

## Step 1 — Create a Docker Hub Account and Repository

1. Go to https://hub.docker.com and sign up (free)
2. Click **Create Repository**
3. Name: `csrag`
4. Visibility: Public (free tier)
5. Create

Your image will be pushed as: `YOUR_USERNAME/csrag:latest`

---

## Step 2 — Launch EC2 Instance

1. Go to **AWS Console → EC2 → Launch Instance**
2. Fill in:
   - **Name**: `csrag-server`
   - **AMI**: Ubuntu Server 24.04 LTS (64-bit x86)
   - **Instance type**: `t3.small` (2 vCPU, 2 GB RAM)
     - t3.micro (1GB) is too small — even with swap, Ollama will be unusable
     - t3.small (2GB) + 4GB swap = effectively 6GB — sufficient for nomic-embed-text
   - **Key pair**: Create new → name it `csrag-key` → download the `.pem` file
   - **Security group**: Allow inbound:
     - SSH port 22 from My IP
     - Custom TCP port 8000 from Anywhere (0.0.0.0/0)
   - **Storage**: 20 GB (increase if needed for Docker images)

3. Launch. Note the **Public IPv4 address**.

---

## Step 3 — SSH into EC2

```bash
ssh -i csrag-key.pem ubuntu@YOUR_PUBLIC_IP
```

---

## Step 4 — Add Swap Space (Critical for t3.small)

t3.small has 2GB RAM. Adding 4GB swap gives the system 6GB effective memory,
which is enough for nomic-embed-text + Postgres + the CSRAG container.

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Make swap permanent across reboots:

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Verify swap is active:

```bash
free -h
```

You should now see 4GB under the Swap row.

---

## Step 5 — Install Docker on EC2

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu
newgrp docker
docker --version
```

---

## Step 6 — Install Ollama on EC2

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Configure Ollama to listen on all interfaces (required for Docker container to reach it):

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 10
```

Pull nomic-embed-text (274MB — takes about 1 minute):

```bash
ollama pull nomic-embed-text
ollama list
```

You should see `nomic-embed-text` in the list.

---

## Step 7 — Create Docker Network and Start Postgres

```bash
docker network create csrag-net

docker run -d \
  --name csrag-postgres \
  --restart always \
  --network csrag-net \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16
```

Verify Postgres is healthy:

```bash
docker exec csrag-postgres pg_isready -U postgres
```

---

## Step 8 — Find Docker Bridge IP

```bash
ip route | grep docker
```

Look for a line like: `172.17.0.0/16 dev docker0 ... src 172.17.0.1`

The IP you need is `172.17.0.1` — this is how your CSRAG container reaches Ollama on the host.

---

## Step 9 — Create the .env File on EC2

```bash
nano /home/ubuntu/.env
```

Paste and fill in your real values:

```
GROQ_API_KEY=your_actual_groq_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_actual_qdrant_key
TAVILY_API_KEY=your_actual_tavily_key
POSTGRES_URI=postgresql://postgres:postgres@csrag-postgres:5432/postgres?sslmode=disable
OLLAMA_BASE_URL=http://172.17.0.1:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
COLLECTION_NAME=csrag_documents
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
MEMORY_LLM_MODEL=llama-3.3-70b-versatile
MEMORY_LLM_TEMPERATURE=0.0
RETRIEVAL_K=4
CHUNK_SIZE=900
CHUNK_OVERLAP=150
STM_MESSAGE_THRESHOLD=6
CRAG_UPPER_THRESHOLD=0.7
CRAG_LOWER_THRESHOLD=0.3
SRAG_MAX_RETRIES=2
MAX_REWRITE_TRIES=2
TAVILY_MAX_RESULTS=5
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*
API_HOST=0.0.0.0
API_PORT=8000
```

Save: Ctrl+X → Y → Enter

---

## Step 10 — First Deploy

After GitHub Actions pushes the image (Step 12 below), run:

```bash
docker pull YOUR_DOCKERHUB_USERNAME/csrag:latest

docker run -d \
  --name csrag \
  --restart always \
  -p 8000:8000 \
  --env-file /home/ubuntu/.env \
  --network csrag-net \
  YOUR_DOCKERHUB_USERNAME/csrag:latest
```

Check it started:

```bash
docker ps
docker logs csrag
```

Test it:

```bash
curl http://localhost:8000/health
```

---

## Step 11 — Add GitHub Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions** → **New repository secret**.

| Secret name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `EC2_HOST` | your EC2 public IP |
| `EC2_USERNAME` | `ubuntu` |
| `EC2_SSH_KEY` | full contents of csrag-key.pem |

---

## Step 12 — Push to Main

```bash
git add .
git commit -m "deploy to ec2"
git push origin main
```

GitHub Actions → CI Pipeline → 3 jobs → green → app is live at:
```
http://YOUR_EC2_PUBLIC_IP:8000/docs
```

---

## Useful Commands on EC2

```bash
# See running containers
docker ps

# App logs (live)
docker logs -f csrag

# Postgres logs
docker logs -f csrag-postgres

# Check memory and swap usage
free -h

# Check disk usage
df -h
docker system df

# Manual redeploy
docker pull YOUR_DOCKERHUB_USERNAME/csrag:latest
docker stop csrag && docker rm csrag
docker run -d --name csrag --restart always -p 8000:8000 \
  --env-file /home/ubuntu/.env --network csrag-net \
  YOUR_DOCKERHUB_USERNAME/csrag:latest

# Clean up old images
docker image prune -f
```

---

## Cost (t3.small, 24/7)

| Resource | Cost |
|----------|------|
| EC2 t3.small (24/7) | ~$15/month |
| EBS 20 GB | ~$2/month |
| Docker Hub public repo | Free |
| Qdrant free tier | Free |
| Groq free tier | Free |
| Tavily free tier | Free |
| **Total** | **~$17/month** |

Stop EC2 when not using it:
```bash
# From AWS Console → EC2 → Instances → select instance → Instance State → Stop
```
