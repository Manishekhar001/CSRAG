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
                              EC2 Instance (Ubuntu)
                                ├── csrag container     (port 8000)
                                ├── postgres container  (internal)
                                └── Ollama native       (port 11434)
```

Public URL: http://YOUR_EC2_PUBLIC_IP:8000

---

## Step 1 — Create a Docker Hub Account and Repository

Docker Hub is free and is used as the container registry (same as BasicRAG).

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
   - **AMI**: Ubuntu Server 24.04 LTS
   - **Instance type**: `t3.medium` (4 GB RAM — Ollama needs minimum 4 GB)
     - t2.micro is free tier but only 1 GB RAM — too small for Ollama
   - **Key pair**: Create new → name it `csrag-key` → download the `.pem` file
   - **Security group**: Allow inbound:
     - SSH (port 22) from My IP
     - Custom TCP port 8000 from Anywhere (0.0.0.0/0)
   - **Storage**: 20 GB

3. Launch. Note the **Public IPv4 address**.

---

## Step 3 — SSH into EC2

```bash
ssh -i csrag-key.pem ubuntu@YOUR_PUBLIC_IP
```

---

## Step 4 — Install Docker on EC2

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

## Step 5 — Install Ollama on EC2

```bash
curl -fsSL https://ollama.com/install.sh | sh

sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 10

ollama pull mxbai-embed-large
ollama list
```

---

## Step 6 — Create the Docker Network and Start Postgres

The CSRAG container and Postgres container communicate over a Docker network.
You only do this once — Postgres data persists in a Docker volume.

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

docker ps
```

Wait a few seconds, then verify Postgres is healthy:

```bash
docker exec csrag-postgres pg_isready -U postgres
```

---

## Step 7 — Create the .env File on EC2

The `.env` lives only on the EC2. It is never in GitHub.

```bash
nano /home/ubuntu/.env
```

Paste and fill in your values:

```
GROQ_API_KEY=your_actual_groq_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_actual_qdrant_key
TAVILY_API_KEY=your_actual_tavily_key
POSTGRES_URI=postgresql://postgres:postgres@csrag-postgres:5432/postgres?sslmode=disable
OLLAMA_BASE_URL=http://172.17.0.1:11434
EMBEDDING_MODEL=mxbai-embed-large
EMBEDDING_DIMENSION=1024
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

Note on OLLAMA_BASE_URL: `172.17.0.1` is the Docker bridge gateway — the IP that Docker containers use to reach the EC2 host. To confirm your exact IP:

```bash
ip route | grep docker
# Look for something like: 172.17.0.0/16 dev docker0
# The gateway is 172.17.0.1
```

---

## Step 8 — Run the App for the First Time

Pull and run the CSRAG container manually the first time:

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

Test from EC2:

```bash
curl http://localhost:8000/health
```

Test from your local browser:

```
http://YOUR_EC2_PUBLIC_IP:8000/docs
```

---

## Step 9 — Add GitHub Secrets

Go to your GitHub repo → **Settings → Secrets and variables → Actions** → **New repository secret**.

Add these five secrets — same pattern as BasicRAG:

| Secret name | Value |
|-------------|-------|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (see below) |
| `EC2_HOST` | your EC2 public IP (e.g., 54.123.45.67) |
| `EC2_USERNAME` | `ubuntu` |
| `EC2_SSH_KEY` | full contents of your csrag-key.pem file |

**How to get DOCKERHUB_TOKEN:**
1. Go to https://hub.docker.com → Account Settings → Security
2. Click **New Access Token**
3. Name: `github-actions`
4. Permissions: Read, Write, Delete
5. Generate → copy the token

**How to get EC2_SSH_KEY:**
- Open `csrag-key.pem` in a text editor
- Copy everything including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`
- Paste that as the secret value

---

## Step 10 — Push to Main and Watch It Deploy

```bash
git add .
git commit -m "initial deployment setup"
git push origin main
```

Go to **GitHub → your repo → Actions tab**.

You will see the CI Pipeline running with 3 jobs:

```
Unit Tests ─────────────────────► pass
     ↓
Docker Build and Test ───────────► pass
     ↓
Deploy to EC2 ───────────────────► push to Docker Hub → SSH → docker pull + run
```

Every job must pass before the next one starts. Deploy only runs on pushes to main (not PRs).

---

## What Happens on Every Push to Main

1. GitHub starts a fresh Ubuntu runner
2. **Job 1** — installs dependencies, runs all pytest tests with fake API keys
3. **Job 2** — builds the Docker image, runs `python -c "from app.main import app"` to verify it loads
4. **Job 3** — logs into Docker Hub, builds + pushes the image, SSHes into EC2, pulls the new image, stops/removes the old container, starts the new one with the `.env` file
5. Old Docker images are pruned to save disk space

---

## Useful Commands on EC2

```bash
# See all running containers
docker ps

# Live app logs
docker logs -f csrag

# Live postgres logs
docker logs -f csrag-postgres

# Restart the app container
docker restart csrag

# Manual redeploy without GitHub Actions
docker pull YOUR_DOCKERHUB_USERNAME/csrag:latest
docker stop csrag && docker rm csrag
docker run -d --name csrag --restart always -p 8000:8000 \
  --env-file /home/ubuntu/.env --network csrag-net \
  YOUR_DOCKERHUB_USERNAME/csrag:latest

# Check disk usage
df -h
docker system df

# Clean up unused images
docker image prune -f
```

---

## Cost

| Resource | Cost |
|----------|------|
| EC2 t3.medium (24/7) | ~$30/month |
| EBS 20 GB | ~$2/month |
| Docker Hub (public repo) | Free |
| Qdrant free tier | Free |
| Groq free tier | Free |
| Tavily free tier | Free |
| **Total** | **~$32/month** |

Stop EC2 when not using it to save money:
```bash
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID
```
