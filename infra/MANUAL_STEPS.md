# CSRAG — AWS Deployment Manual Steps

Complete step-by-step guide. Do these ONCE before the CD pipeline can run.

---

## Architecture

```
Internet → ALB → ECS Fargate (CSRAG container)
                       ↓
              RDS PostgreSQL (LTM + STM)
              Qdrant Cloud (vector DB — external)
              Groq API (LLM — external)
              Tavily API (web search — external)
              EC2 t3.medium (Ollama embeddings — separate)
```

---

## Step 1 — AWS Prerequisites

### 1.1 Create IAM User for GitHub Actions

1. Go to **IAM → Users → Create user**
2. Username: `csrag-github-actions`
3. Attach these policies directly:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
   - `AmazonVPCFullAccess` (for networking)
4. After creation → **Security credentials → Create access key**
5. Choose **Third-party service**
6. Save `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` — you will add these to GitHub Secrets

---

## Step 2 — Networking (VPC)

### 2.1 Use default VPC or create one

If using default VPC, note its **VPC ID** and **Subnet IDs** (need at least 2 in different AZs).

```bash
aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId"
aws ec2 describe-subnets --filters "Name=defaultForAz,Values=true" --query "Subnets[*].SubnetId"
```

### 2.2 Create Security Groups

**ALB security group** (public-facing):
1. EC2 → Security Groups → Create
2. Name: `csrag-alb-sg`
3. Inbound: HTTP 80 from 0.0.0.0/0, HTTPS 443 from 0.0.0.0/0
4. Outbound: All traffic

**ECS security group** (private):
1. Name: `csrag-ecs-sg`
2. Inbound: Port 8000 from `csrag-alb-sg` only
3. Outbound: All traffic

**RDS security group**:
1. Name: `csrag-rds-sg`
2. Inbound: Port 5432 from `csrag-ecs-sg` only
3. Outbound: All traffic

**Ollama security group**:
1. Name: `csrag-ollama-sg`
2. Inbound: Port 11434 from `csrag-ecs-sg` only, Port 22 from your IP
3. Outbound: All traffic

---

## Step 3 — ECR (Container Registry)

```bash
aws ecr create-repository \
  --repository-name csrag \
  --region us-east-1 \
  --image-scanning-configuration scanOnPush=true
```

Note the **repository URI** — looks like:
`123456789.dkr.ecr.us-east-1.amazonaws.com/csrag`

---

## Step 4 — RDS PostgreSQL (LTM + STM)

1. Go to **RDS → Create database**
2. Engine: PostgreSQL 16
3. Template: Free tier (dev) or Production
4. DB instance identifier: `csrag-postgres`
5. Master username: `postgres`
6. Master password: choose a strong password and note it
7. Instance class: `db.t3.micro` (dev) or `db.t3.medium` (prod)
8. Storage: 20 GB GP3
9. VPC: your VPC
10. Security group: `csrag-rds-sg`
11. Public access: No
12. Database name: `csrag`
13. Create database

After creation, note the **Endpoint** (looks like `csrag-postgres.xxxx.us-east-1.rds.amazonaws.com`).

Build the `POSTGRES_URI`:
```
postgresql://postgres:YOUR_PASSWORD@csrag-postgres.xxxx.us-east-1.rds.amazonaws.com:5432/csrag
```

---

## Step 5 — EC2 for Ollama (Embeddings)

Ollama must run on a machine the ECS container can reach. ECS Fargate cannot run Ollama directly.

1. Go to **EC2 → Launch Instance**
2. Name: `csrag-ollama`
3. AMI: Ubuntu Server 24.04 LTS
4. Instance type: `t3.medium` (minimum — mxbai-embed-large needs ~4GB RAM)
5. Key pair: create or choose existing
6. Security group: `csrag-ollama-sg`
7. Storage: 20 GB
8. In **Advanced details → User data**, paste this script:

```bash
#!/bin/bash
apt-get update -y
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 10
OLLAMA_HOST=0.0.0.0 ollama pull mxbai-embed-large
```

9. Launch instance

Note the **Private IP** of the EC2 instance. This becomes your `OLLAMA_BASE_URL`:
```
http://PRIVATE_IP:11434
```

The Ollama service must listen on `0.0.0.0` to be reachable from ECS. SSH in and verify:
```bash
sudo systemctl edit ollama
```
Add:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```
Then: `sudo systemctl restart ollama`

---

## Step 6 — ECS Cluster

1. Go to **ECS → Clusters → Create cluster**
2. Cluster name: `csrag-cluster`
3. Infrastructure: **AWS Fargate** (serverless — no EC2 to manage)
4. Create cluster

---

## Step 7 — ECS Task Definition

1. Go to **ECS → Task definitions → Create new task definition**
2. Task definition family: `csrag-task`
3. Launch type: Fargate
4. CPU: 1 vCPU, Memory: 2 GB (minimum — increase for production)
5. Task role: create new role `csrag-task-role` with `AmazonECSTaskExecutionRolePolicy`
6. Container:
   - Name: `csrag`
   - Image URI: `YOUR_ECR_URI/csrag:latest`
   - Port: 8000
   - Essential: Yes
7. Environment variables — add ALL of these:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your Groq key |
| `QDRANT_URL` | your Qdrant Cloud URL |
| `QDRANT_API_KEY` | your Qdrant key |
| `TAVILY_API_KEY` | your Tavily key |
| `POSTGRES_URI` | postgresql://postgres:PASSWORD@RDS_ENDPOINT:5432/csrag |
| `OLLAMA_BASE_URL` | http://OLLAMA_EC2_PRIVATE_IP:11434 |
| `LOG_LEVEL` | INFO |
| `ALLOWED_ORIGINS` | * |

8. Log configuration:
   - Log driver: awslogs
   - Group: `/ecs/csrag`
   - Region: `us-east-1`
   - Stream prefix: `ecs`
   - Create the log group in CloudWatch first: `aws logs create-log-group --log-group-name /ecs/csrag`

9. Create task definition

---

## Step 8 — Application Load Balancer

1. Go to **EC2 → Load Balancers → Create load balancer**
2. Type: Application Load Balancer
3. Name: `csrag-alb`
4. Scheme: Internet-facing
5. IP type: IPv4
6. VPC: your VPC
7. Subnets: select at least 2 public subnets
8. Security group: `csrag-alb-sg`
9. Listeners: HTTP 80
10. Target group:
    - Name: `csrag-tg`
    - Target type: IP (required for Fargate)
    - Protocol: HTTP, Port: 8000
    - Health check path: `/health`
    - Healthy threshold: 2, Unhealthy: 3, Interval: 30s
11. Create load balancer

Note the **DNS name** of the ALB — this is your public URL.

---

## Step 9 — ECS Service

1. Go to **ECS → csrag-cluster → Create service**
2. Launch type: Fargate
3. Task definition: `csrag-task` (latest)
4. Service name: `csrag-service`
5. Desired tasks: 1 (scale up for production)
6. VPC: your VPC
7. Subnets: private subnets
8. Security group: `csrag-ecs-sg`
9. Public IP: disabled (ALB handles public traffic)
10. Load balancer: Application Load Balancer → `csrag-alb`
11. Target group: `csrag-tg`
12. Create service

---

## Step 10 — GitHub Secrets

Go to your GitHub repository → **Settings → Secrets and variables → Actions → New repository secret**.

Add these secrets:

| Secret name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | from Step 1.1 |
| `AWS_SECRET_ACCESS_KEY` | from Step 1.1 |

The CD pipeline uses these to push to ECR and deploy to ECS.

---

## Step 11 — Push and Verify

1. Push to the `main` branch
2. Watch **Actions** tab in GitHub — CI runs first, then CD
3. CD will:
   - Build the Docker image
   - Push to ECR
   - Update the ECS task definition with the new image
   - Deploy to ECS (rolling update)
4. Check ECS → csrag-cluster → csrag-service → Tasks
5. Check CloudWatch → Log groups → `/ecs/csrag` for app logs
6. Hit `http://YOUR_ALB_DNS/health` — should return `{"status": "healthy"}`
7. Hit `http://YOUR_ALB_DNS/docs` — Swagger UI

---

## Cost Estimate (us-east-1, per month)

| Resource | Cost |
|----------|------|
| ECS Fargate (1 vCPU / 2GB, 24/7) | ~$35 |
| RDS PostgreSQL db.t3.micro | ~$15 |
| EC2 t3.medium (Ollama) | ~$30 |
| ALB | ~$18 |
| ECR storage | ~$1 |
| CloudWatch logs | ~$2 |
| **Total** | **~$101/month** |

To reduce costs: stop EC2 and RDS when not needed. Use `db.t3.micro` for dev.

---

## Environment Variable Reference (ECS Task Definition)

All env vars the container needs — do not leave any blank:

```
GROQ_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
TAVILY_API_KEY=
POSTGRES_URI=postgresql://postgres:PASSWORD@RDS_HOST:5432/csrag
OLLAMA_BASE_URL=http://OLLAMA_PRIVATE_IP:11434
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
