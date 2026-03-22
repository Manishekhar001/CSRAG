#!/bin/bash
# EC2 user-data script — run once at instance launch
# Sets up Ollama with mxbai-embed-large on Ubuntu 24.04
# Instance type recommendation: t3.medium (4 GB RAM minimum)

set -e

apt-get update -y
apt-get install -y curl

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces so ECS can reach it
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be ready
sleep 15

# Pull the embedding model used by CSRAG
ollama pull mxbai-embed-large

echo "Ollama setup complete. mxbai-embed-large is ready."
