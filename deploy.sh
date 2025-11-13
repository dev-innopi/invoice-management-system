#!/bin/sh

set -e

# Check if required tools are installed
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo "Google Cloud SDK is required but not installed. Aborting." >&2; exit 1; }

# Check if user is authenticated with gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="get(account)" 2>/dev/null | grep -q '@'; then
    echo "Not authenticated with Google Cloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Variables
IMAGE_NAME="invoice-management-system"
CONTAINER_NAME="invoice_management_system_container"
GCP_PROJECT_ID="my-poc-94663"
GCP_REGION="asia-south1"
REPOSITORY="invoice-management-system"
# AI_KEY=$1
# Check if OpenAI key is provided
# if [ -z "$AI_KEY" ]; then
#     echo "Error: OpenAI API key not provided. Usage: $0 <openai-key>"
#     exit 1
# fi
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Run container
echo "Running container..."
docker run -d --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest

# Create container commit
echo "Creating container commit..."
docker commit ${CONTAINER_NAME} ${IMAGE_NAME}:committed

# Tag for GCP Artifact Registry
echo "Tagging image for GCP..."
docker tag ${IMAGE_NAME}:committed ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest

# Configure Docker for GCP authentication
echo "Configuring authentication..."
gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Push to GCP Artifact Registry
echo "Pushing to GCP Artifact Registry..."
docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest
# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
# Get the port from the container
PORT=$(docker inspect ${CONTAINER_NAME} --format='{{range $p, $conf := .Config.ExposedPorts}}{{$p}}{{end}}' | cut -d'/' -f1)

# Deploy to Cloud Run with the extracted port
gcloud run deploy ${IMAGE_NAME} \
    --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest \
    --platform managed \
    --region ${GCP_REGION} \
    --project ${GCP_PROJECT_ID} \
    --port ${PORT} \
    --allow-unauthenticated
# Cleanup
echo "Cleaning up..."
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}
docker rm ${IMAGE_NAME}:latest
echo "Deployment complete!"