#!/bin/bash
#
# JamPacked Creative Intelligence Deployment Script
# Supports Docker Compose and Kubernetes deployments
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DEPLOYMENT_TYPE="${1:-docker}"
ENVIRONMENT="${2:-production}"

echo -e "${GREEN}🚀 JamPacked Creative Intelligence Deployment${NC}"
echo "=================================="
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Environment: $ENVIRONMENT"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker found${NC}"
    
    # Check deployment-specific tools
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}❌ kubectl is not installed${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ kubectl found${NC}"
        
        if ! command -v helm &> /dev/null; then
            echo -e "${YELLOW}⚠ Helm not found (optional)${NC}"
        else
            echo -e "${GREEN}✓ Helm found${NC}"
        fi
    fi
    
    # Check if MCP SQLite server exists
    MCP_SERVER_PATH="${MCP_SERVER_PATH:-/Users/pulser/Documents/GitHub/mcp-sqlite-server}"
    if [ ! -d "$MCP_SERVER_PATH" ]; then
        echo -e "${YELLOW}⚠ MCP SQLite server not found at $MCP_SERVER_PATH${NC}"
        echo "  Set MCP_SERVER_PATH environment variable if it's in a different location"
    else
        echo -e "${GREEN}✓ MCP SQLite server found${NC}"
    fi
    
    echo ""
}

# Function to build Docker images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Build core image
    echo "Building jampacked/core..."
    docker build -f deployment/Dockerfile.core -t jampacked/core:latest .
    
    # Build worker image
    echo "Building jampacked/pattern-worker..."
    docker build -f deployment/Dockerfile.worker -t jampacked/pattern-worker:latest .
    docker tag jampacked/pattern-worker:latest jampacked/cultural-worker:latest
    
    # Build GPU worker image if NVIDIA Docker is available
    if command -v nvidia-docker &> /dev/null || docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        echo "Building jampacked/multimodal-worker (GPU)..."
        docker build -f deployment/Dockerfile.gpu -t jampacked/multimodal-worker:latest .
    else
        echo -e "${YELLOW}⚠ GPU support not available, using CPU worker${NC}"
        docker tag jampacked/pattern-worker:latest jampacked/multimodal-worker:latest
    fi
    
    echo -e "${GREEN}✓ Images built successfully${NC}"
    echo ""
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"
    
    cd "$PROJECT_ROOT/deployment"
    
    # Create required directories
    mkdir -p "$PROJECT_ROOT/data/jampacked" "$PROJECT_ROOT/data/mcp" "$PROJECT_ROOT/logs"
    
    # Set environment variables
    export MCP_SERVER_PATH="${MCP_SERVER_PATH:-/Users/pulser/Documents/GitHub/mcp-sqlite-server}"
    export GPU_ENABLED="${GPU_ENABLED:-false}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin}"
    
    # Deploy
    docker-compose -f docker-compose.production.yml up -d
    
    echo -e "${GREEN}✓ Docker Compose deployment complete${NC}"
    echo ""
    
    # Show status
    docker-compose -f docker-compose.production.yml ps
    
    echo ""
    echo -e "${GREEN}Access points:${NC}"
    echo "  - API: http://localhost:8080"
    echo "  - MCP SQLite: localhost:3333"
    echo "  - Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
    echo "  - Prometheus: http://localhost:9091"
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    
    cd "$PROJECT_ROOT/deployment/kubernetes"
    
    # Create namespace
    kubectl create namespace jampacked --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f jampacked-deployment.yaml
    
    # Wait for deployments
    echo "Waiting for deployments to be ready..."
    kubectl -n jampacked wait --for=condition=available --timeout=300s deployment/jampacked-core
    kubectl -n jampacked wait --for=condition=available --timeout=300s deployment/pattern-discovery-worker
    
    echo -e "${GREEN}✓ Kubernetes deployment complete${NC}"
    echo ""
    
    # Show status
    kubectl -n jampacked get all
    
    echo ""
    echo -e "${GREEN}Access points:${NC}"
    kubectl -n jampacked get ingress
}

# Function to run health checks
run_health_checks() {
    echo -e "${YELLOW}Running health checks...${NC}"
    
    # Wait for services to start
    sleep 10
    
    # Check core API
    if curl -f http://localhost:8080/health &> /dev/null; then
        echo -e "${GREEN}✓ Core API is healthy${NC}"
    else
        echo -e "${RED}❌ Core API health check failed${NC}"
    fi
    
    # Check Redis
    if docker exec jampacked-redis redis-cli ping | grep -q PONG; then
        echo -e "${GREEN}✓ Redis is healthy${NC}"
    else
        echo -e "${RED}❌ Redis health check failed${NC}"
    fi
    
    echo ""
}

# Function to initialize JamPacked tables
initialize_database() {
    echo -e "${YELLOW}Initializing JamPacked database...${NC}"
    
    cd "$PROJECT_ROOT"
    
    # Run setup script
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        docker run --rm \
            -v "$PROJECT_ROOT:/app" \
            -v "${MCP_SERVER_PATH:-/Users/pulser/Documents/GitHub/mcp-sqlite-server}/data:/data/mcp" \
            --network deployment_jampacked-network \
            jampacked/core:latest \
            python /app/setup_mcp_integration.py
    else
        kubectl -n jampacked run setup-job \
            --image=jampacked/core:latest \
            --rm -it --restart=Never \
            -- python /app/setup_mcp_integration.py
    fi
    
    echo -e "${GREEN}✓ Database initialized${NC}"
    echo ""
}

# Function to show deployment info
show_deployment_info() {
    echo -e "${GREEN}🎉 JamPacked Deployment Complete!${NC}"
    echo "=================================="
    echo ""
    echo "Quick Start:"
    echo "1. Access the API: http://localhost:8080"
    echo "2. View metrics: http://localhost:3000 (Grafana)"
    echo "3. Run analysis:"
    echo ""
    echo "   python -c \""
    echo "   from jampacked_sqlite_integration import analyze_campaign_via_mcp"
    echo "   import asyncio"
    echo "   "
    echo "   materials = {'text': ['Your campaign text'], 'images': []}"
    echo "   context = {'campaign_name': 'Test Campaign', 'target_cultures': ['us']}"
    echo "   "
    echo "   result = asyncio.run(analyze_campaign_via_mcp(materials, context))"
    echo "   print(f'Campaign ID: {result[\"campaign_id\"]}')"
    echo "   \""
    echo ""
    echo "4. Query in Claude Desktop:"
    echo "   SELECT * FROM jampacked_creative_analysis ORDER BY created_at DESC;"
    echo ""
}

# Function to cleanup deployment
cleanup_deployment() {
    echo -e "${YELLOW}Cleaning up deployment...${NC}"
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        cd "$PROJECT_ROOT/deployment"
        docker-compose -f docker-compose.production.yml down -v
    else
        kubectl delete namespace jampacked
    fi
    
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main deployment flow
main() {
    case "$1" in
        "docker"|"compose")
            check_prerequisites
            build_images
            deploy_docker_compose
            run_health_checks
            initialize_database
            show_deployment_info
            ;;
        "kubernetes"|"k8s")
            check_prerequisites
            build_images
            deploy_kubernetes
            initialize_database
            show_deployment_info
            ;;
        "cleanup"|"down")
            cleanup_deployment
            ;;
        *)
            echo "Usage: $0 {docker|kubernetes|cleanup} [environment]"
            echo ""
            echo "Examples:"
            echo "  $0 docker production    # Deploy with Docker Compose"
            echo "  $0 kubernetes staging   # Deploy to Kubernetes"
            echo "  $0 cleanup             # Remove deployment"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"