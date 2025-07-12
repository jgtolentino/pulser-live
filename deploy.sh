#!/bin/bash

# JamPacked MCP Integration Deployment Script
# Starts all required services

echo "🚀 JamPacked MCP Integration Deployment"
echo "======================================"
echo ""

# Set paths
PROJECT_DIR="/Users/pulser/Documents/GitHub/jampacked-creative-intelligence"
MCP_DIR="$PROJECT_DIR/mcp-integration"
API_DIR="$PROJECT_DIR/api"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

# Function to start a service
start_service() {
    local name=$1
    local cmd=$2
    local log_file="${PROJECT_DIR}/logs/${name}.log"
    
    echo "Starting $name..."
    mkdir -p "${PROJECT_DIR}/logs"
    
    # Start in background and redirect output to log
    nohup $cmd > "$log_file" 2>&1 &
    local pid=$!
    
    # Save PID for later
    echo $pid > "${PROJECT_DIR}/logs/${name}.pid"
    
    sleep 2
    
    # Check if process is still running
    if ps -p $pid > /dev/null; then
        echo -e "${GREEN}✅ $name started (PID: $pid)${NC}"
        echo "   Log: $log_file"
    else
        echo -e "${RED}❌ Failed to start $name${NC}"
        echo "   Check log: $log_file"
        return 1
    fi
}

# Check required ports
echo "Checking ports..."
check_port 3001 || exit 1  # API Server
check_port 8765 || exit 1  # WebSocket Server

echo ""
echo "Starting services..."
echo ""

# Start API server
cd "$API_DIR"
start_service "api-server" "node jampacked-api-server.js"

# Start notification system
cd "$MCP_DIR"
start_service "notification-system" "node realtime-notifications.js"

# Start performance monitoring
start_service "performance-monitor" "node performance-monitoring.js"

# Start agent relay
start_service "agent-relay" "python3 agent_relay.py"

echo ""
echo "✅ All services started!"
echo ""
echo "Service Status:"
echo "==============="
echo "API Server:          http://localhost:3001/health"
echo "WebSocket Server:    ws://localhost:8765"
echo "Agent Relay:         Running"
echo "Performance Monitor: Running"
echo ""
echo "To stop all services, run: ./stop.sh"
echo "To view logs, check: ${PROJECT_DIR}/logs/"
echo ""
echo "To test the integration, run:"
echo "  ./mcp-integration/example_conversation.sh"