#!/bin/bash

# JamPacked MCP Integration Stop Script
# Stops all running services

echo "🛑 Stopping JamPacked MCP Integration Services"
echo "==========================================="
echo ""

PROJECT_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
LOGS_DIR="${PROJECT_DIR}/logs"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to stop a service
stop_service() {
    local name=$1
    local pid_file="${LOGS_DIR}/${name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping $name (PID: $pid)..."
            kill $pid
            
            # Wait for process to stop
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${RED}❌ Failed to stop $name gracefully, forcing...${NC}"
                kill -9 $pid
            else
                echo -e "${GREEN}✅ $name stopped${NC}"
            fi
        else
            echo "$name was not running (PID: $pid)"
        fi
        
        rm -f "$pid_file"
    else
        echo "$name PID file not found"
    fi
}

# Stop all services
stop_service "api-server"
stop_service "notification-system"
stop_service "performance-monitor"
stop_service "agent-relay"

echo ""
echo "✅ All services stopped"