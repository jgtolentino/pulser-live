#!/bin/bash

# JamPacked Services Startup Script
# Ensures all services are running

JAMPACKED_HOME="${JAMPACKED_HOME:-/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence}"
LOG_DIR="$HOME/Library/Logs"

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Function to start service if not running
start_if_needed() {
    local name=$1
    local cmd=$2
    local log_file="$LOG_DIR/jampacked-${name}.log"
    
    if ! is_running "$cmd"; then
        echo "[$(date)] Starting $name..." >> "$LOG_DIR/jampacked-services.log"
        cd "$JAMPACKED_HOME"
        nohup $cmd >> "$log_file" 2>&1 &
        echo "[$(date)] Started $name (PID: $!)" >> "$LOG_DIR/jampacked-services.log"
    fi
}

# Ensure Node.js is in PATH
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# Start services
start_if_needed "api-server" "node api/jampacked-api-server.js"
start_if_needed "notifications" "node mcp-integration/realtime-notifications.js"
start_if_needed "monitoring" "node mcp-integration/performance-monitoring.js"
start_if_needed "agent-relay" "python3 mcp-integration/agent_relay.py"

# Health check
sleep 5
if curl -s http://localhost:3001/health > /dev/null; then
    echo "[$(date)] Health check passed" >> "$LOG_DIR/jampacked-services.log"
else
    echo "[$(date)] Health check failed" >> "$LOG_DIR/jampacked-services.log"
fi