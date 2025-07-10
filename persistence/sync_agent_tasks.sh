#!/bin/bash

# Agent Task Sync Script
# Polls and processes pending tasks regularly

DB_PATH="${DB_PATH:-$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite}"
LOG_FILE="$HOME/Library/Logs/agent-task-sync.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to get pending task count
get_pending_count() {
    sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM agent_task_queue WHERE status = 'pending';" 2>/dev/null || echo "0"
}

# Function to get oldest pending task age (in seconds)
get_oldest_task_age() {
    sqlite3 "$DB_PATH" "
        SELECT CAST((julianday('now') - julianday(MIN(created_at))) * 86400 AS INTEGER)
        FROM agent_task_queue 
        WHERE status = 'pending';
    " 2>/dev/null || echo "0"
}

# Function to check agent health
check_agents() {
    sqlite3 "$DB_PATH" "
        SELECT agent_name, 
               CASE 
                   WHEN julianday('now') - julianday(last_heartbeat) > 0.0208 THEN 'offline'
                   ELSE 'online'
               END as status
        FROM agent_registry;
    " 2>/dev/null
}

# Main sync logic
main() {
    log "Starting task sync check"
    
    # Get pending tasks
    PENDING=$(get_pending_count)
    
    if [ "$PENDING" -gt 0 ]; then
        log "Found $PENDING pending tasks"
        
        # Check oldest task age
        AGE=$(get_oldest_task_age)
        
        if [ "$AGE" -gt 300 ]; then  # If oldest task > 5 minutes
            log "WARNING: Oldest task is $AGE seconds old"
            
            # Check if agent relay is running
            if ! pgrep -f "agent_relay.py" > /dev/null; then
                log "Agent relay not running, starting it..."
                cd "$HOME/Documents/GitHub/jampacked-creative-intelligence"
                python3 mcp-integration/agent_relay.py >> "$LOG_FILE" 2>&1 &
                log "Started agent relay (PID: $!)"
            fi
        fi
        
        # Log agent status
        log "Agent status:"
        check_agents | while read line; do
            log "  $line"
        done
    else
        log "No pending tasks"
    fi
    
    # Clean up old completed tasks (older than 7 days)
    CLEANED=$(sqlite3 "$DB_PATH" "
        DELETE FROM agent_task_queue 
        WHERE status IN ('completed', 'failed') 
        AND julianday('now') - julianday(completed_at) > 7;
        SELECT changes();
    " 2>/dev/null || echo "0")
    
    if [ "$CLEANED" -gt 0 ]; then
        log "Cleaned up $CLEANED old tasks"
    fi
}

# Run main function
main

# Optional: Send notification if critical issues
if [ "$PENDING" -gt 50 ]; then
    osascript -e 'display notification "High number of pending agent tasks" with title "JamPacked Alert"'
fi