#!/bin/bash
# Bruno Simple Executor - Avoids JSON escaping issues

DB="$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"

echo "[Bruno] Starting simple executor..."

while true; do
  # Get pending tasks
  TASK=$(sqlite3 "$DB" "SELECT task_id, payload FROM agent_task_queue WHERE target_agent = 'bruno' AND status IN ('pending', 'approved') LIMIT 1;")
  
  if [ -n "$TASK" ]; then
    ID=$(echo "$TASK" | cut -d'|' -f1)
    PAYLOAD=$(echo "$TASK" | cut -d'|' -f2)
    
    # Mark as in progress
    sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'in_progress' WHERE task_id = '$ID';"
    
    echo "[Bruno] Task $ID started"
    
    # Extract command
    COMMAND=$(echo "$PAYLOAD" | jq -r '.command // empty')
    
    if [ -n "$COMMAND" ]; then
      echo "[Bruno] Running: $COMMAND"
      
      # Execute command
      eval "$COMMAND"
      RESULT=$?
      
      if [ $RESULT -eq 0 ]; then
        sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'completed', result = '{\"success\": true}' WHERE task_id = '$ID';"
        echo "[Bruno] Task $ID completed"
      else
        sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"success\": false}' WHERE task_id = '$ID';"
        echo "[Bruno] Task $ID failed"
      fi
    else
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"error\": \"No command\"}' WHERE task_id = '$ID';"
    fi
  fi
  
  sleep 3
done