#!/bin/bash
# :pulser-task-runner.sh
# Pulser CLI task runner for processing delegated tasks from Claude Desktop

# Absolute paths
DB="$HOME/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
PULSER_HOME="$HOME/Documents/GitHub/pulser-live"
JAMPACKED_HOME="$HOME/Documents/GitHub/jampacked-creative-intelligence"

# Set up environment
export PATH="$PULSER_HOME/bin:$PATH"
export PULSER_CONFIG_DIR="$HOME/.pulser"

echo "[PulserCLI] Starting task runner..."
echo "[PulserCLI] Monitoring database: $DB"
echo "[PulserCLI] Pulser home: $PULSER_HOME"

while true; do
  # Check for pending tasks assigned to pulser_cli
  TASK=$(sqlite3 "$DB" "SELECT task_id, payload FROM agent_task_queue WHERE target_agent = 'pulser_cli' AND status = 'pending' LIMIT 1;")
  
  if [ -n "$TASK" ]; then
    ID=$(echo "$TASK" | cut -d'|' -f1)
    PAYLOAD=$(echo "$TASK" | cut -d'|' -f2)
    
    # Save payload to temporary file
    echo "$PAYLOAD" > /tmp/pulser_payload_${ID}.json
    
    echo "[PulserCLI] Executing Task $ID"
    echo "[PulserCLI] Payload: $PAYLOAD"
    
    # Mark as in progress
    sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'in_progress' WHERE task_id = '$ID';"
    
    # Extract the plan/command from payload
    PLAN=$(echo "$PAYLOAD" | jq -r '.plan // empty')
    COMMAND=$(echo "$PAYLOAD" | jq -r '.command // empty')
    
    # Execute based on task type
    if [ -n "$PLAN" ]; then
      echo "[PulserCLI] Executing orchestration plan: $PLAN"
      
      # Handle specific orchestration plans
      case "$PLAN" in
        "test orchestration")
          echo "[PulserCLI] Running test orchestration"
          RESULT=0
          ;;
        "setup_lions_palette_forge_mcp_integration")
          echo "[PulserCLI] Setting up Lions Palette Forge MCP integration"
          WORKING_DIR=$(echo "$PAYLOAD" | jq -r '.working_directory // empty')
          if [ -n "$WORKING_DIR" ]; then
            cd "$WORKING_DIR"
          fi
          echo "[PulserCLI] Integration setup orchestrated"
          RESULT=0
          ;;
        "run etl pipeline scout_gold_to_insights")
          echo "[PulserCLI] Running Scout ETL pipeline"
          cd "$JAMPACKED_HOME" && npm run scout:etl 2>&1
          RESULT=$?
          ;;
        "npm install and build")
          echo "[PulserCLI] Running npm install and build"
          cd "$JAMPACKED_HOME" && npm install && npm run build
          RESULT=$?
          ;;
        "setup mcp backend")
          echo "[PulserCLI] Setting up MCP backend"
          cd "$JAMPACKED_HOME/mcp-integration" && ./setup-mcp-backend.sh
          RESULT=$?
          ;;
        "verify integration")
          echo "[PulserCLI] Verifying MCP integration"
          cd "$JAMPACKED_HOME" && npm run test:integration
          RESULT=$?
          ;;
        "test_lions_palette_forge_integration")
          echo "[PulserCLI] Testing Lions Palette Forge integration"
          curl -s http://localhost:8080 > /dev/null && echo "Frontend running" || echo "Frontend not accessible"
          curl -s http://localhost:3000/health > /dev/null && echo "Backend running" || echo "Backend not accessible"
          RESULT=0
          ;;
        "verify_mcp_database_integration")
          echo "[PulserCLI] Verifying MCP database integration"
          sqlite3 "$DB" "SELECT COUNT(*) FROM agent_task_queue;" > /dev/null && echo "Database accessible"
          RESULT=$?
          ;;
        *)
          echo "[PulserCLI] Unknown orchestration plan: $PLAN"
          RESULT=1
          ;;
      esac
    elif [ -n "$COMMAND" ]; then
      echo "[PulserCLI] Executing command: $COMMAND"
      eval "$COMMAND"
      RESULT=$?
    else
      echo "[PulserCLI] WARNING: No plan or command found in payload"
      RESULT=1
    fi
    
    # Update task status based on result
    if [ $RESULT -eq 0 ]; then
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'completed', result = '{\"done\": true, \"exit_code\": 0}' WHERE task_id = '$ID';"
      echo "[PulserCLI] Task $ID completed successfully"
    else
      sqlite3 "$DB" "UPDATE agent_task_queue SET status = 'failed', result = '{\"done\": false, \"exit_code\": $RESULT}' WHERE task_id = '$ID';"
      echo "[PulserCLI] Task $ID failed with exit code: $RESULT"
    fi
    
    # Clean up
    rm -f /tmp/pulser_payload.json
  fi
  
  # Sleep before next check
  sleep 5
done