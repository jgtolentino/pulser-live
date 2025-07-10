#!/bin/bash

# Example Conversation Script
# Demonstrates JamPacked → Marian Trivera → Echo workflow

echo "🧠 JamPacked MCP Integration Example"
echo "===================================="
echo ""

# Database path
DB_PATH="${HOME}/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"

# Function to execute SQL
exec_sql() {
    sqlite3 "$DB_PATH" "$1"
}

# Function to create a task
create_task() {
    local source=$1
    local target=$2
    local task_type=$3
    local payload=$4
    local task_id="task_$(date +%s%N)_$$_$RANDOM"
    
    exec_sql "INSERT INTO agent_task_queue (task_id, source_agent, target_agent, task_type, payload, priority) 
              VALUES ('$task_id', '$source', '$target', '$task_type', '$payload', 5);"
    
    echo "$task_id"
}

# Function to check task status
check_task() {
    local task_id=$1
    exec_sql "SELECT status, result FROM agent_task_queue WHERE task_id = '$task_id';"
}

# Function to show queue statistics
show_stats() {
    echo ""
    echo "📊 Current Queue Statistics:"
    exec_sql "SELECT * FROM task_statistics;"
}

# Step 1: JamPacked analyzes a campaign
echo "1️⃣ Creating JamPacked analysis task..."
JAMPACKED_PAYLOAD='{
    "campaign_name": "EcoFuture 2025",
    "client": "GreenTech Corp",
    "brand": "EcoSmart",
    "target_cultures": ["US", "EU", "APAC"],
    "business_objectives": ["awareness", "conversion"],
    "csr_focus": "carbon_neutrality"
}'

TASK1=$(create_task "User" "JamPacked" "campaign_analysis" "$JAMPACKED_PAYLOAD")
echo "   Created task: $TASK1"

# Step 2: Marian performs market research based on JamPacked results
echo ""
echo "2️⃣ Creating Marian Trivera research task..."
MARIAN_PAYLOAD='{
    "research_type": "market_analysis",
    "campaign_id": "ecofuture_2025",
    "focus_areas": ["sustainability_trends", "competitor_campaigns"],
    "regions": ["US", "EU", "APAC"]
}'

TASK2=$(create_task "JamPacked" "Marian Trivera" "market_research" "$MARIAN_PAYLOAD")
echo "   Created task: $TASK2"

# Step 3: Echo generates communication based on insights
echo ""
echo "3️⃣ Creating Echo communication task..."
ECHO_PAYLOAD='{
    "message_type": "campaign_summary",
    "tone": "professional",
    "audience": "c_suite",
    "key_points": ["effectiveness_score", "award_potential", "market_insights"]
}'

TASK3=$(create_task "Marian Trivera" "Echo" "generate_summary" "$ECHO_PAYLOAD")
echo "   Created task: $TASK3"

# Show initial queue state
echo ""
echo "📋 Initial Queue State:"
exec_sql "SELECT task_id, source_agent, target_agent, status FROM active_tasks;"

# Show statistics
show_stats

echo ""
echo "✅ Example tasks created successfully!"
echo ""
echo "To process these tasks, run:"
echo "  python mcp-integration/agent_relay.py"
echo ""
echo "To monitor progress:"
echo "  watch -n 1 'sqlite3 $DB_PATH \"SELECT * FROM active_tasks;\"'"
echo ""
echo "To view results:"
echo "  sqlite3 $DB_PATH \"SELECT task_id, status, substr(result, 1, 50) as result_preview FROM agent_task_queue;\""