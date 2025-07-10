#!/bin/bash
# Test YouTube integration

JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🧪 Testing JamPacked YouTube Integration..."

# Test video URL (short, safe video)
TEST_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TASK_ID="test_$(date +%s)"

echo "📝 Creating test task: $TASK_ID"

# Insert test task
sqlite3 "$SQLITE_DB_PATH" << SQL
INSERT INTO agent_task_queue (
    task_id,
    source_agent,
    target_agent,
    task_type,
    payload,
    status,
    created_at
) VALUES (
    '$TASK_ID',
    'test_script',
    'pulser',
    'analyze_youtube',
    json('{
        "video_url": "$TEST_URL",
        "analysis_focus": "creative_effectiveness",
        "test_mode": true
    }'),
    'pending',
    datetime('now')
);
SQL

echo "✅ Test task created: $TASK_ID"
echo "⏳ Waiting for processing..."

# Monitor task progress
for i in {1..60}; do
    STATUS=$(sqlite3 "$SQLITE_DB_PATH" "SELECT status FROM agent_task_queue WHERE task_id='$TASK_ID'")
    echo "📊 Step $i: Status = $STATUS"
    
    if [ "$STATUS" = "completed" ]; then
        echo "🎉 Test completed successfully!"
        sqlite3 "$SQLITE_DB_PATH" "SELECT result FROM agent_task_queue WHERE task_id='$TASK_ID'"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "❌ Test failed!"
        sqlite3 "$SQLITE_DB_PATH" "SELECT result FROM agent_task_queue WHERE task_id='$TASK_ID'"
        break
    fi
    
    sleep 5
done
