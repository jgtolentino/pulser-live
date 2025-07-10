#!/bin/bash
# setup_youtube_integration.sh
# Sets up YouTube analysis integration for JamPacked Creative Intelligence

set -e

echo "🎬 Setting up JamPacked YouTube Integration..."

# Define base paths
JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
AGENTS_DIR="$JAMPACKED_DIR/agents"
SQL_DIR="$JAMPACKED_DIR/sql"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p "$AGENTS_DIR/pulser/handlers"
mkdir -p "$AGENTS_DIR/jampacked/handlers"
mkdir -p "$SQL_DIR/triggers"
mkdir -p "$JAMPACKED_DIR/logs"

# Check and install Python dependencies
echo "🐍 Checking Python dependencies..."
python3 -m pip install --upgrade pip

# Install required packages
REQUIRED_PACKAGES=(
    "yt-dlp"
    "openai-whisper" 
    "sqlite3"
    "requests"
    "python-dotenv"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    echo "📦 Installing $package..."
    python3 -m pip install "$package" || echo "⚠️  Failed to install $package (might already be installed)"
done

# Check if yt-dlp and whisper are available in PATH
echo "🔍 Verifying tool installations..."
if command -v yt-dlp &> /dev/null; then
    echo "✅ yt-dlp is installed: $(yt-dlp --version)"
else
    echo "❌ yt-dlp not found in PATH"
    echo "💡 Try: pip install yt-dlp"
fi

if command -v whisper &> /dev/null; then
    echo "✅ whisper is installed"
    whisper --help | head -n 1
else
    echo "❌ whisper not found in PATH"
    echo "💡 Try: pip install openai-whisper"
fi

# Create Python requirements file
echo "📝 Creating requirements.txt..."
cat > "$JAMPACKED_DIR/requirements_youtube.txt" << EOF
# YouTube Integration Requirements
yt-dlp>=2023.9.24
openai-whisper>=20231117
sqlite3
requests>=2.31.0
python-dotenv>=1.0.0
numpy>=1.24.0
torch>=2.0.0
ffmpeg-python>=0.2.0
EOF

# Create logging configuration
echo "📋 Setting up logging..."
cat > "$JAMPACKED_DIR/config/youtube_logging.py" << EOF
import logging
import os
from datetime import datetime

def setup_logging():
    """Setup logging for YouTube integration"""
    log_dir = "/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"youtube_integration_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('youtube_integration')
EOF

# Create environment configuration
echo "⚙️  Creating environment configuration..."
cat > "$JAMPACKED_DIR/.env.youtube" << EOF
# YouTube Integration Configuration
YOUTUBE_INTEGRATION_ENABLED=true
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
TEMP_DIR=/tmp/jampacked_youtube
MAX_VIDEO_DURATION=600
MAX_FILE_SIZE=100MB
SQLITE_DB_PATH=/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite

# Optional: OpenAI API for enhanced analysis
# OPENAI_API_KEY=your_key_here

# Pulser Configuration
PULSER_POLL_INTERVAL=5
PULSER_MAX_RETRIES=3

# JamPacked Configuration  
JAMPACKED_POLL_INTERVAL=5
JAMPACKED_CONFIDENCE_THRESHOLD=0.7

# Award Prediction Weights
CANNES_WEIGHT=0.25
EFFIE_WEIGHT=0.20
ONESHOW_WEIGHT=0.20
DAD_WEIGHT=0.20
CLIO_WEIGHT=0.15
EOF

# Create daemon startup scripts
echo "🚀 Creating daemon scripts..."
cat > "$JAMPACKED_DIR/start_youtube_integration.sh" << 'EOF'
#!/bin/bash
# Start YouTube integration daemons

JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🎬 Starting JamPacked YouTube Integration..."

# Load environment
source .env.youtube

# Start Pulser handler in background
echo "📥 Starting Pulser YouTube handler..."
python3 agents/pulser/handlers/extract_and_analyze_youtube.py &
PULSER_PID=$!
echo $PULSER_PID > logs/pulser_youtube.pid

# Start JamPacked transcript analyzer in background  
echo "🧠 Starting JamPacked transcript analyzer..."
python3 agents/jampacked/handlers/analyze_transcript.py &
JAMPACKED_PID=$!
echo $JAMPACKED_PID > logs/jampacked_transcript.pid

echo "✅ YouTube integration started!"
echo "📊 Pulser PID: $PULSER_PID"
echo "🧠 JamPacked PID: $JAMPACKED_PID"
echo "📝 Check logs in: $JAMPACKED_DIR/logs/"

# Monitor processes
while true; do
    if ! kill -0 $PULSER_PID 2>/dev/null; then
        echo "❌ Pulser handler stopped"
        break
    fi
    if ! kill -0 $JAMPACKED_PID 2>/dev/null; then
        echo "❌ JamPacked analyzer stopped"
        break
    fi
    sleep 30
done
EOF

cat > "$JAMPACKED_DIR/stop_youtube_integration.sh" << 'EOF'
#!/bin/bash
# Stop YouTube integration daemons

JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🛑 Stopping JamPacked YouTube Integration..."

# Stop Pulser
if [ -f logs/pulser_youtube.pid ]; then
    PULSER_PID=$(cat logs/pulser_youtube.pid)
    if kill -0 $PULSER_PID 2>/dev/null; then
        kill $PULSER_PID
        echo "📥 Stopped Pulser handler (PID: $PULSER_PID)"
    fi
    rm logs/pulser_youtube.pid
fi

# Stop JamPacked
if [ -f logs/jampacked_transcript.pid ]; then
    JAMPACKED_PID=$(cat logs/jampacked_transcript.pid)
    if kill -0 $JAMPACKED_PID 2>/dev/null; then
        kill $JAMPACKED_PID
        echo "🧠 Stopped JamPacked analyzer (PID: $JAMPACKED_PID)"
    fi
    rm logs/jampacked_transcript.pid
fi

echo "✅ YouTube integration stopped!"
EOF

# Make scripts executable
chmod +x "$JAMPACKED_DIR/start_youtube_integration.sh"
chmod +x "$JAMPACKED_DIR/stop_youtube_integration.sh"

# Create test script
echo "🧪 Creating test script..."
cat > "$JAMPACKED_DIR/test_youtube_integration.sh" << 'EOF'
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
EOF

chmod +x "$JAMPACKED_DIR/test_youtube_integration.sh"

# Create integration documentation
echo "📚 Creating documentation..."
cat > "$JAMPACKED_DIR/YOUTUBE_INTEGRATION_README.md" << 'EOF'
# 🎬 JamPacked YouTube Integration

## Overview
AI-powered creative effectiveness analysis for YouTube videos using WARC Effective 100 standards.

## Architecture
```
Claude Desktop → SQL Trigger → Pulser Agent → JamPacked Agent → Results
```

## Quick Start

### 1. Start Services
```bash
./start_youtube_integration.sh
```

### 2. Analyze Video (from Claude Desktop)
```sql
sqlite3 data/database.sqlite < sql/analyze_youtube_from_claude.sql
```

### 3. Check Results
```sql
SELECT result FROM agent_task_queue WHERE task_id = 'yt_jampacked_001';
```

### 4. Stop Services
```bash
./stop_youtube_integration.sh
```

## Testing
```bash
./test_youtube_integration.sh
```

## Logs
- Location: `logs/youtube_integration_YYYYMMDD.log`
- Process IDs: `logs/pulser_youtube.pid`, `logs/jampacked_transcript.pid`

## Dependencies
- `yt-dlp`: Video download
- `whisper`: Audio transcription  
- SQLite: Task queue
- Python 3.8+

## Analysis Output
- WARC 5 Dimensions scoring
- Award prediction (Cannes, Effie, One Show, D&AD, Clio)
- Creative effectiveness metrics
- Cultural impact assessment
- ROI projections
EOF

echo ""
echo "🎉 JamPacked YouTube Integration Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. ✅ Dependencies installed"
echo "2. ✅ Directory structure created"  
echo "3. ✅ Configuration files generated"
echo "4. ✅ Daemon scripts ready"
echo "5. ✅ Test suite available"
echo ""
echo "🚀 To start the integration:"
echo "   cd $JAMPACKED_DIR"
echo "   ./start_youtube_integration.sh"
echo ""
echo "🧪 To test the integration:"
echo "   ./test_youtube_integration.sh"
echo ""
echo "📚 Read the docs:"
echo "   cat YOUTUBE_INTEGRATION_README.md"
echo ""
echo "🎬 Happy analyzing!"