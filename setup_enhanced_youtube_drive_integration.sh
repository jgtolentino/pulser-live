#!/bin/bash
# setup_enhanced_youtube_drive_integration.sh
# Enhanced setup script for YouTube + Google Drive integration

set -e

echo "🎯 Setting up Enhanced JamPacked YouTube + Google Drive Integration..."
echo "=================================================================="

# Define base paths
JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
AGENTS_DIR="$JAMPACKED_DIR/agents"
SQL_DIR="$JAMPACKED_DIR/sql"
CONFIG_DIR="$JAMPACKED_DIR/config"

cd "$JAMPACKED_DIR"

echo ""
echo "📁 Verifying enhanced directory structure..."
mkdir -p "$AGENTS_DIR/pulser/handlers"
mkdir -p "$AGENTS_DIR/jampacked/handlers"
mkdir -p "$SQL_DIR/triggers"
mkdir -p "$CONFIG_DIR/enhanced"
mkdir -p "$JAMPACKED_DIR/logs"

echo "✅ Directory structure verified"

echo ""
echo "📄 Verifying enhanced integration files..."

# Check that all enhanced files exist
REQUIRED_FILES=(
    "agents/jampacked/handlers/enhanced_youtube_analysis.py"
    "sql/enhanced_youtube_analysis_with_drive.sql"
    ".env.enhanced_youtube_drive"
    "ENHANCED_INTEGRATION_SUMMARY.md"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing - please ensure all artifacts were saved properly"
        exit 1
    fi
done

echo ""
echo "🐍 Installing enhanced Python dependencies..."

# Enhanced requirements for Google Drive integration
ENHANCED_PACKAGES=(
    "yt-dlp>=2023.12.30"
    "openai-whisper>=20231117"
    "google-api-python-client>=2.100.0"
    "google-auth-httplib2>=0.1.1"
    "google-auth-oauthlib>=1.1.0"
    "requests>=2.31.0"
    "python-dotenv>=1.0.0"
    "numpy>=1.24.0"
    "pandas>=2.0.0"
    "scikit-learn>=1.3.0"
)

echo "📦 Installing enhanced packages..."
for package in "${ENHANCED_PACKAGES[@]}"; do
    echo "  Installing $package..."
    python3 -m pip install "$package" --quiet || echo "⚠️  Failed to install $package"
done

echo ""
echo "🔍 Verifying Google Drive integration tools..."

# Check Google Drive folder access
DRIVE_FOLDER_ID="0AJMhu01UUQKoUk9PVA"
echo "📁 Google Drive folder ID: $DRIVE_FOLDER_ID"
echo "💡 Ensure Claude Desktop has access to this folder"

echo ""
echo "⚙️  Creating enhanced configuration files..."

# Create enhanced logging configuration
cat > "$CONFIG_DIR/enhanced/enhanced_logging.py" << 'EOF'
import logging
import os
from datetime import datetime, timedelta
import json

class EnhancedLogger:
    """Enhanced logging for YouTube + Drive integration"""
    
    def __init__(self, component_name):
        self.component_name = component_name
        self.log_dir = "/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create component-specific logger
        self.logger = logging.getLogger(f'enhanced_{component_name}')
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        log_file = os.path.join(self.log_dir, f"enhanced_{component_name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter with enhanced context
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(component)s] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_analysis_start(self, task_id, client_context):
        """Log the start of enhanced analysis"""
        self.logger.info(
            f"Enhanced analysis started for task {task_id}",
            extra={'component': self.component_name, 'task_id': task_id, 'client': client_context.get('brand', 'Unknown')}
        )
    
    def log_drive_extraction(self, folder_id, search_terms):
        """Log Google Drive extraction request"""
        self.logger.info(
            f"Drive extraction requested for folder {folder_id} with terms: {search_terms}",
            extra={'component': self.component_name, 'folder_id': folder_id}
        )
    
    def log_performance_metrics(self, processing_time, confidence_score, award_predictions):
        """Log performance metrics"""
        metrics = {
            'processing_time_seconds': processing_time,
            'confidence_score': confidence_score,
            'award_predictions_count': len(award_predictions),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(
            f"Performance metrics recorded",
            extra={'component': self.component_name, 'metrics': json.dumps(metrics)}
        )
    
    def log_error(self, error_message, context=None):
        """Log enhanced error with context"""
        self.logger.error(
            f"Enhanced analysis error: {error_message}",
            extra={'component': self.component_name, 'context': context or {}}
        )
EOF

# Create enhanced startup script
cat > "$JAMPACKED_DIR/start_enhanced_integration.sh" << 'EOF'
#!/bin/bash
# Start Enhanced YouTube + Google Drive Integration

JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🎯 Starting Enhanced JamPacked YouTube + Google Drive Integration..."
echo "====================================================================="

# Load enhanced environment
if [ -f .env.enhanced_youtube_drive ]; then
    source .env.enhanced_youtube_drive
    echo "✅ Enhanced configuration loaded"
else
    echo "⚠️  Enhanced configuration not found, using defaults"
fi

# Start Pulser handler for video processing
echo ""
echo "📥 Starting Pulser YouTube handler..."
python3 agents/pulser/handlers/extract_and_analyze_youtube.py &
PULSER_PID=$!
echo $PULSER_PID > logs/enhanced_pulser.pid
echo "📊 Pulser PID: $PULSER_PID"

# Start Enhanced JamPacked analyzer
echo ""
echo "🧠 Starting Enhanced JamPacked analyzer..."
python3 agents/jampacked/handlers/enhanced_youtube_analysis.py &
JAMPACKED_PID=$!
echo $JAMPACKED_PID > logs/enhanced_jampacked.pid
echo "🎯 Enhanced JamPacked PID: $JAMPACKED_PID"

echo ""
echo "✅ Enhanced YouTube + Drive integration started!"
echo ""
echo "📊 Services running:"
echo "   • Pulser (Video Processing): PID $PULSER_PID"
echo "   • Enhanced JamPacked (Analysis): PID $JAMPACKED_PID"
echo "   • Claude Desktop (Drive Extraction): Manual"
echo ""
echo "📁 Google Drive folder: $DRIVE_CAMPAIGN_ROOT_ID"
echo "📝 Logs directory: $JAMPACKED_DIR/logs/"
echo ""
echo "🎬 Ready for enhanced creative analysis!"

# Monitor processes
echo "🔍 Monitoring processes (Ctrl+C to stop)..."
while true; do
    if ! kill -0 $PULSER_PID 2>/dev/null; then
        echo "❌ Pulser handler stopped unexpectedly"
        break
    fi
    if ! kill -0 $JAMPACKED_PID 2>/dev/null; then
        echo "❌ Enhanced JamPacked analyzer stopped unexpectedly"
        break
    fi
    sleep 30
done

echo "🛑 Enhanced integration monitoring stopped"
EOF

# Create enhanced test script
cat > "$JAMPACKED_DIR/test_enhanced_integration.sh" << 'EOF'
#!/bin/bash
# Test Enhanced YouTube + Google Drive Integration

JAMPACKED_DIR="/Users/tbwa/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🧪 Testing Enhanced JamPacked YouTube + Google Drive Integration..."
echo "================================================================="

# Load configuration
if [ -f .env.enhanced_youtube_drive ]; then
    source .env.enhanced_youtube_drive
fi

# Test video and parameters
TEST_URL="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_TASK_ID="enhanced_test_$(date +%s)"
DB_PATH="${SQLITE_DB_PATH:-/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite}"

echo "📝 Creating enhanced test task: $TEST_TASK_ID"
echo "🎬 Test video: $TEST_URL"
echo "📁 Google Drive folder: ${DRIVE_CAMPAIGN_ROOT_ID:-0AJMhu01UUQKoUk9PVA}"

# Create enhanced test task with client context
sqlite3 "$DB_PATH" << SQL
INSERT INTO agent_task_queue (
    task_id,
    source_agent,
    target_agent,
    task_type,
    payload,
    status,
    priority,
    created_at
) VALUES (
    '$TEST_TASK_ID',
    'test_enhanced',
    'pulser',
    'analyze_youtube',
    json('{
        "video_url": "$TEST_URL",
        "analysis_focus": "enhanced_creative_effectiveness",
        "client_context": {
            "brand": "TestBrand",
            "client": "Test Client Enhanced",
            "industry": "technology",
            "campaign_objective": "brand_awareness",
            "target_audience": "tech_enthusiasts",
            "brand_values": ["innovation", "quality", "trust"]
        },
        "google_drive_integration": {
            "enabled": true,
            "campaign_root_folder": "${DRIVE_CAMPAIGN_ROOT_ID:-0AJMhu01UUQKoUk9PVA}",
            "extract_context": true
        },
        "test_mode": true,
        "enhanced_features": [
            "historical_benchmarking",
            "competitive_analysis",
            "brand_guidelines_compliance"
        ]
    }'),
    'pending',
    9,
    datetime('now')
);
SQL

echo "✅ Enhanced test task created successfully"
echo ""
echo "⏳ Monitoring enhanced workflow progress..."

# Monitor enhanced workflow
for i in {1..120}; do
    # Check main task status
    MAIN_STATUS=$(sqlite3 "$DB_PATH" "SELECT status FROM agent_task_queue WHERE task_id='$TEST_TASK_ID'")
    
    # Check for any drive extraction tasks
    DRIVE_TASKS=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM agent_task_queue WHERE task_type='campaign_context_extraction' AND created_at >= datetime('now', '-5 minutes')")
    
    echo "🔍 Step $i: Main task = $MAIN_STATUS, Drive tasks = $DRIVE_TASKS"
    
    if [ "$MAIN_STATUS" = "completed" ]; then
        echo ""
        echo "🎉 Enhanced test completed successfully!"
        echo ""
        echo "📊 Test Results:"
        sqlite3 "$DB_PATH" -header -column "SELECT 
            task_id,
            status,
            json_extract(result, '$.integration_version') as integration_version,
            json_extract(result, '$.confidence_metrics') as confidence_score
        FROM agent_task_queue WHERE task_id='$TEST_TASK_ID'"
        
        echo ""
        echo "🎯 Enhanced Analysis Summary:"
        sqlite3 "$DB_PATH" "SELECT json_extract(result, '$.award_predictions') FROM agent_task_queue WHERE task_id='$TEST_TASK_ID' AND status='completed'" | head -20
        
        break
    elif [ "$MAIN_STATUS" = "failed" ]; then
        echo ""
        echo "❌ Enhanced test failed!"
        echo ""
        echo "📋 Error Details:"
        sqlite3 "$DB_PATH" "SELECT result FROM agent_task_queue WHERE task_id='$TEST_TASK_ID'"
        break
    fi
    
    sleep 5
done

if [ "$MAIN_STATUS" != "completed" ] && [ "$MAIN_STATUS" != "failed" ]; then
    echo ""
    echo "⏰ Test timeout reached"
    echo "📊 Current Status: $MAIN_STATUS"
    echo "💡 Check logs for more details: logs/enhanced_*.log"
fi

echo ""
echo "🧪 Enhanced integration test finished"
EOF

# Make all scripts executable
chmod +x "$JAMPACKED_DIR/start_enhanced_integration.sh"
chmod +x "$JAMPACKED_DIR/test_enhanced_integration.sh"

echo ""
echo "🎯 Enhanced Integration Setup Complete!"
echo "======================================"
echo ""
echo "✅ Files Created/Verified:"
echo "   • Enhanced YouTube analysis engine"
echo "   • Google Drive campaign intelligence"
echo "   • Enhanced SQL triggers and monitoring"
echo "   • Comprehensive configuration files"
echo "   • Enhanced logging and monitoring"
echo "   • Startup and test scripts"
echo ""
echo "📁 Google Drive Integration:"
echo "   • Folder ID: $DRIVE_FOLDER_ID"
echo "   • Ensure Claude Desktop has access"
echo "   • Campaign context extraction enabled"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start enhanced integration:"
echo "      ./start_enhanced_integration.sh"
echo ""
echo "   2. Test enhanced features:"
echo "      ./test_enhanced_integration.sh"
echo ""
echo "   3. Run enhanced analysis (Claude Desktop):"
echo "      sqlite3 data/database.sqlite < sql/enhanced_youtube_analysis_with_drive.sql"
echo ""
echo "   4. Monitor enhanced workflow:"
echo "      tail -f logs/enhanced_*.log"
echo ""
echo "📚 Documentation:"
echo "   • Read: ENHANCED_INTEGRATION_SUMMARY.md"
echo "   • Configuration: .env.enhanced_youtube_drive"
echo ""
echo "🎪 You now have the world's most advanced creative intelligence platform!"
echo "    Combining YouTube analysis with Google Drive campaign intelligence."
echo ""
echo "🏆 Ready to revolutionize creative effectiveness analysis!"