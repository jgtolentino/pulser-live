#!/bin/bash
# Start YouTube integration daemons

JAMPACKED_DIR="/Users/pulser/Documents/GitHub/jampacked-creative-intelligence"
cd "$JAMPACKED_DIR"

echo "🎬 Starting JamPacked YouTube Integration..."

# Add Python user bin to PATH
export PATH="/Users/pulser/Library/Python/3.9/bin:$PATH"

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
