#!/bin/bash
# Stop YouTube integration daemons

JAMPACKED_DIR="/Users/pulser/Documents/GitHub/jampacked-creative-intelligence"
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
