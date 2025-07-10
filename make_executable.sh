#!/bin/bash
# make_executable.sh - Make all scripts executable

echo "🔧 Making scripts executable..."

chmod +x /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/setup_youtube_integration.sh
chmod +x /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/start_youtube_integration.sh
chmod +x /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/stop_youtube_integration.sh
chmod +x /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence/test_youtube_integration.sh

echo "✅ All scripts are now executable!"
echo ""
echo "🚀 Next steps:"
echo "1. Run: ./setup_youtube_integration.sh"
echo "2. Then: ./start_youtube_integration.sh"
echo "3. Test: ./test_youtube_integration.sh"