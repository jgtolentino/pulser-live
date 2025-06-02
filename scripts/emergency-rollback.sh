#!/bin/bash
# scripts/emergency-rollback.sh

echo "🚨 Emergency Rollback Procedure"
echo "=============================="
echo "Timestamp: $(date)"
echo ""

# Get current status
echo "🔍 Checking current deployment status..."
CURRENT_STATUS=$(curl -s https://pulser-live.vercel.app | grep -c "JavaScript is required")
CONTENT_SIZE=$(curl -s https://pulser-live.vercel.app | wc -c)
SUCCESS_INDICATORS=$(curl -s https://pulser-live.vercel.app | grep -cE "(PULSER|TBWA)" || echo "0")

echo "Current status:"
echo "  - 'JavaScript is required' count: $CURRENT_STATUS (should be 0)"
echo "  - Content size: $CONTENT_SIZE bytes (should be >5000)"
echo "  - Success indicators: $SUCCESS_INDICATORS (should be >0)"
echo ""

if [ $CURRENT_STATUS -gt 0 ] || [ $CONTENT_SIZE -lt 5000 ] || [ $SUCCESS_INDICATORS -eq 0 ]; then
    echo "❌ SITE IS CONFIRMED BROKEN"
    echo ""
    echo "Sample content from deployed site:"
    echo "-----------------------------------"
    curl -s https://pulser-live.vercel.app | head -20
    echo "-----------------------------------"
    echo ""
    
    echo "🔄 Proceeding with emergency procedures..."
    echo ""
    
    # Show recent commits
    echo "📋 Recent commits:"
    git log --oneline -5
    echo ""
    
    echo "⚠️  Manual intervention required:"
    echo "1. Check Vercel dashboard at https://vercel.com/jgtolentino/pulser-live"
    echo "2. Look for build errors or deployment failures"
    echo "3. Consider clearing Vercel cache manually"
    echo "4. If needed, revert to commit before Pulser transformation:"
    echo "   git reset --hard ff4f512"
    echo "   git push --force origin main"
    echo ""
    
    exit 1
else
    echo "✅ Site appears to be working correctly"
    echo ""
    echo "Verification details:"
    echo "  - No 'JavaScript is required' message found"
    echo "  - Content size is adequate ($CONTENT_SIZE bytes)"
    echo "  - Found expected content markers"
    echo ""
    
    echo "Sample working content:"
    echo "----------------------"
    curl -s https://pulser-live.vercel.app | grep -E "(PULSER|TBWA)" | head -5
    echo "----------------------"
fi