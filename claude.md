# claude.md - Deployment Verification Rules

## 🚨 CRITICAL: Current Site Status

**The site is currently BROKEN as of June 3, 2025 03:43 PST**

- **Production URL**: https://pulser-live.vercel.app
- **Status**: 🔴 BROKEN - Shows "JavaScript is required"
- **Last Working Commit**: f48c92b (attempted rollback in progress)
- **Root Error**: Array destructuring error in React components prevents mounting

## Definition of "Working Deployment"

ALL of these must be true before claiming success:

1. ✅ `curl https://pulser-live.vercel.app` shows "PULSER" or "TBWA" text
2. ✅ NO "JavaScript is required" message anywhere in response
3. ✅ Page content has >5000 characters
4. ✅ Browser console has 0 JavaScript errors
5. ✅ React root element has mounted children
6. ✅ Site loads in under 3 seconds

## Mandatory Verification Sequence

**NEVER claim deployment success without running these exact commands:**

```bash
# 1. Wait minimum 3 minutes after git push
echo "Waiting 3 minutes for deployment..." && sleep 180

# 2. Check for failure indicators (must return 0)
BROKEN_CHECK=$(curl -s https://pulser-live.vercel.app | grep -c "JavaScript is required")
echo "Broken indicator count: $BROKEN_CHECK (must be 0)"

# 3. Check for success indicators (must be >0)
SUCCESS_CHECK=$(curl -s https://pulser-live.vercel.app | grep -cE "(PULSER|TBWA)")
echo "Success indicator count: $SUCCESS_CHECK (must be >0)"

# 4. Check content size (must be >5000)
CONTENT_SIZE=$(curl -s https://pulser-live.vercel.app | wc -c)
echo "Content size: $CONTENT_SIZE bytes (must be >5000)"

# 5. If ANY check fails, deployment is BROKEN
if [ $BROKEN_CHECK -gt 0 ] || [ $SUCCESS_CHECK -eq 0 ] || [ $CONTENT_SIZE -lt 5000 ]; then
    echo "❌ DEPLOYMENT IS BROKEN - DO NOT CLAIM SUCCESS"
    exit 1
else
    echo "✅ All basic checks passed"
fi
```

## Known Issues to Watch For

### 1. Array Destructuring Error
```tsx
// ❌ WRONG - Causes "Dn is not a function" error
const [ref, isVisible] = useScrollReveal();

// ✅ CORRECT
const { ref, isVisible } = useScrollReveal();
```

### 2. Missing/Broken Components
- Check that all imported components exist
- Verify no circular imports
- Ensure all hooks are properly imported

### 3. Build Cache Issues
- Vercel may serve stale bundles
- Look for bundle filename mismatches in error logs
- Force cache clear if needed

## Emergency Procedures

### If Deployment is Broken:
```bash
# 1. Immediate rollback to last working commit
git reset --hard f48c92b
git push --force origin main

# 2. Wait and verify rollback
sleep 180
curl -s https://pulser-live.vercel.app | grep -E "(JavaScript|PULSER)"

# 3. If still broken, escalate to manual Vercel intervention
```

### If Verification Script Lies:
```bash
# The postdeploy-verify.js script may give false positives
# Always use curl as ground truth:
curl -s https://pulser-live.vercel.app | head -50

# If you see "JavaScript is required", the script is wrong
```

## You MUST NOT:

- ❌ Claim success if you see "JavaScript is required"
- ❌ Trust verification scripts without manual curl confirmation
- ❌ Check deployment in less than 3 minutes after pushing
- ❌ Say "it should be working" without proof
- ❌ Ignore error patterns in build logs
- ❌ Deploy without testing locally first

## You MUST:

- ✅ Run the verification sequence above
- ✅ Wait full 3 minutes after any git push
- ✅ Check multiple times if unsure
- ✅ Use curl output as source of truth
- ✅ Acknowledge when deployment is broken
- ✅ Provide exact error messages and timestamps

## Current Investigation Status

**Emergency rollback executed at 03:40 PST**
- Fixed array destructuring in PulserFeaturesSection
- Waiting for Vercel to deploy fixed version
- Next check due at 03:43+ PST

**If this rollback fails:**
- Check Vercel build logs manually
- Consider removing all Pulser components temporarily
- Revert to pure TBWA version that definitely worked