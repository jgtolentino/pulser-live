-- sql/analyze_youtube_from_claude.sql
-- Claude Desktop SQL Trigger for YouTube Creative Analysis
-- Initiates the Pulser → JamPacked workflow for video analysis

-- Example usage from Claude Desktop:
-- sqlite3 data/database.sqlite < sql/analyze_youtube_from_claude.sql

-- Insert YouTube analysis task
INSERT INTO agent_task_queue (
    task_id,
    source_agent,
    target_agent,
    task_type,
    payload,
    status,
    created_at
) VALUES (
    'yt_jampacked_001',
    'claude_desktop',
    'pulser',
    'analyze_youtube',
    json('{
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "analysis_focus": "creative_effectiveness",
        "client_context": {
            "brand": "Sample Brand",
            "campaign_objective": "brand_awareness",
            "target_audience": "millennials_gen_z",
            "industry": "technology"
        },
        "warc_dimensions": [
            "strategic_planning_rigor",
            "creative_idea_excellence", 
            "business_results_delivery",
            "brand_building_impact",
            "cultural_social_value"
        ],
        "award_shows": [
            "cannes_lions",
            "effie_awards",
            "one_show",
            "dad_pencils",
            "clio_awards"
        ],
        "output_format": "comprehensive_report",
        "urgency": "normal"
    }'),
    'pending',
    datetime('now')
);

-- Verify task was created
SELECT 
    task_id,
    source_agent,
    target_agent,
    task_type,
    status,
    created_at
FROM agent_task_queue 
WHERE task_id = 'yt_jampacked_001';

-- Monitor all YouTube analysis tasks
SELECT 
    task_id,
    target_agent,
    task_type,
    status,
    created_at,
    CASE 
        WHEN status = 'completed' THEN '✅'
        WHEN status = 'in_progress' THEN '⏳'
        WHEN status = 'failed' THEN '❌'
        ELSE '⏸️'
    END as status_icon
FROM agent_task_queue 
WHERE task_type IN ('analyze_youtube', 'analyze_transcript')
ORDER BY created_at DESC;

-- Quick status check for our specific task
.mode table
.headers on
SELECT 
    '🎬 YouTube Analysis Status' as component,
    CASE 
        WHEN EXISTS(SELECT 1 FROM agent_task_queue WHERE task_id = 'yt_jampacked_001' AND status = 'completed') 
        THEN '✅ Analysis Complete'
        WHEN EXISTS(SELECT 1 FROM agent_task_queue WHERE task_id = 'yt_jampacked_001' AND status = 'in_progress') 
        THEN '⏳ Processing...'
        WHEN EXISTS(SELECT 1 FROM agent_task_queue WHERE task_id = 'yt_jampacked_001' AND status = 'failed') 
        THEN '❌ Failed'
        ELSE '⏸️ Pending'
    END as status;

-- Get results when ready
SELECT 
    'Results for yt_jampacked_001:' as info,
    result
FROM agent_task_queue 
WHERE task_id = 'yt_jampacked_001' 
AND status = 'completed';

-- Advanced query: Track the complete workflow
WITH workflow_tracking AS (
    SELECT 
        task_id,
        target_agent,
        task_type,
        status,
        created_at,
        ROW_NUMBER() OVER (ORDER BY created_at) as step_number
    FROM agent_task_queue 
    WHERE task_id LIKE 'yt_jampacked_%' 
    OR (source_agent = 'pulser' AND task_type = 'analyze_transcript')
)
SELECT 
    step_number,
    CASE target_agent
        WHEN 'pulser' THEN '🎬 Video Processing'
        WHEN 'jampacked' THEN '🧠 Creative Analysis'
        ELSE target_agent
    END as workflow_step,
    task_type,
    CASE status
        WHEN 'completed' THEN '✅ Done'
        WHEN 'in_progress' THEN '⏳ Working'
        WHEN 'failed' THEN '❌ Error'
        ELSE '⏸️ Queued'
    END as status,
    created_at
FROM workflow_tracking
ORDER BY step_number;