-- Agent Task Queue Schema for MCP Integration
-- This schema enables agent-to-agent communication and task routing

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS agent_task_queue;
DROP TABLE IF EXISTS agent_task_history;
DROP TABLE IF EXISTS agent_registry;

-- Agent Registry: Track available agents and their capabilities
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL UNIQUE,
    agent_type TEXT NOT NULL,
    capabilities TEXT,
    status TEXT DEFAULT 'active',
    endpoint TEXT,
    last_heartbeat DATETIME,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Agent Task Queue: Main queue for inter-agent tasks
CREATE TABLE IF NOT EXISTS agent_task_queue (
    task_id TEXT PRIMARY KEY,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    task_type TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    payload TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    error_message TEXT,
    result TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_agent) REFERENCES agent_registry(agent_name),
    FOREIGN KEY (target_agent) REFERENCES agent_registry(agent_name)
);

-- Agent Task History: Archive of completed/failed tasks
CREATE TABLE IF NOT EXISTS agent_task_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms INTEGER,
    payload TEXT,
    result TEXT,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES agent_task_queue(task_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_queue_status ON agent_task_queue(status);
CREATE INDEX IF NOT EXISTS idx_queue_target ON agent_task_queue(target_agent);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON agent_task_queue(priority DESC, scheduled_at ASC);
CREATE INDEX IF NOT EXISTS idx_queue_scheduled ON agent_task_queue(scheduled_at);
CREATE INDEX IF NOT EXISTS idx_history_task ON agent_task_history(task_id);
CREATE INDEX IF NOT EXISTS idx_registry_status ON agent_registry(status);

-- Triggers for automatic timestamp updates
CREATE TRIGGER IF NOT EXISTS update_agent_registry_timestamp
AFTER UPDATE ON agent_registry
BEGIN
    UPDATE agent_registry SET updated_at = CURRENT_TIMESTAMP WHERE agent_id = NEW.agent_id;
END;

CREATE TRIGGER IF NOT EXISTS update_task_queue_timestamp
AFTER UPDATE ON agent_task_queue
BEGIN
    UPDATE agent_task_queue SET updated_at = CURRENT_TIMESTAMP WHERE task_id = NEW.task_id;
END;

-- Trigger to archive completed tasks
CREATE TRIGGER IF NOT EXISTS archive_completed_tasks
AFTER UPDATE OF status ON agent_task_queue
WHEN NEW.status IN ('completed', 'failed', 'cancelled')
BEGIN
    INSERT INTO agent_task_history (
        task_id, source_agent, target_agent, task_type,
        status, duration_ms, payload, result, error_message
    )
    SELECT 
        task_id, source_agent, target_agent, task_type,
        NEW.status,
        CASE 
            WHEN NEW.completed_at IS NOT NULL AND NEW.started_at IS NOT NULL 
            THEN CAST((julianday(NEW.completed_at) - julianday(NEW.started_at)) * 86400000 AS INTEGER)
            ELSE NULL
        END,
        payload, result, error_message
    FROM agent_task_queue
    WHERE task_id = NEW.task_id;
END;

-- Insert default agents
INSERT OR IGNORE INTO agent_registry (agent_id, agent_name, agent_type, capabilities, endpoint) VALUES
    ('jampacked_001', 'JamPacked', 'analyzer', 'creative_analysis,award_prediction,csr_scoring', 'http://localhost:3001'),
    ('claude_desktop_001', 'Claude Desktop', 'interface', 'sql_queries,data_visualization', 'mcp://claude-desktop'),
    ('claude_code_001', 'Claude Code', 'processor', 'python_execution,data_processing', 'mcp://claude-code'),
    ('marian_001', 'Marian Trivera', 'researcher', 'market_analysis,trend_detection', 'http://localhost:3002'),
    ('echo_001', 'Echo', 'communicator', 'response_generation,summarization', 'http://localhost:3003');

-- Sample view for active tasks
CREATE VIEW IF NOT EXISTS active_tasks AS
SELECT 
    t.task_id,
    t.source_agent,
    t.target_agent,
    t.task_type,
    t.priority,
    t.status,
    t.retry_count,
    t.scheduled_at,
    s.agent_type as source_type,
    g.agent_type as target_type
FROM agent_task_queue t
JOIN agent_registry s ON t.source_agent = s.agent_name
JOIN agent_registry g ON t.target_agent = g.agent_name
WHERE t.status IN ('pending', 'running', 'retrying')
ORDER BY t.priority DESC, t.scheduled_at ASC;

-- Function to enqueue a new task (as a prepared statement template)
-- Usage: INSERT INTO agent_task_queue (task_id, source_agent, target_agent, task_type, payload, priority)
-- VALUES (?, ?, ?, ?, ?, ?);

-- Function to claim next task for an agent (as a prepared statement template)
-- Usage: UPDATE agent_task_queue 
-- SET status = 'running', started_at = CURRENT_TIMESTAMP 
-- WHERE task_id = (
--     SELECT task_id FROM agent_task_queue 
--     WHERE target_agent = ? AND status = 'pending' 
--     ORDER BY priority DESC, scheduled_at ASC 
--     LIMIT 1
-- );

-- Statistics view
CREATE VIEW IF NOT EXISTS task_statistics AS
SELECT 
    target_agent,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
    COUNT(CASE WHEN status = 'running' THEN 1 END) as running_count,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count,
    AVG(CASE 
        WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
        THEN CAST((julianday(completed_at) - julianday(started_at)) * 86400000 AS INTEGER)
        ELSE NULL 
    END) as avg_duration_ms
FROM agent_task_queue
GROUP BY target_agent;

PRAGMA foreign_keys = ON;