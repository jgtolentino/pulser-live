-- Task Queue Schema for Agent Delegation
-- Enables Claude Desktop to delegate tasks to Bruno and Pulser CLI

CREATE TABLE IF NOT EXISTS agent_task_queue (
  task_id TEXT PRIMARY KEY,
  source_agent TEXT NOT NULL,
  target_agent TEXT NOT NULL,
  task_type TEXT NOT NULL,
  payload TEXT NOT NULL,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
  result TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  retry_count INTEGER DEFAULT 0
);

-- Index for efficient queries
CREATE INDEX IF NOT EXISTS idx_task_status_target ON agent_task_queue(status, target_agent);
CREATE INDEX IF NOT EXISTS idx_task_created ON agent_task_queue(created_at);

-- Sample delegation tasks from Claude Desktop
-- INSERT INTO agent_task_queue
-- (task_id, source_agent, target_agent, task_type, payload)
-- VALUES (
--   'task_001',
--   'claude_desktop',
--   'bruno',
--   'exec',
--   '{"command": "sudo docker restart postgres_container"}'
-- );

-- INSERT INTO agent_task_queue
-- (task_id, source_agent, target_agent, task_type, payload)
-- VALUES (
--   'task_002',
--   'claude_desktop',
--   'pulser_cli',
--   'orchestrate',
--   '{"plan": "run etl pipeline scout_gold_to_insights"}'
-- );