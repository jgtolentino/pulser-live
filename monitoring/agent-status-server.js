#!/usr/bin/env node
/**
 * Agent Status Server
 * Provides real-time monitoring API for all agents
 */

const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.MONITOR_PORT || 3002;

// Database path
const DB_PATH = path.join(
  process.env.HOME,
  'Documents/GitHub/mcp-sqlite-server/data/database.sqlite'
);

// Enable CORS for dashboard
app.use(cors());
app.use(express.json());

// Serve static files (dashboard)
app.use(express.static(path.join(__dirname)));

// Agent registry with roles
const AGENT_REGISTRY = {
  claude_cli: {
    name: 'Claude Code CLI',
    role: 'Local dev + task runner',
    icon: '🖥️'
  },
  claude_desktop: {
    name: 'Claude Desktop',
    role: 'UI / document analysis',
    icon: '🖼️'
  },
  claude_agent: {
    name: 'Claude Agent',
    role: 'Task orchestration',
    icon: '🤖'
  },
  bruno: {
    name: 'Bruno',
    role: 'System-level executor',
    icon: '🔨'
  },
  pulser_cli: {
    name: 'Pulser CLI',
    role: 'Pipeline orchestrator',
    icon: '🎯'
  },
  jampacked: {
    name: 'JamPacked',
    role: 'Creative intelligence',
    icon: '🧠'
  }
};

// Get database connection
function getDb() {
  return new sqlite3.Database(DB_PATH, sqlite3.OPEN_READONLY);
}

// API: Get agent status
app.get('/api/mcp/agents/status', async (req, res) => {
  const db = getDb();
  
  try {
    // Get last task for each agent
    const query = `
      SELECT 
        source_agent,
        target_agent,
        task_id,
        task_type,
        status,
        created_at,
        updated_at
      FROM agent_task_queue
      WHERE task_id IN (
        SELECT MAX(task_id)
        FROM agent_task_queue
        GROUP BY source_agent
        UNION
        SELECT MAX(task_id)
        FROM agent_task_queue
        GROUP BY target_agent
      )
      ORDER BY updated_at DESC
    `;
    
    db.all(query, [], (err, rows) => {
      if (err) {
        res.status(500).json({ error: 'Database error' });
        return;
      }
      
      // Build agent status map
      const agentStatus = {};
      
      // Initialize all agents as idle
      Object.keys(AGENT_REGISTRY).forEach(agentId => {
        agentStatus[agentId] = {
          ...AGENT_REGISTRY[agentId],
          status: 'Idle',
          last_task: 'No recent tasks',
          last_activity: null
        };
      });
      
      // Update with actual task data
      rows.forEach(row => {
        // Check source agents
        if (row.source_agent && AGENT_REGISTRY[row.source_agent]) {
          const agent = agentStatus[row.source_agent];
          agent.last_task = `Sent: ${row.task_type} (${row.task_id})`;
          agent.last_activity = row.created_at;
          
          if (row.status === 'pending' || row.status === 'in_progress') {
            agent.status = 'Active';
          } else {
            agent.status = 'Running';
          }
        }
        
        // Check target agents
        if (row.target_agent && AGENT_REGISTRY[row.target_agent]) {
          const agent = agentStatus[row.target_agent];
          
          // Only update if more recent than source activity
          if (!agent.last_activity || row.updated_at > agent.last_activity) {
            agent.last_task = `Processing: ${row.task_type} (${row.task_id})`;
            agent.last_activity = row.updated_at;
            
            if (row.status === 'in_progress') {
              agent.status = 'Processing';
            } else if (row.status === 'completed') {
              agent.status = 'Running';
            } else if (row.status === 'failed') {
              agent.status = 'Error';
            }
          }
        }
      });
      
      // Convert to array
      const agents = Object.values(agentStatus);
      res.json(agents);
    });
  } finally {
    db.close();
  }
});

// API: Get task queue statistics
app.get('/api/mcp/tasks/stats', async (req, res) => {
  const db = getDb();
  
  try {
    const query = `
      SELECT 
        status,
        COUNT(*) as count
      FROM agent_task_queue
      WHERE created_at > datetime('now', '-1 hour')
      GROUP BY status
    `;
    
    db.all(query, [], (err, rows) => {
      if (err) {
        res.status(500).json({ error: 'Database error' });
        return;
      }
      
      const stats = {
        pending: 0,
        in_progress: 0,
        completed: 0,
        failed: 0,
        approved: 0
      };
      
      rows.forEach(row => {
        stats[row.status] = row.count;
      });
      
      res.json(stats);
    });
  } finally {
    db.close();
  }
});

// API: Get recent tasks
app.get('/api/mcp/tasks/recent', async (req, res) => {
  const db = getDb();
  const limit = parseInt(req.query.limit) || 20;
  
  try {
    const query = `
      SELECT 
        task_id,
        source_agent,
        target_agent,
        task_type,
        status,
        created_at,
        updated_at
      FROM agent_task_queue
      ORDER BY created_at DESC
      LIMIT ?
    `;
    
    db.all(query, [limit], (err, rows) => {
      if (err) {
        res.status(500).json({ error: 'Database error' });
        return;
      }
      
      res.json(rows);
    });
  } finally {
    db.close();
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'agent-monitor' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Agent Status Server running on http://localhost:${PORT}`);
  console.log(`📊 Dashboard: http://localhost:${PORT}/dashboard.html`);
  console.log(`🔍 API: http://localhost:${PORT}/api/mcp/agents/status`);
});