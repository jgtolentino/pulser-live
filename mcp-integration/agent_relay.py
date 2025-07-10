#!/usr/bin/env python3
"""
Agent Relay Dispatcher for MCP Integration
Polls the SQLite queue and dispatches tasks to appropriate agents
"""

import sqlite3
import json
import time
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Optional, Any
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AgentRelay')

class AgentRelay:
    def __init__(self, db_path: str = None):
        """Initialize the Agent Relay dispatcher"""
        self.db_path = db_path or os.path.expanduser('~/Documents/GitHub/mcp-sqlite-server/data/database.sqlite')
        self.running = False
        self.session = None
        self.agent_handlers = {
            'JamPacked': self.handle_jampacked_task,
            'Claude Desktop': self.handle_claude_desktop_task,
            'Claude Code': self.handle_claude_code_task,
            'Marian Trivera': self.handle_marian_task,
            'Echo': self.handle_echo_task
        }
        
    def get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def start(self):
        """Start the agent relay dispatcher"""
        logger.info(f"🚀 Starting Agent Relay Dispatcher")
        logger.info(f"📁 Database: {self.db_path}")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        self.running = True
        
        # Start heartbeat task
        asyncio.create_task(self.heartbeat_loop())
        
        # Main processing loop
        await self.process_loop()
        
    async def stop(self):
        """Stop the dispatcher gracefully"""
        logger.info("🛑 Stopping Agent Relay Dispatcher")
        self.running = False
        
        if self.session:
            await self.session.close()
            
    async def process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get next tasks for all agents
                tasks = self.get_pending_tasks()
                
                if tasks:
                    # Process tasks concurrently
                    await asyncio.gather(*[
                        self.process_task(task) for task in tasks
                    ], return_exceptions=True)
                else:
                    # No tasks, wait before polling again
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(5)
    
    def get_pending_tasks(self, limit: int = 10) -> list:
        """Get pending tasks from the queue"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM agent_task_queue
                WHERE status = 'pending' 
                  AND scheduled_at <= datetime('now')
                  AND retry_count < max_retries
                ORDER BY priority DESC, scheduled_at ASC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    async def process_task(self, task: Dict[str, Any]):
        """Process a single task"""
        task_id = task['task_id']
        target_agent = task['target_agent']
        
        logger.info(f"📋 Processing task {task_id} for {target_agent}")
        
        # Mark task as running
        self.update_task_status(task_id, 'running')
        
        try:
            # Get handler for target agent
            handler = self.agent_handlers.get(target_agent)
            
            if not handler:
                raise ValueError(f"No handler for agent: {target_agent}")
            
            # Execute task
            result = await handler(task)
            
            # Mark task as completed
            self.complete_task(task_id, result)
            logger.info(f"✅ Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Task {task_id} failed: {e}")
            self.fail_task(task_id, str(e))
            
            # Retry if applicable
            if task['retry_count'] < task['max_retries']:
                self.retry_task(task_id)
    
    def update_task_status(self, task_id: str, status: str):
        """Update task status"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE agent_task_queue
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
            """, (status, task_id))
            conn.commit()
    
    def complete_task(self, task_id: str, result: Any):
        """Mark task as completed"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE agent_task_queue
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    result = ?
                WHERE task_id = ?
            """, (json.dumps(result), task_id))
            conn.commit()
    
    def fail_task(self, task_id: str, error_message: str):
        """Mark task as failed"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE agent_task_queue
                SET status = 'failed',
                    completed_at = CURRENT_TIMESTAMP,
                    error_message = ?,
                    retry_count = retry_count + 1
                WHERE task_id = ?
            """, (error_message, task_id))
            conn.commit()
    
    def retry_task(self, task_id: str):
        """Schedule task for retry"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE agent_task_queue
                SET status = 'retrying',
                    scheduled_at = datetime('now', '+' || (retry_count * 30) || ' seconds')
                WHERE task_id = ?
            """, (task_id,))
            conn.commit()
    
    async def heartbeat_loop(self):
        """Update agent heartbeats periodically"""
        while self.running:
            try:
                with self.get_connection() as conn:
                    # Update heartbeat for active agents
                    conn.execute("""
                        UPDATE agent_registry
                        SET last_heartbeat = CURRENT_TIMESTAMP
                        WHERE agent_name IN ('JamPacked', 'Agent Relay')
                    """)
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                
            await asyncio.sleep(30)  # Every 30 seconds
    
    # Agent-specific handlers
    
    async def handle_jampacked_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JamPacked analysis tasks"""
        payload = json.loads(task['payload'])
        
        # Call JamPacked API
        async with self.session.post(
            'http://localhost:3001/api/analyze',
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            result = await response.json()
            
        return result
    
    async def handle_claude_desktop_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Claude Desktop SQL tasks"""
        payload = json.loads(task['payload'])
        
        # Execute SQL query
        with self.get_connection() as conn:
            cursor = conn.execute(payload['query'], payload.get('params', []))
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
        return {
            'columns': columns,
            'rows': [dict(zip(columns, row)) for row in rows],
            'row_count': len(rows)
        }
    
    async def handle_claude_code_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Claude Code Python execution tasks"""
        payload = json.loads(task['payload'])
        
        # In production, this would execute Python code safely
        # For now, return mock result
        return {
            'status': 'executed',
            'output': f"Processed {payload.get('code_type', 'analysis')} task",
            'execution_time': 1.23
        }
    
    async def handle_marian_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Marian Trivera research tasks"""
        payload = json.loads(task['payload'])
        
        # Mock market analysis
        return {
            'market_trends': ['sustainability', 'digital transformation', 'personalization'],
            'competitor_insights': ['Brand X increased spend by 20%', 'Brand Y launched new campaign'],
            'recommendations': ['Focus on eco-friendly messaging', 'Increase digital presence']
        }
    
    async def handle_echo_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Echo communication tasks"""
        payload = json.loads(task['payload'])
        
        # Mock response generation
        return {
            'response': f"Processed communication task: {payload.get('message_type', 'general')}",
            'summary': 'Task completed successfully',
            'sentiment': 'positive'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                FROM agent_task_queue
            """)
            
            stats = dict(cursor.fetchone())
            
            # Get per-agent statistics
            cursor = conn.execute("""
                SELECT target_agent, COUNT(*) as task_count
                FROM agent_task_queue
                WHERE status IN ('pending', 'running')
                GROUP BY target_agent
            """)
            
            stats['by_agent'] = {row['target_agent']: row['task_count'] 
                                for row in cursor.fetchall()}
            
            return stats

async def main():
    """Main entry point"""
    relay = AgentRelay()
    
    # Print startup statistics
    stats = relay.get_statistics()
    logger.info(f"📊 Queue Statistics: {stats}")
    
    try:
        await relay.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await relay.stop()

if __name__ == "__main__":
    asyncio.run(main())