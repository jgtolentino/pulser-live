#!/usr/bin/env python3
"""
Verification script for JamPacked YouTube Integration
"""
import subprocess
import sqlite3
import json
import os
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    packages = ['yt-dlp', 'openai-whisper', 'sqlite3']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} Python package is available")
        except ImportError:
            print(f"❌ {package} Python package is missing")
    
    # Check command line tools
    tools = ['yt-dlp', 'whisper']
    for tool in tools:
        try:
            if tool == 'whisper':
                result = subprocess.run([tool, '--help'], capture_output=True, check=True)
                if b'usage: whisper' in result.stdout:
                    print(f"✅ {tool} command is available")
                else:
                    print(f"❌ {tool} command is not working")
            else:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
                print(f"✅ {tool} command is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {tool} command is missing")

def check_database():
    """Check database connectivity and schema"""
    print("\n🗄️  Checking database...")
    
    db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_task_queue'")
        if cursor.fetchone():
            print("✅ agent_task_queue table exists")
            
            # Check table structure
            cursor.execute("PRAGMA table_info(agent_task_queue)")
            columns = cursor.fetchall()
            expected_columns = ['task_id', 'source_agent', 'target_agent', 'task_type', 'payload', 'status', 'result', 'created_at', 'updated_at']
            table_columns = [col[1] for col in columns]
            
            missing_columns = [col for col in expected_columns if col not in table_columns]
            if missing_columns:
                print(f"❌ Missing columns: {missing_columns}")
            else:
                print("✅ Table schema is correct")
            
            # Check for tasks
            cursor.execute("SELECT COUNT(*) FROM agent_task_queue")
            task_count = cursor.fetchone()[0]
            print(f"📊 Found {task_count} tasks in queue")
            
        else:
            print("❌ agent_task_queue table does not exist")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

def test_simple_youtube_download():
    """Test basic YouTube download functionality"""
    print("\n🎬 Testing YouTube download...")
    
    try:
        # Test with a short video
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        # Test metadata extraction
        cmd = ['yt-dlp', '--dump-json', '--no-download', test_url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        
        metadata = json.loads(result.stdout)
        print(f"✅ Successfully extracted metadata for: {metadata.get('title', 'Unknown')}")
        print(f"   Duration: {metadata.get('duration', 0)} seconds")
        print(f"   Views: {metadata.get('view_count', 0)}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ YouTube download test timed out")
        return False
    except Exception as e:
        print(f"❌ YouTube download test failed: {e}")
        return False

def create_test_task():
    """Create a test task in the database"""
    print("\n📝 Creating test task...")
    
    db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        task_id = f"verify_test_{int(datetime.now().timestamp())}"
        payload = {
            "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "analysis_focus": "creative_effectiveness",
            "test_mode": True
        }
        
        cursor.execute("""
            INSERT INTO agent_task_queue 
            (task_id, source_agent, target_agent, task_type, payload, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            'verify_script',
            'pulser',
            'analyze_youtube',
            json.dumps(payload),
            'pending',
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created test task: {task_id}")
        return task_id
        
    except Exception as e:
        print(f"❌ Failed to create test task: {e}")
        return None

def main():
    """Main verification function"""
    print("🚀 JamPacked YouTube Integration Verification\n")
    
    check_dependencies()
    check_database()
    
    # Test YouTube functionality
    if test_simple_youtube_download():
        print("✅ YouTube integration is ready!")
    else:
        print("❌ YouTube integration has issues")
    
    # Create test task
    task_id = create_test_task()
    if task_id:
        print(f"\n🧪 Test task created: {task_id}")
        print("📋 To monitor: sqlite3 /Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite \"SELECT * FROM agent_task_queue WHERE task_id = '{}';\"".format(task_id))
    
    print("\n🎉 Verification complete!")

if __name__ == "__main__":
    main()