# agents/pulser/handlers/extract_and_analyze_youtube.py
import sqlite3
import json
import subprocess
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

class PulserYouTubeHandler:
    """
    Pulser Agent: YouTube Video Extraction and Transcription
    Downloads video, extracts audio, transcribes with Whisper
    """
    
    def __init__(self, db_path="/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.mkdtemp()
        
    def process_youtube_task(self, task_data):
        """
        Main processing method for YouTube analysis tasks
        """
        try:
            video_url = task_data.get('video_url')
            if not video_url:
                raise ValueError("No video URL provided")
            
            # Step 1: Download video metadata
            metadata = self.extract_video_metadata(video_url)
            
            # Step 2: Download video/audio
            audio_path = self.download_audio(video_url)
            
            # Step 3: Transcribe with Whisper
            transcript = self.transcribe_audio(audio_path)
            
            # Step 4: Clean up temporary files
            self.cleanup_temp_files(audio_path)
            
            # Step 5: Delegate to JamPacked for analysis
            analysis_task_id = self.delegate_to_jampacked(transcript, metadata, task_data)
            
            return {
                'status': 'success',
                'transcript': transcript,
                'metadata': metadata,
                'analysis_task_id': analysis_task_id,
                'message': 'Video processed successfully, analysis delegated to JamPacked'
            }
            
        except Exception as e:
            self.logger.error(f"YouTube processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to process YouTube video'
            }
    
    def extract_video_metadata(self, video_url):
        """
        Extract video metadata using yt-dlp
        """
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            # Extract relevant metadata for creative analysis
            return {
                'title': metadata.get('title', ''),
                'description': metadata.get('description', ''),
                'duration': metadata.get('duration', 0),
                'view_count': metadata.get('view_count', 0),
                'like_count': metadata.get('like_count', 0),
                'upload_date': metadata.get('upload_date', ''),
                'uploader': metadata.get('uploader', ''),
                'channel': metadata.get('channel', ''),
                'tags': metadata.get('tags', []),
                'categories': metadata.get('categories', []),
                'resolution': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                'fps': metadata.get('fps', 0),
                'format': metadata.get('ext', ''),
                'thumbnail': metadata.get('thumbnail', ''),
                'video_url': video_url,
                'extracted_at': datetime.now().isoformat()
            }
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract metadata: {e.stderr}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse metadata JSON: {str(e)}")
    
    def download_audio(self, video_url):
        """
        Download audio from YouTube video using yt-dlp
        """
        try:
            audio_filename = f"audio_{uuid4().hex}.wav"
            audio_path = os.path.join(self.temp_dir, audio_filename)
            
            cmd = [
                'yt-dlp',
                '-x',  # Extract audio only
                '--audio-format', 'wav',
                '--audio-quality', '0',  # Best quality
                '-o', audio_path.replace('.wav', '.%(ext)s'),
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # yt-dlp might change the extension, find the actual file
            base_path = audio_path.replace('.wav', '')
            for ext in ['.wav', '.m4a', '.mp3']:
                potential_path = base_path + ext
                if os.path.exists(potential_path):
                    return potential_path
            
            raise Exception("Audio file not found after download")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to download audio: {e.stderr}")
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using OpenAI Whisper
        """
        try:
            cmd = [
                'whisper',
                audio_path,
                '--model', 'base',  # Use base model for speed, upgrade to 'large' for accuracy
                '--output_format', 'txt',
                '--output_dir', self.temp_dir,
                '--language', 'en',  # Default to English, could be auto-detected
                '--fp16', 'False'  # Disable for CPU compatibility
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find transcript file
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            transcript_path = os.path.join(self.temp_dir, f"{base_name}.txt")
            
            if os.path.exists(transcript_path):
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                return transcript
            else:
                raise Exception("Transcript file not found")
                
        except subprocess.CalledProcessError as e:
            raise Exception(f"Whisper transcription failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"Transcription error: {str(e)}")
    
    def delegate_to_jampacked(self, transcript, metadata, original_task):
        """
        Create task for JamPacked to analyze the transcript
        """
        analysis_task_id = f"jampacked_analysis_{uuid4().hex[:8]}"
        
        payload = {
            'transcript': transcript,
            'metadata': metadata,
            'original_request': original_task,
            'analysis_type': 'youtube_creative_effectiveness'
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_task_queue 
            (task_id, source_agent, target_agent, task_type, payload, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_task_id,
            'pulser',
            'jampacked',
            'analyze_transcript',
            json.dumps(payload),
            'pending',
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created analysis task {analysis_task_id} for JamPacked")
        return analysis_task_id
    
    def cleanup_temp_files(self, *file_paths):
        """
        Clean up temporary files
        """
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Cleaned up {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def __del__(self):
        """
        Cleanup temp directory on destruction
        """
        try:
            import shutil
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

def run_pulser_task_runner():
    """
    Task runner for Pulser YouTube processing
    Polls the task queue for YouTube analysis requests
    """
    db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    handler = PulserYouTubeHandler(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🎬 Pulser YouTube Handler started")
    
    while True:
        try:
            # Poll for YouTube analysis tasks
            cursor.execute("""
                SELECT task_id, payload FROM agent_task_queue 
                WHERE target_agent = 'pulser' 
                AND task_type = 'analyze_youtube' 
                AND status = 'pending'
                LIMIT 1
            """)
            
            task = cursor.fetchone()
            if task:
                task_id, payload = task
                task_data = json.loads(payload)
                
                print(f"📥 Processing YouTube task {task_id}")
                
                # Update status to in_progress
                cursor.execute("""
                    UPDATE agent_task_queue 
                    SET status = 'in_progress', updated_at = ?
                    WHERE task_id = ?
                """, (datetime.now().isoformat(), task_id))
                conn.commit()
                
                try:
                    # Process the YouTube video
                    result = handler.process_youtube_task(task_data)
                    
                    # Update with results
                    cursor.execute("""
                        UPDATE agent_task_queue 
                        SET status = 'completed', result = ?, updated_at = ?
                        WHERE task_id = ?
                    """, (json.dumps(result), datetime.now().isoformat(), task_id))
                    
                    print(f"✅ Completed YouTube processing for task {task_id}")
                    if result.get('analysis_task_id'):
                        print(f"🧠 Delegated analysis to JamPacked: {result['analysis_task_id']}")
                    
                except Exception as e:
                    # Update with error
                    error_result = {
                        'status': 'error',
                        'error': str(e),
                        'message': 'Pulser processing failed'
                    }
                    
                    cursor.execute("""
                        UPDATE agent_task_queue 
                        SET status = 'failed', result = ?, updated_at = ?
                        WHERE task_id = ?
                    """, (json.dumps(error_result), datetime.now().isoformat(), task_id))
                    
                    print(f"❌ Failed YouTube processing for task {task_id}: {e}")
                
                conn.commit()
            
            import time
            time.sleep(5)  # Poll every 5 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Pulser task runner stopped")
            break
        except Exception as e:
            print(f"❌ Task runner error: {e}")
            import time
            time.sleep(10)  # Wait longer on error
    
    conn.close()

# Installation check
def check_dependencies():
    """
    Check if required tools are installed
    """
    dependencies = ['yt-dlp', 'whisper']
    missing = []
    
    for tool in dependencies:
        try:
            if tool == 'whisper':
                result = subprocess.run([tool, '--help'], capture_output=True, check=True)
                if b'usage: whisper' in result.stdout:
                    print(f"✅ {tool} is installed")
                else:
                    raise subprocess.CalledProcessError(1, tool)
            else:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
                print(f"✅ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)
            print(f"❌ {tool} is missing")
    
    if missing:
        print("\n📦 Install missing dependencies:")
        for tool in missing:
            if tool == 'yt-dlp':
                print("   pip install yt-dlp")
            elif tool == 'whisper':
                print("   pip install openai-whisper")
        return False
    
    print("\n🎉 All dependencies are installed!")
    return True

if __name__ == "__main__":
    if check_dependencies():
        run_pulser_task_runner()
    else:
        print("⚠️  Please install missing dependencies before running")