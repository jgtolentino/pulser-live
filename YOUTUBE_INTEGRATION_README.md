# 🎬 JamPacked YouTube Integration

## Overview
AI-powered creative effectiveness analysis for YouTube videos using WARC Effective 100 standards.

## Architecture
```
Claude Desktop → SQL Trigger → Pulser Agent → JamPacked Agent → Results
```

## Quick Start

### 1. Start Services
```bash
./start_youtube_integration.sh
```

### 2. Analyze Video (from Claude Desktop)
```sql
sqlite3 data/database.sqlite < sql/analyze_youtube_from_claude.sql
```

### 3. Check Results
```sql
SELECT result FROM agent_task_queue WHERE task_id = 'yt_jampacked_001';
```

### 4. Stop Services
```bash
./stop_youtube_integration.sh
```

## Testing
```bash
# Run integration tests
python -m pytest tests/test_youtube_integration.py
```

## Logs
- Location: `logs/youtube_integration_YYYYMMDD.log`
- Process IDs: `logs/pulser_youtube.pid`, `logs/jampacked_transcript.pid`

## Dependencies
- `yt-dlp`: Video download
- `whisper`: Audio transcription  
- SQLite: Task queue
- Python 3.8+

## Analysis Output
- WARC 5 Dimensions scoring
- Award prediction (Cannes, Effie, One Show, D&AD, Clio)
- Creative effectiveness metrics
- Cultural impact assessment
- ROI projections
