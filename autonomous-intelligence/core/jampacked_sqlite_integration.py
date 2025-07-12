#!/usr/bin/env python3
"""
JamPacked Integration with Existing SQLite MCP Server
Seamlessly stores JamPacked analysis in your existing MCP SQLite database
"""

import sqlite3
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from jampacked_custom_intelligence import JamPackedIntelligenceSuite


class JamPackedSQLiteIntegration:
    """
    Integrates JamPacked with existing SQLite MCP server
    """
    
    def __init__(self, db_path: str = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"):
        self.db_path = db_path
        self.jampacked = JamPackedIntelligenceSuite()
        
        # Initialize JamPacked tables in existing database
        self.init_jampacked_tables()
        
        print(f"✅ JamPacked integrated with SQLite MCP at: {db_path}")
    
    def init_jampacked_tables(self):
        """
        Create JamPacked tables in existing SQLite database
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Creative Analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jampacked_creative_analysis (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                campaign_name TEXT,
                analysis_type TEXT NOT NULL,
                creative_effectiveness_score REAL,
                attention_score REAL,
                emotion_score REAL,
                brand_recall_score REAL,
                cultural_alignment_score REAL,
                multimodal_score REAL,
                analysis_results TEXT,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pattern Discoveries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jampacked_pattern_discoveries (
                id TEXT PRIMARY KEY,
                campaign_id TEXT,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT,
                novelty_score REAL,
                confidence_score REAL,
                business_impact REAL,
                pattern_data TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Cultural Insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jampacked_cultural_insights (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                culture TEXT NOT NULL,
                effectiveness_score REAL,
                appropriateness_score REAL,
                adaptation_recommendations TEXT,
                risk_assessment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Optimization Recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jampacked_optimizations (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                optimization_type TEXT,
                description TEXT,
                predicted_impact REAL,
                implementation_effort TEXT,
                priority_score REAL,
                ab_test_plan TEXT,
                status TEXT DEFAULT 'proposed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Campaign Sessions table (for cross-interface continuity)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jampacked_sessions (
                session_id TEXT PRIMARY KEY,
                campaign_id TEXT,
                session_type TEXT,
                interface TEXT,
                metadata TEXT,
                artifacts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_campaign_id ON jampacked_creative_analysis(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_campaign ON jampacked_pattern_discoveries(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cultural_campaign ON jampacked_cultural_insights(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_optimization_campaign ON jampacked_optimizations(campaign_id)")
        
        conn.commit()
        conn.close()
    
    async def analyze_and_store(self, 
                              materials: Dict[str, Any],
                              campaign_context: Dict[str, Any],
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run JamPacked analysis and store results in SQLite
        """
        
        # Generate campaign ID if not provided
        campaign_id = campaign_context.get('campaign_id', self.generate_campaign_id(campaign_context))
        campaign_context['campaign_id'] = campaign_id
        
        # Create or update session
        if not session_id:
            session_id = self.create_session(campaign_id, 'analysis', 'python_api')
        else:
            self.update_session(session_id)
        
        # Run JamPacked analysis
        print(f"🔍 Running JamPacked analysis for campaign: {campaign_id}")
        analysis_results = await self.jampacked.analyze_campaign_materials(materials, campaign_context)
        
        # Store results in SQLite
        self.store_creative_analysis(campaign_id, campaign_context, analysis_results)
        self.store_pattern_discoveries(campaign_id, analysis_results)
        self.store_cultural_insights(campaign_id, analysis_results)
        self.store_optimizations(campaign_id, analysis_results)
        
        # Return results with storage confirmation
        return {
            'campaign_id': campaign_id,
            'session_id': session_id,
            'analysis_results': analysis_results,
            'storage_status': 'success',
            'access_via_mcp': True,
            'sql_queries': self.get_useful_queries(campaign_id)
        }
    
    def store_creative_analysis(self, campaign_id: str, context: Dict, results: Dict):
        """
        Store creative analysis results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        analysis_id = self.generate_id('analysis', campaign_id)
        
        # Extract scores
        scores = results.get('overall_scores', {})
        
        cursor.execute("""
            INSERT INTO jampacked_creative_analysis 
            (id, campaign_id, campaign_name, analysis_type, 
             creative_effectiveness_score, attention_score, emotion_score,
             brand_recall_score, cultural_alignment_score, multimodal_score,
             analysis_results, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id,
            campaign_id,
            context.get('campaign_name', 'Unnamed Campaign'),
            'comprehensive',
            scores.get('effectiveness', 0),
            scores.get('attention', 0),
            scores.get('emotion', 0),
            scores.get('brand_recall', 0),
            scores.get('cultural_alignment', 0),
            scores.get('multimodal', 0),
            json.dumps(results.get('detailed_analysis', {})),
            json.dumps(results.get('actionable_recommendations', []))
        ))
        
        conn.commit()
        conn.close()
    
    def store_pattern_discoveries(self, campaign_id: str, results: Dict):
        """
        Store discovered patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        patterns = results.get('detailed_analysis', {}).get('discovered_patterns', {})
        
        # Store novel patterns
        if 'novel_discoveries' in patterns:
            for pattern in patterns['novel_discoveries']:
                pattern_id = self.generate_id('pattern', campaign_id)
                
                cursor.execute("""
                    INSERT INTO jampacked_pattern_discoveries
                    (id, campaign_id, pattern_type, pattern_description,
                     novelty_score, confidence_score, business_impact, pattern_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    campaign_id,
                    pattern.get('type', 'unknown'),
                    pattern.get('description', ''),
                    pattern.get('novelty_score', 0),
                    pattern.get('confidence', 0),
                    pattern.get('business_impact', 0),
                    json.dumps(pattern)
                ))
        
        conn.commit()
        conn.close()
    
    def store_cultural_insights(self, campaign_id: str, results: Dict):
        """
        Store cultural effectiveness insights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cultural_insights = results.get('detailed_analysis', {}).get('cultural_insights', {})
        
        if 'individual_cultures' in cultural_insights:
            for culture, insights in cultural_insights['individual_cultures'].items():
                insight_id = self.generate_id('cultural', f"{campaign_id}_{culture}")
                
                cursor.execute("""
                    INSERT INTO jampacked_cultural_insights
                    (id, campaign_id, culture, effectiveness_score,
                     appropriateness_score, adaptation_recommendations, risk_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight_id,
                    campaign_id,
                    culture,
                    insights.get('cultural_effectiveness_score', 0),
                    insights.get('appropriateness_assessment', {}).get('score', 0),
                    json.dumps(insights.get('adaptation_recommendations', [])),
                    json.dumps(insights.get('risk_assessment', {}))
                ))
        
        conn.commit()
        conn.close()
    
    def store_optimizations(self, campaign_id: str, results: Dict):
        """
        Store optimization recommendations
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        recommendations = results.get('actionable_recommendations', [])
        
        for rec in recommendations:
            opt_id = self.generate_id('opt', campaign_id)
            
            cursor.execute("""
                INSERT INTO jampacked_optimizations
                (id, campaign_id, optimization_type, description,
                 predicted_impact, implementation_effort, priority_score, ab_test_plan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                opt_id,
                campaign_id,
                rec.get('type', 'general'),
                rec.get('title', ''),
                rec.get('impact', 0),
                rec.get('effort', 'medium'),
                rec.get('priority', 0),
                json.dumps(rec.get('ab_test_plan', {}))
            ))
        
        conn.commit()
        conn.close()
    
    def create_session(self, campaign_id: str, session_type: str, interface: str) -> str:
        """
        Create new analysis session
        """
        session_id = self.generate_id('session', campaign_id)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO jampacked_sessions
            (session_id, campaign_id, session_type, interface, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            campaign_id,
            session_type,
            interface,
            json.dumps({'start_time': datetime.now().isoformat()})
        ))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_session(self, session_id: str):
        """
        Update session last accessed time
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE jampacked_sessions 
            SET last_accessed = datetime('now')
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_analysis_results(self, campaign_id: str) -> Dict[str, Any]:
        """
        Retrieve analysis results from SQLite
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest analysis
        cursor.execute("""
            SELECT * FROM jampacked_creative_analysis
            WHERE campaign_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (campaign_id,))
        
        analysis = cursor.fetchone()
        
        # Get patterns
        cursor.execute("""
            SELECT * FROM jampacked_pattern_discoveries
            WHERE campaign_id = ?
            ORDER BY novelty_score DESC
        """, (campaign_id,))
        
        patterns = cursor.fetchall()
        
        # Get cultural insights
        cursor.execute("""
            SELECT * FROM jampacked_cultural_insights
            WHERE campaign_id = ?
        """, (campaign_id,))
        
        cultural = cursor.fetchall()
        
        # Get optimizations
        cursor.execute("""
            SELECT * FROM jampacked_optimizations
            WHERE campaign_id = ?
            ORDER BY priority_score DESC
        """, (campaign_id,))
        
        optimizations = cursor.fetchall()
        
        conn.close()
        
        return {
            'analysis': self.row_to_dict(analysis) if analysis else None,
            'patterns': [self.row_to_dict(p) for p in patterns],
            'cultural_insights': [self.row_to_dict(c) for c in cultural],
            'optimizations': [self.row_to_dict(o) for o in optimizations]
        }
    
    def get_useful_queries(self, campaign_id: str) -> Dict[str, str]:
        """
        Return useful SQL queries for Claude to use
        """
        return {
            'get_latest_analysis': f"SELECT * FROM jampacked_creative_analysis WHERE campaign_id = '{campaign_id}' ORDER BY created_at DESC LIMIT 1",
            'get_top_patterns': f"SELECT * FROM jampacked_pattern_discoveries WHERE campaign_id = '{campaign_id}' ORDER BY novelty_score DESC LIMIT 10",
            'get_cultural_insights': f"SELECT * FROM jampacked_cultural_insights WHERE campaign_id = '{campaign_id}'",
            'get_high_priority_optimizations': f"SELECT * FROM jampacked_optimizations WHERE campaign_id = '{campaign_id}' AND priority_score > 0.7 ORDER BY priority_score DESC",
            'get_all_campaigns': "SELECT DISTINCT campaign_id, campaign_name FROM jampacked_creative_analysis ORDER BY created_at DESC",
            'get_session_history': f"SELECT * FROM jampacked_sessions WHERE campaign_id = '{campaign_id}' ORDER BY last_accessed DESC"
        }
    
    def generate_campaign_id(self, context: Dict) -> str:
        """Generate unique campaign ID"""
        name = context.get('campaign_name', 'unnamed')
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{name}_{timestamp}".encode()).hexdigest()[:16]
    
    def generate_id(self, prefix: str, seed: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{prefix}_{seed}_{timestamp}".encode()).hexdigest()[:16]
    
    def row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary"""
        if not row:
            return {}
        
        # Get column names from cursor description
        # For now, return as list (in production, would map to column names)
        return {
            'data': row
        }


# Convenience functions for Claude MCP access

async def analyze_campaign_via_mcp(materials: Dict[str, Any], 
                                  campaign_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze campaign and store in MCP SQLite database
    Claude Desktop can then query results using SQL
    """
    integration = JamPackedSQLiteIntegration()
    return await integration.analyze_and_store(materials, campaign_context)


def get_campaign_results_via_mcp(campaign_id: str) -> Dict[str, Any]:
    """
    Retrieve campaign results from MCP SQLite database
    """
    integration = JamPackedSQLiteIntegration()
    return integration.get_analysis_results(campaign_id)


# Example usage showing Claude MCP integration
async def demonstrate_mcp_integration():
    """
    Demonstrate how JamPacked integrates with existing MCP SQLite
    """
    
    print("🔗 JamPacked + MCP SQLite Integration Demo")
    print("=" * 50)
    
    # Sample campaign
    materials = {
        'text': ['Innovative solutions for modern businesses'],
        'images': []  # Would include actual image data
    }
    
    context = {
        'campaign_name': 'Q4 Product Launch',
        'target_cultures': ['us', 'uk'],
        'business_objectives': ['awareness', 'conversion']
    }
    
    # Analyze and store in MCP SQLite
    results = await analyze_campaign_via_mcp(materials, context)
    
    print(f"\n✅ Analysis stored in MCP SQLite")
    print(f"Campaign ID: {results['campaign_id']}")
    print(f"Session ID: {results['session_id']}")
    
    print("\n📊 You can now query results in Claude Desktop using:")
    for query_name, query in results['sql_queries'].items():
        print(f"\n-- {query_name}")
        print(f"{query}")
    
    print("\n🎯 Benefits of this integration:")
    print("1. No duplicate infrastructure - uses your existing MCP SQLite")
    print("2. Claude Desktop can query results directly via SQL") 
    print("3. Claude Code can access same data through shared database")
    print("4. All JamPacked intelligence accessible through standard SQL")
    
    return results


if __name__ == "__main__":
    asyncio.run(demonstrate_mcp_integration())