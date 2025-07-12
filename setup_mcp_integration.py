#!/usr/bin/env python3
"""
Setup script to integrate JamPacked with existing MCP SQLite server
Run this once to initialize JamPacked tables in your MCP database
"""

import sys
import os
import json
from pathlib import Path

# Add JamPacked to path
sys.path.append(str(Path(__file__).parent / 'autonomous-intelligence' / 'core'))

from jampacked_sqlite_integration import JamPackedSQLiteIntegration


def setup_jampacked_mcp_integration():
    """
    Setup JamPacked integration with existing MCP SQLite server
    """
    
    print("🚀 JamPacked MCP Integration Setup")
    print("=" * 50)
    
    # Configuration
    mcp_db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    
    print(f"\n📍 MCP SQLite Database: {mcp_db_path}")
    
    # Check if database exists
    if not Path(mcp_db_path).exists():
        print("❌ MCP SQLite database not found!")
        print("Please ensure your MCP SQLite server is properly configured.")
        return False
    
    # Initialize JamPacked integration
    print("\n🔧 Initializing JamPacked tables in MCP database...")
    
    try:
        integration = JamPackedSQLiteIntegration(db_path=mcp_db_path)
        print("✅ JamPacked tables created successfully!")
        
        # Create example queries file for Claude Desktop
        create_example_queries()
        
        # Show next steps
        print("\n📋 Next Steps:")
        print("\n1. In Claude Desktop, your existing MCP SQLite server now has JamPacked tables")
        print("\n2. Use these SQL queries to access JamPacked analysis:")
        print("   - SELECT * FROM jampacked_creative_analysis")
        print("   - SELECT * FROM jampacked_pattern_discoveries")
        print("   - SELECT * FROM jampacked_cultural_insights")
        print("   - SELECT * FROM jampacked_optimizations")
        
        print("\n3. To analyze a campaign from Python:")
        print("   from jampacked_sqlite_integration import analyze_campaign_via_mcp")
        print("   results = await analyze_campaign_via_mcp(materials, context)")
        
        print("\n4. Claude Desktop and Claude Code now share the same JamPacked data!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False


def create_example_queries():
    """
    Create example SQL queries for Claude Desktop
    """
    
    queries = {
        "JamPacked_Queries": {
            "description": "Useful queries for accessing JamPacked analysis in Claude Desktop",
            
            "get_all_campaigns": {
                "description": "List all analyzed campaigns",
                "query": """
                    SELECT DISTINCT 
                        campaign_id,
                        campaign_name,
                        creative_effectiveness_score,
                        created_at
                    FROM jampacked_creative_analysis
                    ORDER BY created_at DESC
                """
            },
            
            "get_campaign_analysis": {
                "description": "Get full analysis for a specific campaign",
                "query": """
                    SELECT 
                        campaign_name,
                        creative_effectiveness_score,
                        attention_score,
                        emotion_score,
                        brand_recall_score,
                        cultural_alignment_score,
                        analysis_results,
                        recommendations
                    FROM jampacked_creative_analysis
                    WHERE campaign_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
            },
            
            "get_novel_patterns": {
                "description": "Find most novel patterns discovered",
                "query": """
                    SELECT 
                        campaign_id,
                        pattern_type,
                        pattern_description,
                        novelty_score,
                        business_impact
                    FROM jampacked_pattern_discoveries
                    WHERE novelty_score > 0.8
                    ORDER BY novelty_score DESC
                    LIMIT 20
                """
            },
            
            "get_cultural_recommendations": {
                "description": "Get cultural adaptation recommendations",
                "query": """
                    SELECT 
                        ci.culture,
                        ci.effectiveness_score,
                        ci.adaptation_recommendations,
                        ca.campaign_name
                    FROM jampacked_cultural_insights ci
                    JOIN jampacked_creative_analysis ca ON ci.campaign_id = ca.campaign_id
                    WHERE ci.effectiveness_score < 0.7
                    ORDER BY ci.effectiveness_score ASC
                """
            },
            
            "get_high_impact_optimizations": {
                "description": "Find high-impact optimization opportunities",
                "query": """
                    SELECT 
                        o.description,
                        o.predicted_impact,
                        o.implementation_effort,
                        o.priority_score,
                        ca.campaign_name
                    FROM jampacked_optimizations o
                    JOIN jampacked_creative_analysis ca ON o.campaign_id = ca.campaign_id
                    WHERE o.predicted_impact > 0.3
                    AND o.implementation_effort IN ('low', 'medium')
                    ORDER BY o.priority_score DESC
                """
            },
            
            "campaign_performance_summary": {
                "description": "Summary of campaign performance metrics",
                "query": """
                    SELECT 
                        campaign_name,
                        COUNT(DISTINCT pd.id) as patterns_discovered,
                        AVG(ci.effectiveness_score) as avg_cultural_score,
                        COUNT(DISTINCT o.id) as optimization_count,
                        MAX(ca.creative_effectiveness_score) as effectiveness_score
                    FROM jampacked_creative_analysis ca
                    LEFT JOIN jampacked_pattern_discoveries pd ON ca.campaign_id = pd.campaign_id
                    LEFT JOIN jampacked_cultural_insights ci ON ca.campaign_id = ci.campaign_id
                    LEFT JOIN jampacked_optimizations o ON ca.campaign_id = o.campaign_id
                    GROUP BY ca.campaign_id, campaign_name
                    ORDER BY effectiveness_score DESC
                """
            }
        }
    }
    
    # Save queries to file
    queries_path = Path(__file__).parent / "jampacked_mcp_queries.json"
    with open(queries_path, 'w') as f:
        json.dump(queries, f, indent=2)
    
    print(f"\n📄 Example queries saved to: {queries_path}")


def verify_integration():
    """
    Verify that JamPacked tables exist in MCP database
    """
    import sqlite3
    
    db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for JamPacked tables
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'jampacked_%'
    """)
    
    tables = cursor.fetchall()
    
    print("\n✅ JamPacked tables in MCP database:")
    for table in tables:
        print(f"   - {table[0]}")
    
    conn.close()
    
    return len(tables) > 0


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           JamPacked + MCP SQLite Integration              ║
    ║                                                           ║
    ║  This will add JamPacked intelligence tables to your      ║
    ║  existing MCP SQLite database, allowing both Claude       ║
    ║  Desktop and Claude Code to access the same data.        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Run setup
    if setup_jampacked_mcp_integration():
        print("\n✅ Setup completed successfully!")
        
        # Verify
        if verify_integration():
            print("\n🎉 JamPacked is now integrated with your MCP SQLite server!")
            print("\nYou can now:")
            print("1. Run JamPacked analysis from Python")
            print("2. Query results in Claude Desktop using SQL")
            print("3. Access same data from Claude Code")
            print("\nNo duplicate infrastructure needed!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")