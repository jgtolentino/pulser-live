#!/usr/bin/env python3
"""
Example: Using JamPacked with Claude Desktop and Claude Code Integration

This example demonstrates the seamless workflow between Claude Code (Python)
and Claude Desktop (SQL) through the MCP SQLite integration.
"""

import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

# Import JamPacked integration
from jampacked_sqlite_integration import (
    analyze_campaign_via_mcp,
    get_campaign_results_via_mcp,
    JamPackedSQLiteIntegration
)


async def example_complete_workflow():
    """
    Complete example workflow showing Claude Code and Desktop integration
    """
    
    print("🚀 JamPacked Creative Intelligence - Claude Integration Example")
    print("=" * 60)
    
    # Step 1: Prepare campaign materials (Claude Code)
    print("\n📊 Step 1: Preparing campaign materials in Claude Code...")
    
    campaign_materials = {
        'text': [
            'Revolutionary AI-Powered Creative Intelligence',
            'Transform your creative effectiveness with data-driven insights',
            'Join the future of advertising analytics'
        ],
        'images': [
            np.random.rand(224, 224, 3),  # Placeholder for actual image
            np.random.rand(224, 224, 3)
        ],
        'metadata': {
            'brand': 'JamPacked Intelligence',
            'industry': 'Marketing Technology',
            'launch_date': '2024-01-15'
        }
    }
    
    campaign_context = {
        'campaign_name': 'JamPacked Q1 2024 Launch',
        'target_cultures': ['us', 'uk', 'japan', 'brazil'],
        'business_objectives': ['brand_awareness', 'lead_generation', 'thought_leadership'],
        'budget_tier': 'premium',
        'expected_awards': ['Cannes Lions', 'D&AD', 'One Show'],
        'csr_focus': 'democratizing_ai_access'
    }
    
    # Step 2: Run analysis (Claude Code)
    print("\n🔍 Step 2: Running JamPacked analysis...")
    
    results = await analyze_campaign_via_mcp(campaign_materials, campaign_context)
    
    campaign_id = results['campaign_id']
    print(f"✅ Analysis complete! Campaign ID: {campaign_id}")
    
    # Step 3: Display results that Claude Desktop can query
    print("\n📝 Step 3: Results now available in Claude Desktop via SQL:")
    print("-" * 60)
    
    print("Example SQL queries you can run in Claude Desktop:")
    print()
    
    # Query 1: Overall effectiveness
    print("-- 1. Get overall campaign effectiveness")
    print(f"""SELECT 
    campaign_name,
    creative_effectiveness_score,
    attention_score,
    emotion_score,
    brand_recall_score,
    award_prestige_score,
    csr_authenticity_score
FROM jampacked_creative_analysis
WHERE campaign_id = '{campaign_id}';""")
    print()
    
    # Query 2: Pattern discoveries
    print("-- 2. Find novel patterns discovered")
    print(f"""SELECT 
    pattern_type,
    pattern_description,
    novelty_score,
    business_impact
FROM jampacked_pattern_discoveries
WHERE campaign_id = '{campaign_id}'
AND novelty_score > 0.8
ORDER BY novelty_score DESC;""")
    print()
    
    # Query 3: Cultural insights
    print("-- 3. Get cultural adaptation recommendations")
    print(f"""SELECT 
    culture,
    effectiveness_score,
    adaptation_recommendations
FROM jampacked_cultural_insights
WHERE campaign_id = '{campaign_id}'
ORDER BY effectiveness_score DESC;""")
    print()
    
    # Query 4: Optimization opportunities
    print("-- 4. Find high-impact optimizations")
    print(f"""SELECT 
    optimization_type,
    description,
    predicted_impact,
    priority_score
FROM jampacked_optimizations
WHERE campaign_id = '{campaign_id}'
AND predicted_impact > 0.2
ORDER BY priority_score DESC;""")
    
    print("\n" + "-" * 60)
    
    # Step 4: Demonstrate programmatic access (Claude Code)
    print("\n💻 Step 4: Accessing results programmatically in Claude Code...")
    
    # Get results using Python API
    campaign_results = get_campaign_results_via_mcp(campaign_id)
    
    if campaign_results['analysis']:
        analysis = campaign_results['analysis']['data']
        print(f"\nCreative Effectiveness Score: {analysis[5]:.2f}")  # Assuming score is at index 5
        
    print(f"Patterns Discovered: {len(campaign_results['patterns'])}")
    print(f"Cultural Insights: {len(campaign_results['cultural_insights'])}")
    print(f"Optimizations Suggested: {len(campaign_results['optimizations'])}")
    
    # Step 5: Advanced integration example
    print("\n🔗 Step 5: Advanced Integration Example...")
    
    # Create a session for cross-interface work
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Created session: {session_id}")
    print("\nYou can now:")
    print("1. Continue this analysis in Claude Desktop using the session ID")
    print("2. Run additional analyses in Claude Code")
    print("3. Generate reports combining both interfaces")
    
    # Step 6: Show award and CSR analysis
    print("\n🏆 Step 6: Award and CSR Analysis Results...")
    
    print("\n-- Award Recognition Analysis")
    print(f"""SELECT 
    award_prestige_score,
    award_level_highest,
    award_category_diversity,
    award_recency_weight
FROM jampacked_creative_analysis
WHERE campaign_id = '{campaign_id}';""")
    
    print("\n-- CSR Authenticity Analysis")
    print(f"""SELECT 
    csr_presence_binary,
    csr_category,
    csr_message_prominence,
    csr_authenticity_score,
    csr_audience_alignment
FROM jampacked_creative_analysis
WHERE campaign_id = '{campaign_id}';""")
    
    return campaign_id


async def example_batch_analysis():
    """
    Example of batch processing multiple campaigns
    """
    print("\n📦 Batch Analysis Example")
    print("=" * 60)
    
    # Prepare multiple campaigns
    campaigns = [
        {
            'name': 'Tech Product Launch',
            'materials': {
                'text': ['Innovative technology for everyone'],
                'images': [np.random.rand(224, 224, 3)]
            },
            'context': {
                'campaign_name': 'Tech Launch 2024',
                'target_cultures': ['us', 'eu'],
                'business_objectives': ['product_launch']
            }
        },
        {
            'name': 'Sustainability Campaign',
            'materials': {
                'text': ['Building a greener tomorrow together'],
                'images': [np.random.rand(224, 224, 3)]
            },
            'context': {
                'campaign_name': 'Green Future 2024',
                'target_cultures': ['global'],
                'business_objectives': ['csr_awareness'],
                'csr_focus': 'environmental_sustainability'
            }
        }
    ]
    
    # Process all campaigns
    campaign_ids = []
    for campaign in campaigns:
        print(f"\nAnalyzing: {campaign['name']}")
        results = await analyze_campaign_via_mcp(
            campaign['materials'],
            campaign['context']
        )
        campaign_ids.append(results['campaign_id'])
        print(f"✅ Complete: {results['campaign_id']}")
    
    # Show batch query
    print("\n📊 Batch Results Query for Claude Desktop:")
    print(f"""
-- Compare all campaigns analyzed today
SELECT 
    campaign_name,
    creative_effectiveness_score,
    award_prestige_score,
    csr_authenticity_score,
    (SELECT COUNT(*) FROM jampacked_pattern_discoveries pd 
     WHERE pd.campaign_id = ca.campaign_id) as patterns_found,
    (SELECT COUNT(*) FROM jampacked_optimizations o 
     WHERE o.campaign_id = ca.campaign_id) as optimizations_available
FROM jampacked_creative_analysis ca
WHERE campaign_id IN ({','.join([f"'{cid}'" for cid in campaign_ids])})
ORDER BY creative_effectiveness_score DESC;
    """)
    
    return campaign_ids


def show_integration_benefits():
    """
    Display key benefits of Claude integration
    """
    print("\n✨ Integration Benefits")
    print("=" * 60)
    
    benefits = {
        "Seamless Workflow": [
            "Start analysis in Claude Code with Python",
            "Query results instantly in Claude Desktop with SQL",
            "No data export/import needed"
        ],
        "Real-time Collaboration": [
            "Results available immediately across interfaces",
            "Share campaign IDs between team members",
            "Consistent analysis framework"
        ],
        "Best Tool for Each Task": [
            "Claude Code: Complex ML analysis, automation",
            "Claude Desktop: Business queries, reporting",
            "Both: Access to same 200+ variables"
        ],
        "No Infrastructure Overhead": [
            "Uses existing MCP SQLite server",
            "No additional databases needed",
            "Unified backup and security"
        ]
    }
    
    for category, items in benefits.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ✓ {item}")


async def main():
    """
    Run the complete integration example
    """
    print("\n" + "🎯 " * 20)
    print("JamPacked Creative Intelligence")
    print("Claude Desktop + Claude Code Integration Demo")
    print("🎯 " * 20)
    
    # Run examples
    campaign_id = await example_complete_workflow()
    
    # Show batch processing
    batch_ids = await example_batch_analysis()
    
    # Display benefits
    show_integration_benefits()
    
    # Final message
    print("\n" + "=" * 60)
    print("🎉 Integration demo complete!")
    print(f"\nAnalyzed campaigns are now available in both:")
    print("- Claude Desktop (via SQL queries)")
    print("- Claude Code (via Python API)")
    print("\nTry the SQL queries above in Claude Desktop to see the results!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if integration is properly set up
    try:
        integration = JamPackedSQLiteIntegration()
        print("✅ JamPacked MCP Integration is properly configured!")
        
        # Run the example
        asyncio.run(main())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. MCP SQLite server is installed")
        print("2. Run: python setup_mcp_integration.py")
        print("3. Database path is correct in configuration")