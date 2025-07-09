#!/usr/bin/env python3
"""
Integration tests for JamPacked Creative Intelligence Suite
Tests all autonomous capabilities and MCP integration
"""

import asyncio
import unittest
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

# Add JamPacked to path
sys.path.append(str(Path(__file__).parent.parent / 'autonomous-intelligence' / 'core'))

from jampacked_custom_intelligence import JamPackedIntelligenceSuite
from jampacked_sqlite_integration import JamPackedSQLiteIntegration, analyze_campaign_via_mcp
from autonomous_jampacked import AutonomousJamPacked
from true_autonomous_jampacked import JamPackedAutonomousIntelligence


class TestJamPackedIntegration(unittest.TestCase):
    """Test suite for JamPacked integration with MCP SQLite"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test workspace
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test_database.sqlite"
        
        # Initialize integration with test database
        self.integration = JamPackedSQLiteIntegration(db_path=str(self.test_db_path))
        
        # Sample test data
        self.sample_materials = {
            'text': ['Test campaign headline', 'Test campaign tagline'],
            'images': [np.random.rand(224, 224, 3).astype(np.uint8)]  # Mock image data
        }
        
        self.sample_context = {
            'campaign_name': 'Test Campaign Q1 2024',
            'target_cultures': ['us', 'uk'],
            'business_objectives': ['brand_awareness', 'engagement']
        }
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_table_creation(self):
        """Test that JamPacked tables are created correctly"""
        conn = sqlite3.connect(str(self.test_db_path))
        cursor = conn.cursor()
        
        # Check for all required tables
        expected_tables = [
            'jampacked_creative_analysis',
            'jampacked_pattern_discoveries',
            'jampacked_cultural_insights',
            'jampacked_optimizations',
            'jampacked_sessions'
        ]
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE 'jampacked_%'
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} not found")
    
    @patch('jampacked_custom_intelligence.JamPackedIntelligenceSuite')
    async def test_analyze_and_store(self, mock_suite):
        """Test campaign analysis and storage"""
        # Mock the analysis results
        mock_instance = MagicMock()
        mock_suite.return_value = mock_instance
        
        mock_instance.analyze_campaign_materials.return_value = {
            'overall_scores': {
                'effectiveness': 0.85,
                'attention': 0.82,
                'emotion': 0.78,
                'brand_recall': 0.79,
                'cultural_alignment': 0.88,
                'multimodal': 0.84
            },
            'detailed_analysis': {
                'discovered_patterns': {
                    'novel_discoveries': [
                        {
                            'type': 'visual_emotional',
                            'description': 'Color palette triggers positive response',
                            'novelty_score': 0.92,
                            'confidence': 0.87,
                            'business_impact': 0.78
                        }
                    ]
                },
                'cultural_insights': {
                    'individual_cultures': {
                        'us': {
                            'cultural_effectiveness_score': 0.89,
                            'appropriateness_assessment': {'score': 0.91},
                            'adaptation_recommendations': ['Increase local references'],
                            'risk_assessment': {'level': 'low'}
                        }
                    }
                }
            },
            'actionable_recommendations': [
                {
                    'type': 'creative',
                    'title': 'Optimize color contrast',
                    'impact': 0.15,
                    'effort': 'low',
                    'priority': 0.85,
                    'ab_test_plan': {'variants': 2}
                }
            ]
        }
        
        # Run analysis
        result = await self.integration.analyze_and_store(
            self.sample_materials,
            self.sample_context
        )
        
        # Verify results
        self.assertIn('campaign_id', result)
        self.assertIn('session_id', result)
        self.assertEqual(result['storage_status'], 'success')
        self.assertTrue(result['access_via_mcp'])
        
        # Verify data was stored
        conn = sqlite3.connect(str(self.test_db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM jampacked_creative_analysis WHERE campaign_id = ?", 
                      (result['campaign_id'],))
        count = cursor.fetchone()[0]
        self.assertEqual(count, 1)
        
        conn.close()
    
    def test_session_management(self):
        """Test session creation and tracking"""
        # Create session
        session_id = self.integration.create_session(
            campaign_id='test_campaign_123',
            session_type='analysis',
            interface='python_api'
        )
        
        self.assertIsNotNone(session_id)
        
        # Verify session was created
        conn = sqlite3.connect(str(self.test_db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jampacked_sessions WHERE session_id = ?", (session_id,))
        session = cursor.fetchone()
        
        self.assertIsNotNone(session)
        conn.close()
    
    def test_query_generation(self):
        """Test SQL query generation for Claude"""
        queries = self.integration.get_useful_queries('test_campaign_123')
        
        expected_keys = [
            'get_latest_analysis',
            'get_top_patterns',
            'get_cultural_insights',
            'get_high_priority_optimizations',
            'get_all_campaigns',
            'get_session_history'
        ]
        
        for key in expected_keys:
            self.assertIn(key, queries)
            self.assertIn('test_campaign_123', queries[key])


class TestAutonomousCapabilities(unittest.TestCase):
    """Test suite for autonomous intelligence features"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.autonomous = AutonomousJamPacked(workspace_root=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    @patch('autonomous_jampacked.torch.load')
    @patch('autonomous_jampacked.torch.save')
    async def test_pattern_discovery(self, mock_save, mock_load):
        """Test autonomous pattern discovery"""
        # Mock loading existing patterns
        mock_load.side_effect = FileNotFoundError()
        
        # Create test data
        test_features = np.random.rand(100, 50)
        test_labels = np.random.randint(0, 3, 100)
        
        # Run pattern discovery
        patterns = await self.autonomous.pattern_discoverer.discover_patterns(
            test_features,
            test_labels
        )
        
        # Verify patterns were discovered
        self.assertIsInstance(patterns, dict)
        self.assertIn('clusters', patterns)
        self.assertIn('novel_patterns', patterns)
        
        # Verify patterns were saved
        mock_save.assert_called()
    
    async def test_meta_learning(self):
        """Test meta-learning adaptation"""
        # Create test task
        test_task = {
            'task_id': 'test_001',
            'features': np.random.rand(50, 20),
            'labels': np.random.randint(0, 2, 50)
        }
        
        # Test strategy selection
        strategy = await self.autonomous.meta_learner.select_strategy(test_task)
        
        self.assertIsNotNone(strategy)
        self.assertIn('algorithm', strategy)
        self.assertIn('parameters', strategy)
    
    async def test_causal_discovery(self):
        """Test causal relationship discovery"""
        # Create test data with known causal structure
        n_samples = 200
        x = np.random.randn(n_samples)
        y = 2 * x + np.random.randn(n_samples) * 0.1
        z = x + y + np.random.randn(n_samples) * 0.1
        
        data = np.column_stack([x, y, z])
        
        # Discover causal relationships
        causal_graph = await self.autonomous.causal_discoverer.discover_causal_structure(
            data,
            variable_names=['x', 'y', 'z']
        )
        
        # Verify causal structure
        self.assertIsNotNone(causal_graph)
        self.assertIn('edges', causal_graph)
        self.assertIn('nodes', causal_graph)
        
        # Should discover x->y and y->z relationships
        edges = causal_graph['edges']
        self.assertTrue(any(edge['from'] == 'x' and edge['to'] == 'y' for edge in edges))


class TestEvolutionaryLearning(unittest.TestCase):
    """Test suite for evolutionary learning capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        # Import evolutionary learning module
        sys.path.append(str(Path(__file__).parent.parent / 'engines' / 'evolutionary'))
        from evolutionary_learning import EvolutionaryLearningEngine
        self.evolutionary = EvolutionaryLearningEngine()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_neural_evolution(self):
        """Test neural architecture evolution"""
        # Create initial population
        population_size = 10
        population = self.evolutionary.neural_evolution.create_initial_population(
            population_size,
            input_dim=10,
            output_dim=2
        )
        
        self.assertEqual(len(population), population_size)
        
        # Test mutation
        mutated = self.evolutionary.neural_evolution.mutate(population[0])
        self.assertIsNotNone(mutated)
        
        # Test crossover
        offspring = self.evolutionary.neural_evolution.crossover(
            population[0], 
            population[1]
        )
        self.assertIsNotNone(offspring)
    
    def test_pattern_evolution(self):
        """Test pattern evolution capabilities"""
        # Create test patterns
        initial_patterns = [
            {'type': 'visual', 'score': 0.7},
            {'type': 'emotional', 'score': 0.8},
            {'type': 'cultural', 'score': 0.6}
        ]
        
        # Evolve patterns
        evolved = self.evolutionary.evolve_patterns(
            initial_patterns,
            generations=5,
            fitness_function=lambda p: p.get('score', 0)
        )
        
        self.assertIsNotNone(evolved)
        self.assertGreaterEqual(len(evolved), len(initial_patterns))


class TestEnhancedCRISPDM(unittest.TestCase):
    """Test suite for Enhanced CRISP-DM with AI autonomy"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        sys.path.append(str(Path(__file__).parent.parent / 'autonomous-intelligence' / 'core'))
        from enhanced_crisp_dm import EnhancedCRISPDM
        self.crisp_dm = EnhancedCRISPDM(workspace_root=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    async def test_autonomous_pipeline(self):
        """Test autonomous CRISP-DM pipeline execution"""
        # Create test project
        project_config = {
            'name': 'Test Creative Campaign',
            'data_sources': ['test_data.csv'],
            'objectives': ['maximize_engagement'],
            'constraints': {'budget': 10000}
        }
        
        # Mock data understanding phase
        with patch.object(self.crisp_dm, 'data_understanding') as mock_understanding:
            mock_understanding.return_value = {
                'data_quality': 0.85,
                'feature_importance': {'color': 0.9, 'text': 0.7},
                'recommendations': ['Clean missing values']
            }
            
            # Run autonomous pipeline
            results = await self.crisp_dm.run_autonomous_pipeline(project_config)
            
            self.assertIsNotNone(results)
            self.assertIn('business_understanding', results)
            self.assertIn('final_model', results)
            self.assertIn('deployment_plan', results)
    
    def test_self_modification(self):
        """Test CRISP-DM self-modification capabilities"""
        # Test pipeline optimization
        initial_config = {
            'phases': ['business', 'data', 'preparation', 'modeling', 'evaluation'],
            'parameters': {'modeling_iterations': 10}
        }
        
        # Simulate performance feedback
        feedback = {
            'accuracy': 0.75,
            'runtime': 120,
            'resource_usage': 0.8
        }
        
        # Apply self-modification
        optimized_config = self.crisp_dm.self_modify(initial_config, feedback)
        
        self.assertIsNotNone(optimized_config)
        self.assertNotEqual(initial_config, optimized_config)


class TestMCPIntegrationEndToEnd(unittest.TestCase):
    """End-to-end test for MCP integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.test_dir) / "test_database.sqlite"
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    @patch('jampacked_sqlite_integration.JamPackedIntelligenceSuite')
    async def test_full_workflow(self, mock_suite):
        """Test complete workflow from analysis to query"""
        # Mock the intelligence suite
        mock_instance = MagicMock()
        mock_suite.return_value = mock_instance
        
        # Mock comprehensive analysis results
        mock_instance.analyze_campaign_materials.return_value = {
            'overall_scores': {
                'effectiveness': 0.87,
                'attention': 0.84,
                'emotion': 0.89,
                'brand_recall': 0.82,
                'cultural_alignment': 0.91,
                'multimodal': 0.86
            },
            'detailed_analysis': {
                'discovered_patterns': {
                    'novel_discoveries': [
                        {
                            'type': 'cross_cultural_emotion',
                            'description': 'Unified emotional response across cultures',
                            'novelty_score': 0.94,
                            'confidence': 0.91,
                            'business_impact': 0.88
                        }
                    ]
                },
                'cultural_insights': {
                    'individual_cultures': {
                        'us': {
                            'cultural_effectiveness_score': 0.92,
                            'appropriateness_assessment': {'score': 0.95}
                        },
                        'uk': {
                            'cultural_effectiveness_score': 0.89,
                            'appropriateness_assessment': {'score': 0.93}
                        }
                    }
                }
            },
            'actionable_recommendations': [
                {
                    'type': 'creative',
                    'title': 'Enhance visual storytelling',
                    'impact': 0.25,
                    'effort': 'medium',
                    'priority': 0.92
                }
            ]
        }
        
        # 1. Analyze campaign
        materials = {
            'text': ['Revolutionary product launch', 'Innovation meets tradition'],
            'images': [np.zeros((224, 224, 3), dtype=np.uint8)]
        }
        
        context = {
            'campaign_name': 'Innovation Campaign 2024',
            'target_cultures': ['us', 'uk'],
            'business_objectives': ['market_penetration', 'brand_evolution']
        }
        
        # Use the actual integration function
        result = await analyze_campaign_via_mcp(materials, context)
        
        # 2. Verify storage
        self.assertEqual(result['storage_status'], 'success')
        campaign_id = result['campaign_id']
        
        # 3. Query results (simulating Claude Desktop access)
        conn = sqlite3.connect(str(self.test_db_path))
        cursor = conn.cursor()
        
        # Get analysis
        cursor.execute("""
            SELECT creative_effectiveness_score, recommendations 
            FROM jampacked_creative_analysis 
            WHERE campaign_id = ?
        """, (campaign_id,))
        
        analysis = cursor.fetchone()
        self.assertIsNotNone(analysis)
        
        # Get patterns
        cursor.execute("""
            SELECT COUNT(*) FROM jampacked_pattern_discoveries 
            WHERE campaign_id = ? AND novelty_score > 0.9
        """, (campaign_id,))
        
        pattern_count = cursor.fetchone()[0]
        self.assertGreater(pattern_count, 0)
        
        conn.close()


# Run tests
if __name__ == '__main__':
    # Run async tests
    unittest.main(verbosity=2)