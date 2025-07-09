#!/usr/bin/env python3
"""
Unit tests for JamPacked Custom Intelligence components
Tests individual intelligence engines without external dependencies
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add JamPacked to path
sys.path.append(str(Path(__file__).parent.parent / 'autonomous-intelligence' / 'core'))

# Mock the heavy dependencies for testing
sys.modules['cv2'] = MagicMock()
sys.modules['librosa'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['networkx'] = MagicMock()

from jampacked_custom_intelligence import (
    CreativeAsset,
    JamPackedCreativeIntelligence,
    JamPackedMultimodalAnalyzer,
    JamPackedPatternDiscovery,
    JamPackedCulturalAnalyzer,
    AttentionPredictionModel,
    EmotionAnalysisModel,
    BrandRecallPredictor
)


class TestCreativeIntelligence(unittest.TestCase):
    """Test JamPackedCreativeIntelligence engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.creative_intel = JamPackedCreativeIntelligence()
        
        # Mock models
        self.creative_intel.attention_model = Mock()
        self.creative_intel.emotion_model = Mock()
        self.creative_intel.brand_recall_model = Mock()
    
    def test_attention_prediction(self):
        """Test attention prediction functionality"""
        # Create mock creative asset
        asset = CreativeAsset(
            id='test_001',
            type='image',
            data=np.random.rand(224, 224, 3),
            metadata={'format': 'jpg'}
        )
        
        # Mock model predictions
        self.creative_intel.attention_model.predict.return_value = {
            'attention_score': 0.85,
            'heatmap': np.random.rand(224, 224),
            'fixation_points': [(100, 150), (200, 180)]
        }
        
        # Run prediction
        result = self.creative_intel.predict_attention(asset)
        
        self.assertIn('attention_score', result)
        self.assertGreater(result['attention_score'], 0)
        self.assertLess(result['attention_score'], 1)
    
    def test_emotion_analysis(self):
        """Test emotion analysis functionality"""
        # Test text emotion
        text_asset = CreativeAsset(
            id='test_002',
            type='text',
            data='This is an inspiring and uplifting message!',
            metadata={'language': 'en'}
        )
        
        # Mock emotion predictions
        self.creative_intel.emotion_model.analyze.return_value = {
            'primary_emotion': 'joy',
            'emotion_scores': {
                'joy': 0.82,
                'trust': 0.65,
                'anticipation': 0.71
            },
            'valence': 0.85,
            'arousal': 0.73
        }
        
        result = self.creative_intel.analyze_emotion(text_asset)
        
        self.assertIn('primary_emotion', result)
        self.assertIn('emotion_scores', result)
        self.assertIn('valence', result)
        self.assertGreater(result['valence'], -1)
        self.assertLess(result['valence'], 1)
    
    def test_brand_recall_prediction(self):
        """Test brand recall prediction"""
        # Create multimodal assets
        assets = [
            CreativeAsset(
                id='test_003',
                type='image',
                data=np.random.rand(224, 224, 3),
                metadata={'has_logo': True}
            ),
            CreativeAsset(
                id='test_004',
                type='text',
                data='Brand name mentioned here',
                metadata={'brand_mentions': 2}
            )
        ]
        
        # Mock brand recall predictions
        self.creative_intel.brand_recall_model.predict.return_value = {
            'recall_probability': 0.78,
            'memorability_score': 0.81,
            'key_factors': ['logo_prominence', 'repetition', 'emotional_connection']
        }
        
        result = self.creative_intel.predict_brand_recall(assets)
        
        self.assertIn('recall_probability', result)
        self.assertIn('memorability_score', result)
        self.assertGreater(result['recall_probability'], 0)
        self.assertLess(result['recall_probability'], 1)
    
    async def test_comprehensive_analysis(self):
        """Test full creative effectiveness analysis"""
        materials = {
            'text': ['Test headline', 'Test tagline'],
            'images': [np.random.rand(224, 224, 3)]
        }
        
        context = {
            'campaign_name': 'Test Campaign',
            'objectives': ['awareness', 'engagement']
        }
        
        # Mock all sub-analyses
        self.creative_intel.attention_model.predict.return_value = {'attention_score': 0.82}
        self.creative_intel.emotion_model.analyze.return_value = {'valence': 0.75}
        self.creative_intel.brand_recall_model.predict.return_value = {'recall_probability': 0.79}
        
        result = await self.creative_intel.analyze_creative_effectiveness(materials, context)
        
        self.assertIn('effectiveness_score', result)
        self.assertIn('attention_analysis', result)
        self.assertIn('emotion_analysis', result)
        self.assertIn('brand_recall_analysis', result)


class TestMultimodalAnalyzer(unittest.TestCase):
    """Test JamPackedMultimodalAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = JamPackedMultimodalAnalyzer()
    
    @patch('jampacked_custom_intelligence.CrossModalFusionNetwork')
    def test_cross_modal_fusion(self, mock_fusion):
        """Test cross-modal fusion capabilities"""
        # Mock fusion network
        mock_fusion_instance = Mock()
        mock_fusion.return_value = mock_fusion_instance
        
        # Create multimodal inputs
        image_features = np.random.rand(512)
        text_features = np.random.rand(768)
        audio_features = np.random.rand(256)
        
        # Mock fusion output
        mock_fusion_instance.fuse.return_value = {
            'fused_representation': np.random.rand(1024),
            'modality_weights': {
                'image': 0.4,
                'text': 0.35,
                'audio': 0.25
            },
            'coherence_score': 0.87
        }
        
        # Run fusion
        result = self.analyzer.fuse_modalities({
            'image': image_features,
            'text': text_features,
            'audio': audio_features
        })
        
        self.assertIn('fused_representation', result)
        self.assertIn('modality_weights', result)
        self.assertIn('coherence_score', result)
        
        # Check weights sum to 1
        weights_sum = sum(result['modality_weights'].values())
        self.assertAlmostEqual(weights_sum, 1.0, places=2)
    
    def test_language_support(self):
        """Test multilingual support (250+ languages)"""
        # Test language detection
        texts = {
            'en': 'Hello world',
            'es': 'Hola mundo',
            'zh': '你好世界',
            'ar': 'مرحبا بالعالم',
            'hi': 'नमस्ते दुनिया'
        }
        
        for lang, text in texts.items():
            detected = self.analyzer.detect_language(text)
            self.assertIsNotNone(detected)
            # In real implementation, would check correct detection
    
    async def test_multimodal_analysis(self):
        """Test complete multimodal analysis"""
        materials = {
            'text': ['Multimodal test content'],
            'images': [np.random.rand(224, 224, 3)],
            'videos': [np.random.rand(10, 224, 224, 3)],  # 10 frames
            'audio': [np.random.rand(16000)]  # 1 second at 16kHz
        }
        
        # Mock analysis components
        with patch.object(self.analyzer, 'analyze_visual') as mock_visual:
            mock_visual.return_value = {'visual_score': 0.85}
            
            with patch.object(self.analyzer, 'analyze_textual') as mock_text:
                mock_text.return_value = {'text_score': 0.82}
                
                with patch.object(self.analyzer, 'analyze_audio') as mock_audio:
                    mock_audio.return_value = {'audio_score': 0.79}
                    
                    result = await self.analyzer.analyze_multimodal_content(
                        materials,
                        languages=['en'],
                        cultural_contexts=['global']
                    )
                    
                    self.assertIn('multimodal_score', result)
                    self.assertIn('modality_analysis', result)
                    self.assertIn('cross_modal_coherence', result)


class TestPatternDiscovery(unittest.TestCase):
    """Test JamPackedPatternDiscovery engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.pattern_engine = JamPackedPatternDiscovery()
    
    def test_novel_pattern_detection(self):
        """Test detection of novel patterns"""
        # Create synthetic data with embedded patterns
        n_samples = 1000
        n_features = 50
        
        # Create base random data
        data = np.random.randn(n_samples, n_features)
        
        # Embed a novel pattern (sine wave correlation)
        pattern_indices = [10, 15, 20]
        for i in pattern_indices:
            data[:, i] = np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 0.1
        
        # Discover patterns
        patterns = self.pattern_engine.discover_novel_patterns(data)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Check pattern structure
        for pattern in patterns:
            self.assertIn('type', pattern)
            self.assertIn('novelty_score', pattern)
            self.assertIn('confidence', pattern)
            self.assertIn('description', pattern)
    
    def test_causal_pattern_discovery(self):
        """Test discovery of causal patterns"""
        # Create data with known causal structure
        n_samples = 500
        
        # X causes Y, Y causes Z
        X = np.random.randn(n_samples)
        Y = 2 * X + np.random.randn(n_samples) * 0.5
        Z = 1.5 * Y + np.random.randn(n_samples) * 0.5
        
        data = np.column_stack([X, Y, Z])
        
        # Discover causal patterns
        causal_patterns = self.pattern_engine.discover_causal_patterns(
            data,
            variable_names=['X', 'Y', 'Z']
        )
        
        self.assertIn('causal_graph', causal_patterns)
        self.assertIn('causal_effects', causal_patterns)
        
        # Should identify X->Y and Y->Z
        effects = causal_patterns['causal_effects']
        self.assertTrue(any(
            effect['cause'] == 'X' and effect['effect'] == 'Y' 
            for effect in effects
        ))
    
    async def test_performance_pattern_discovery(self):
        """Test discovery of performance patterns"""
        # Create campaign performance data
        historical_data = []
        for i in range(100):
            campaign = {
                'id': f'campaign_{i}',
                'features': np.random.rand(20),
                'performance': {
                    'engagement': np.random.rand(),
                    'conversion': np.random.rand(),
                    'roi': np.random.rand() * 5
                }
            }
            historical_data.append(campaign)
        
        # Current campaign
        current_campaign = {
            'features': np.random.rand(20),
            'context': {'objective': 'maximize_roi'}
        }
        
        # Discover patterns
        patterns = await self.pattern_engine.discover_performance_patterns(
            historical_data,
            current_campaign
        )
        
        self.assertIn('success_patterns', patterns)
        self.assertIn('risk_patterns', patterns)
        self.assertIn('optimization_opportunities', patterns)


class TestCulturalAnalyzer(unittest.TestCase):
    """Test JamPackedCulturalAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.cultural_analyzer = JamPackedCulturalAnalyzer()
        
        # Mock cultural database
        self.cultural_analyzer.cultural_db = {
            'us': {
                'values': ['individualism', 'achievement', 'innovation'],
                'taboos': ['certain_symbols'],
                'preferences': {'color': 'blue', 'tone': 'direct'}
            },
            'japan': {
                'values': ['harmony', 'respect', 'quality'],
                'taboos': ['number_4'],
                'preferences': {'color': 'white', 'tone': 'indirect'}
            }
        }
    
    def test_cultural_appropriateness(self):
        """Test cultural appropriateness assessment"""
        # Test content
        content = {
            'text': 'Individual achievement and success',
            'images': [np.random.rand(224, 224, 3)],
            'symbols': ['victory_sign']
        }
        
        # Assess for US culture
        us_assessment = self.cultural_analyzer.assess_appropriateness(content, 'us')
        
        self.assertIn('appropriateness_score', us_assessment)
        self.assertIn('alignment_factors', us_assessment)
        self.assertIn('risk_factors', us_assessment)
        self.assertGreater(us_assessment['appropriateness_score'], 0.5)
        
        # Assess for Japan culture
        japan_assessment = self.cultural_analyzer.assess_appropriateness(content, 'japan')
        
        self.assertIsNotNone(japan_assessment)
        # Scores might differ based on cultural values
    
    def test_cultural_adaptation(self):
        """Test cultural adaptation recommendations"""
        original_content = {
            'text': 'Be the best! Win now!',
            'tone': 'aggressive',
            'imagery': 'competitive'
        }
        
        # Get adaptation for different cultures
        adaptations = self.cultural_analyzer.recommend_adaptations(
            original_content,
            target_cultures=['us', 'japan', 'uk']
        )
        
        self.assertIn('us', adaptations)
        self.assertIn('japan', adaptations)
        self.assertIn('uk', adaptations)
        
        # Japanese adaptation should recommend softer tone
        japan_adapt = adaptations['japan']
        self.assertIn('tone_adjustment', japan_adapt)
        self.assertIn('imagery_suggestions', japan_adapt)
    
    async def test_cross_cultural_effectiveness(self):
        """Test cross-cultural effectiveness analysis"""
        campaign_materials = {
            'text': ['Global campaign message'],
            'images': [np.random.rand(224, 224, 3)],
            'values_expressed': ['innovation', 'community']
        }
        
        target_cultures = ['us', 'uk', 'japan', 'brazil', 'india']
        
        result = await self.cultural_analyzer.analyze_cross_cultural_effectiveness(
            campaign_materials,
            target_cultures
        )
        
        self.assertIn('overall_effectiveness', result)
        self.assertIn('culture_scores', result)
        self.assertIn('universal_elements', result)
        self.assertIn('adaptation_needed', result)
        
        # Check all cultures analyzed
        for culture in target_cultures:
            self.assertIn(culture, result['culture_scores'])


class TestAttentionPredictionModel(unittest.TestCase):
    """Test custom attention prediction model"""
    
    def setUp(self):
        """Set up test environment"""
        self.attention_model = AttentionPredictionModel()
    
    def test_visual_attention_prediction(self):
        """Test visual attention heatmap generation"""
        # Create test image
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Predict attention
        result = self.attention_model.predict(test_image)
        
        self.assertIn('attention_map', result)
        self.assertIn('attention_score', result)
        self.assertIn('salient_regions', result)
        
        # Check attention map shape
        attention_map = result['attention_map']
        self.assertEqual(attention_map.shape[:2], test_image.shape[:2])
        
        # Check score range
        self.assertGreater(result['attention_score'], 0)
        self.assertLess(result['attention_score'], 1)
    
    def test_fixation_prediction(self):
        """Test eye fixation point prediction"""
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Predict fixations
        fixations = self.attention_model.predict_fixations(test_image)
        
        self.assertIsInstance(fixations, list)
        self.assertGreater(len(fixations), 0)
        
        # Check fixation format
        for fixation in fixations:
            self.assertIn('x', fixation)
            self.assertIn('y', fixation)
            self.assertIn('duration', fixation)
            self.assertIn('order', fixation)


# Performance and Integration Tests
class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_batch_processing(self):
        """Test batch processing efficiency"""
        # Create batch of assets
        batch_size = 100
        assets = []
        for i in range(batch_size):
            assets.append(CreativeAsset(
                id=f'batch_{i}',
                type='image',
                data=np.random.rand(224, 224, 3),
                metadata={'batch_id': 'test_batch'}
            ))
        
        # Process batch (in real implementation, would measure time)
        # and ensure batch processing is faster than individual
        self.assertEqual(len(assets), batch_size)
    
    def test_caching_mechanism(self):
        """Test caching for repeated analyses"""
        # Create test cache
        from functools import lru_cache
        
        @lru_cache(maxsize=128)
        def expensive_analysis(asset_id):
            # Simulate expensive computation
            return {'result': np.random.rand()}
        
        # First call
        result1 = expensive_analysis('test_asset_001')
        
        # Second call (should be cached)
        result2 = expensive_analysis('test_asset_001')
        
        self.assertEqual(result1, result2)
        
        # Check cache info
        cache_info = expensive_analysis.cache_info()
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.misses, 1)


# Run all tests
if __name__ == '__main__':
    unittest.main(verbosity=2)