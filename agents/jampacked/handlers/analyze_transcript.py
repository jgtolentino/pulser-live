# agents/jampacked/handlers/analyze_transcript.py
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

class YouTubeTranscriptAnalyzer:
    """
    JamPacked Creative Intelligence YouTube Transcript Analysis
    Integrates with existing WARC Effectiveness framework
    """
    
    def __init__(self, db_path="/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    def analyze_transcript(self, task_data):
        """
        Main analysis method that connects to your existing models
        """
        try:
            transcript = task_data.get('transcript', '')
            video_metadata = task_data.get('metadata', {})
            
            # Extract creative features using WARC framework
            creative_features = self.extract_warc_dimensions(transcript, video_metadata)
            
            # Apply your existing award prediction logic
            award_predictions = self.predict_awards(creative_features)
            
            # Analyze brand effectiveness
            brand_analysis = self.analyze_brand_effectiveness(transcript, video_metadata)
            
            # Cultural impact assessment
            cultural_impact = self.assess_cultural_impact(transcript)
            
            # Compile comprehensive analysis
            analysis_result = {
                'warc_dimensions': creative_features,
                'award_predictions': award_predictions,
                'brand_effectiveness': brand_analysis,
                'cultural_impact': cultural_impact,
                'video_metadata': video_metadata,
                'analysis_timestamp': datetime.now().isoformat(),
                'confidence_score': self.calculate_confidence(creative_features)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
    
    def extract_warc_dimensions(self, transcript, metadata):
        """
        Extract WARC Effective 100 dimensions from video content
        Maps to your existing creative_effectiveness_schema.sql
        """
        return {
            # Strategic Planning Rigor
            'strategic_planning': {
                'insight_strength': self.score_insight_strength(transcript),
                'objective_clarity': self.score_objective_clarity(transcript),
                'target_definition': self.score_target_definition(transcript),
                'strategy_originality': self.score_strategy_originality(transcript)
            },
            
            # Creative Idea Excellence  
            'creative_excellence': {
                'innovation_level': self.score_innovation(transcript, metadata),
                'emotional_resonance': self.score_emotional_impact(transcript),
                'craft_quality': self.score_production_craft(metadata),
                'memorability': self.score_memorability(transcript)
            },
            
            # Business Results Delivery
            'business_results': {
                'engagement_potential': self.predict_engagement(metadata),
                'conversion_likelihood': self.score_conversion_potential(transcript),
                'roi_projection': self.project_roi(transcript, metadata),
                'viral_potential': self.assess_viral_potential(transcript, metadata)
            },
            
            # Brand Building Impact
            'brand_building': {
                'brand_salience': self.score_brand_presence(transcript),
                'distinctive_assets': self.identify_brand_assets(transcript),
                'brand_consistency': self.score_brand_consistency(transcript),
                'memory_encoding': self.score_memory_encoding(transcript)
            },
            
            # Cultural & Social Value
            'cultural_impact': {
                'authenticity': self.score_authenticity(transcript),
                'cultural_relevance': self.score_cultural_relevance(transcript),
                'social_conversation': self.predict_social_conversation(transcript),
                'purpose_alignment': self.score_purpose_alignment(transcript)
            }
        }
    
    def predict_awards(self, creative_features):
        """
        Leverage your existing award prediction models
        """
        # Map WARC dimensions to award prediction features
        award_features = {
            'innovation_level': creative_features['creative_excellence']['innovation_level'],
            'emotional_resonance': creative_features['creative_excellence']['emotional_resonance'],
            'craft_excellence': creative_features['creative_excellence']['craft_quality'],
            'cultural_relevance': creative_features['cultural_impact']['cultural_relevance'],
            'brand_integration': creative_features['brand_building']['brand_salience'],
            'roi_potential': creative_features['business_results']['roi_projection']
        }
        
        # This would integrate with your existing award-prediction-model.js
        # For now, return structured prediction format
        return {
            'cannes_lions': {
                'probability': min(award_features['innovation_level'] * 0.9, 0.95),
                'predicted_level': self.map_probability_to_level(award_features['innovation_level']),
                'recommended_categories': ['Digital Craft', 'Film', 'Creative Strategy']
            },
            'effie_awards': {
                'probability': award_features['roi_potential'] * 0.85,
                'predicted_level': self.map_probability_to_effie_level(award_features['roi_potential']),
                'effectiveness_score': award_features['roi_potential']
            },
            'one_show': {
                'probability': (award_features['innovation_level'] + award_features['craft_excellence']) / 2,
                'predicted_level': self.map_probability_to_level((award_features['innovation_level'] + award_features['craft_excellence']) / 2)
            }
        }
    
    def analyze_brand_effectiveness(self, transcript, metadata):
        """
        Brand-specific effectiveness analysis
        """
        return {
            'brand_mentions': self.count_brand_mentions(transcript),
            'product_integration': self.score_product_integration(transcript),
            'call_to_action_strength': self.score_cta_strength(transcript),
            'brand_personality_expression': self.score_brand_personality(transcript),
            'distinctive_asset_usage': self.identify_distinctive_assets(transcript)
        }
    
    def assess_cultural_impact(self, transcript):
        """
        Cultural and social impact assessment
        """
        return {
            'trending_topics_alignment': self.check_trending_topics(transcript),
            'social_issues_engagement': self.score_social_engagement(transcript),
            'demographic_resonance': self.score_demographic_appeal(transcript),
            'conversation_starter_potential': self.score_conversation_potential(transcript)
        }
    
    # Scoring methods (implement based on NLP/ML approaches)
    def score_insight_strength(self, transcript):
        """Score the strength of consumer insight"""
        insight_keywords = ['because', 'insight', 'truth', 'human', 'behavior', 'need']
        score = sum(1 for word in insight_keywords if word in transcript.lower()) / len(insight_keywords)
        return min(score * 1.5, 1.0)
    
    def score_innovation(self, transcript, metadata):
        """Score innovation level"""
        innovation_indicators = ['first', 'new', 'never', 'breakthrough', 'revolutionary', 'unique']
        tech_indicators = ['AI', 'AR', 'VR', 'blockchain', 'NFT', 'metaverse']
        
        base_score = sum(1 for word in innovation_indicators if word.lower() in transcript.lower())
        tech_score = sum(1 for word in tech_indicators if word in transcript)
        
        # Factor in video metadata (resolution, format innovations)
        production_innovation = 0.2 if metadata.get('resolution', 1080) > 1080 else 0
        
        return min((base_score * 0.1 + tech_score * 0.15 + production_innovation), 1.0)
    
    def score_emotional_impact(self, transcript):
        """Score emotional resonance"""
        emotion_words = ['love', 'joy', 'fear', 'surprise', 'anger', 'sad', 'happy', 'excited']
        emotional_phrases = ['makes you feel', 'touches your heart', 'brings tears', 'makes you laugh']
        
        word_score = sum(1 for word in emotion_words if word in transcript.lower()) / 20
        phrase_score = sum(1 for phrase in emotional_phrases if phrase in transcript.lower()) / 4
        
        return min(word_score + phrase_score, 1.0)
    
    def predict_engagement(self, metadata):
        """Predict engagement potential based on video characteristics"""
        duration = metadata.get('duration', 30)
        
        # Optimal duration for engagement (15-60 seconds for social)
        duration_score = 1.0 if 15 <= duration <= 60 else max(0.3, 1.0 - abs(duration - 37.5) / 100)
        
        # Factor in other metadata
        quality_score = 0.8 if metadata.get('resolution', 720) >= 1080 else 0.5
        
        return min(duration_score * 0.7 + quality_score * 0.3, 1.0)
    
    def calculate_confidence(self, creative_features):
        """Calculate overall confidence in analysis"""
        dimension_scores = []
        for dimension in creative_features.values():
            if isinstance(dimension, dict):
                scores = [v for v in dimension.values() if isinstance(v, (int, float))]
                if scores:
                    dimension_scores.append(sum(scores) / len(scores))
        
        if dimension_scores:
            return sum(dimension_scores) / len(dimension_scores)
        return 0.7  # Default confidence
    
    # Helper methods for mapping scores
    def map_probability_to_level(self, prob):
        if prob > 0.85: return 'Gold'
        if prob > 0.75: return 'Silver' 
        if prob > 0.65: return 'Bronze'
        if prob > 0.5: return 'Shortlist'
        return 'No Award'
    
    def map_probability_to_effie_level(self, prob):
        if prob > 0.85: return 'Grand Effie'
        if prob > 0.75: return 'Gold Effie'
        if prob > 0.65: return 'Silver Effie'
        if prob > 0.55: return 'Bronze Effie'
        return 'No Award'
    
    # Additional scoring methods would be implemented here...
    def count_brand_mentions(self, transcript): return transcript.lower().count('brand') * 0.1
    def score_product_integration(self, transcript): return 0.7  # Placeholder
    def score_cta_strength(self, transcript): return 0.6  # Placeholder
    def score_brand_personality(self, transcript): return 0.8  # Placeholder
    def identify_distinctive_assets(self, transcript): return []  # Placeholder
    def check_trending_topics(self, transcript): return 0.5  # Placeholder
    def score_social_engagement(self, transcript): return 0.6  # Placeholder
    def score_demographic_appeal(self, transcript): return 0.7  # Placeholder
    def score_conversation_potential(self, transcript): return 0.5  # Placeholder
    def score_objective_clarity(self, transcript): return 0.7  # Placeholder
    def score_target_definition(self, transcript): return 0.6  # Placeholder
    def score_strategy_originality(self, transcript): return 0.8  # Placeholder
    def score_production_craft(self, metadata): return 0.8  # Placeholder
    def score_memorability(self, transcript): return 0.7  # Placeholder
    def score_conversion_potential(self, transcript): return 0.6  # Placeholder
    def project_roi(self, transcript, metadata): return 2.5  # Placeholder
    def assess_viral_potential(self, transcript, metadata): return 0.5  # Placeholder
    def score_brand_presence(self, transcript): return 0.7  # Placeholder
    def identify_brand_assets(self, transcript): return []  # Placeholder
    def score_brand_consistency(self, transcript): return 0.8  # Placeholder
    def score_memory_encoding(self, transcript): return 0.7  # Placeholder
    def score_authenticity(self, transcript): return 0.8  # Placeholder
    def score_cultural_relevance(self, transcript): return 0.7  # Placeholder
    def predict_social_conversation(self, transcript): return 0.6  # Placeholder
    def score_purpose_alignment(self, transcript): return 0.7  # Placeholder

# Task runner integration
def run_task_runner():
    """
    Polls SQLite queue for transcript analysis tasks
    Integrates with your existing task queue system
    """
    db_path = "/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    analyzer = YouTubeTranscriptAnalyzer(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    while True:
        # Poll for transcript analysis tasks
        cursor.execute("""
            SELECT task_id, payload FROM agent_task_queue 
            WHERE target_agent = 'jampacked' 
            AND task_type = 'analyze_transcript' 
            AND status = 'pending'
            LIMIT 1
        """)
        
        task = cursor.fetchone()
        if task:
            task_id, payload = task
            task_data = json.loads(payload)
            
            # Update status to in_progress
            cursor.execute("""
                UPDATE agent_task_queue 
                SET status = 'in_progress', updated_at = ?
                WHERE task_id = ?
            """, (datetime.now().isoformat(), task_id))
            conn.commit()
            
            try:
                # Run analysis
                result = analyzer.analyze_transcript(task_data)
                
                # Update with results
                cursor.execute("""
                    UPDATE agent_task_queue 
                    SET status = 'completed', result = ?, updated_at = ?
                    WHERE task_id = ?
                """, (json.dumps(result), datetime.now().isoformat(), task_id))
                
                print(f"✅ Completed analysis for task {task_id}")
                
            except Exception as e:
                # Update with error
                cursor.execute("""
                    UPDATE agent_task_queue 
                    SET status = 'failed', result = ?, updated_at = ?
                    WHERE task_id = ?
                """, (json.dumps({'error': str(e)}), datetime.now().isoformat(), task_id))
                
                print(f"❌ Failed analysis for task {task_id}: {e}")
            
            conn.commit()
        
        import time
        time.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    run_task_runner()