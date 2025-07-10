const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs').promises;
const sqlite3 = require('sqlite3').verbose();

// Import JamPacked modules
const { analyzeWithJampacked } = require('../mcp-integration/analyze-with-jampacked-handler');
const { AwardPredictionModel } = require('../models/award-prediction-model');
const { CSRAuthenticityScorer } = require('../models/csr-authenticity-scorer');
const { PatternDiscoveryEngine } = require('./pattern-discovery-engine');

/**
 * JamPacked Creative Intelligence API Server
 */
class JamPackedAPI {
  constructor(port = 3001) {
    this.app = express();
    this.port = port;
    this.upload = multer({ dest: 'uploads/' });
    this.db = null;
    
    // Initialize modules
    this.awardPredictor = new AwardPredictionModel();
    this.csrScorer = new CSRAuthenticityScorer();
    this.patternEngine = new PatternDiscoveryEngine();
    
    this.setupMiddleware();
    this.setupRoutes();
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
    
    // Request logging
    this.app.use((req, res, next) => {
      console.log(`📥 ${req.method} ${req.path} - ${new Date().toISOString()}`);
      next();
    });
  }

  /**
   * Setup API routes
   */
  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'healthy', 
        service: 'JamPacked Creative Intelligence API',
        version: '1.0.0',
        timestamp: new Date().toISOString()
      });
    });

    // Main analysis endpoint
    this.app.post('/api/analyze', this.upload.array('creatives', 10), async (req, res) => {
      try {
        const analysisId = `analysis_${uuidv4()}`;
        const { campaign_name, client, brand, target_cultures, business_objectives, csr_focus } = req.body;
        
        // Process uploaded files
        const creativeAssets = await this.processUploadedFiles(req.files);
        
        // Create campaign context
        const campaignContext = {
          campaign_id: `campaign_${uuidv4()}`,
          campaign_name,
          client,
          brand,
          target_cultures: JSON.parse(target_cultures || '[]'),
          business_objectives: JSON.parse(business_objectives || '[]'),
          csr_focus: csr_focus || null,
          analysis_id: analysisId
        };

        // Run comprehensive analysis
        const results = await this.runComprehensiveAnalysis(creativeAssets, campaignContext);
        
        res.json({
          success: true,
          analysis_id: analysisId,
          results
        });
        
      } catch (error) {
        console.error('❌ Analysis error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // Pattern discovery endpoint
    this.app.post('/api/patterns/discover', async (req, res) => {
      try {
        const { campaign_id, historical_data } = req.body;
        
        const patterns = await this.patternEngine.discoverPatterns({
          campaign_id,
          historical_data: historical_data || []
        });
        
        res.json({
          success: true,
          patterns
        });
        
      } catch (error) {
        console.error('❌ Pattern discovery error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // Award prediction endpoint
    this.app.post('/api/predict-awards', async (req, res) => {
      try {
        const { campaign_data, creative_features } = req.body;
        
        const predictions = await this.awardPredictor.predictAwards({
          campaign_data,
          creative_features
        });
        
        res.json({
          success: true,
          predictions
        });
        
      } catch (error) {
        console.error('❌ Award prediction error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // CSR authenticity scoring endpoint
    this.app.post('/api/csr/score', async (req, res) => {
      try {
        const { campaign_id, csr_content, brand_history } = req.body;
        
        const csrScore = await this.csrScorer.scoreAuthenticity({
          campaign_id,
          csr_content,
          brand_history: brand_history || {}
        });
        
        res.json({
          success: true,
          csr_analysis: csrScore
        });
        
      } catch (error) {
        console.error('❌ CSR scoring error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // Optimization recommendations endpoint
    this.app.post('/api/optimize', async (req, res) => {
      try {
        const { campaign_id, current_scores, business_objectives } = req.body;
        
        const optimizations = await this.generateOptimizations({
          campaign_id,
          current_scores,
          business_objectives
        });
        
        res.json({
          success: true,
          optimizations
        });
        
      } catch (error) {
        console.error('❌ Optimization error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // Get analysis results
    this.app.get('/api/analysis/:analysisId', async (req, res) => {
      try {
        const { analysisId } = req.params;
        
        const results = await this.getAnalysisResults(analysisId);
        
        if (!results) {
          return res.status(404).json({ 
            success: false, 
            error: 'Analysis not found' 
          });
        }
        
        res.json({
          success: true,
          results
        });
        
      } catch (error) {
        console.error('❌ Get analysis error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });

    // List all analyses
    this.app.get('/api/analyses', async (req, res) => {
      try {
        const { limit = 50, offset = 0 } = req.query;
        
        const analyses = await this.listAnalyses(limit, offset);
        
        res.json({
          success: true,
          analyses
        });
        
      } catch (error) {
        console.error('❌ List analyses error:', error);
        res.status(500).json({ 
          success: false, 
          error: error.message 
        });
      }
    });
  }

  /**
   * Process uploaded creative files
   */
  async processUploadedFiles(files) {
    const assets = [];
    
    for (const file of files) {
      const fileData = await fs.readFile(file.path);
      const asset = {
        id: uuidv4(),
        filename: file.originalname,
        mimetype: file.mimetype,
        size: file.size,
        data: fileData.toString('base64'),
        type: this.detectAssetType(file.mimetype)
      };
      
      assets.push(asset);
      
      // Clean up uploaded file
      await fs.unlink(file.path);
    }
    
    return assets;
  }

  /**
   * Run comprehensive analysis
   */
  async runComprehensiveAnalysis(assets, context) {
    console.log(`🔍 Running comprehensive analysis for ${context.campaign_name}`);
    
    // 1. Basic creative effectiveness
    const effectiveness = await this.analyzeCreativeEffectiveness(assets, context);
    
    // 2. Award prediction
    const awardPredictions = await this.awardPredictor.predictAwards({
      campaign_data: context,
      creative_features: effectiveness.features
    });
    
    // 3. CSR analysis (if applicable)
    let csrAnalysis = null;
    if (context.csr_focus) {
      csrAnalysis = await this.csrScorer.scoreAuthenticity({
        campaign_id: context.campaign_id,
        csr_content: { focus: context.csr_focus },
        brand_history: await this.getBrandHistory(context.brand)
      });
    }
    
    // 4. Pattern discovery
    const patterns = await this.patternEngine.discoverPatterns({
      campaign_id: context.campaign_id,
      creative_features: effectiveness.features
    });
    
    // 5. Generate optimizations
    const optimizations = await this.generateOptimizations({
      campaign_id: context.campaign_id,
      current_scores: effectiveness.scores,
      business_objectives: context.business_objectives
    });
    
    // 6. Store results
    await this.storeAnalysisResults({
      analysis_id: context.analysis_id,
      campaign_id: context.campaign_id,
      effectiveness,
      awardPredictions,
      csrAnalysis,
      patterns,
      optimizations
    });
    
    return {
      analysis_id: context.analysis_id,
      campaign_id: context.campaign_id,
      timestamp: new Date().toISOString(),
      effectiveness,
      award_predictions: awardPredictions,
      csr_analysis: csrAnalysis,
      patterns_discovered: patterns,
      optimization_recommendations: optimizations
    };
  }

  /**
   * Analyze creative effectiveness
   */
  async analyzeCreativeEffectiveness(assets, context) {
    // Extract features from assets
    const features = await this.extractCreativeFeatures(assets);
    
    // Calculate scores
    const scores = {
      visual_complexity: this.calculateVisualComplexity(features),
      message_clarity: Math.random() * 0.3 + 0.7,
      emotional_appeal: Math.random() * 0.4 + 0.6,
      brand_prominence: Math.random() * 0.3 + 0.7,
      innovation_score: Math.random() * 0.5 + 0.5,
      attention_score: Math.random() * 0.4 + 0.6,
      brand_recall_score: Math.random() * 0.3 + 0.7,
      cultural_alignment: await this.calculateCulturalAlignment(features, context.target_cultures)
    };
    
    // Calculate overall effectiveness
    const overallScore = Object.values(scores).reduce((a, b) => a + b) / Object.keys(scores).length;
    
    return {
      overall_score: overallScore,
      scores,
      features,
      confidence: 0.85 + Math.random() * 0.1
    };
  }

  /**
   * Extract creative features
   */
  async extractCreativeFeatures(assets) {
    const features = {
      asset_count: assets.length,
      asset_types: [...new Set(assets.map(a => a.type))],
      total_size: assets.reduce((sum, a) => sum + a.size, 0),
      visual_features: {},
      text_features: {}
    };
    
    // Extract type-specific features
    for (const asset of assets) {
      if (asset.type === 'image') {
        // Simulate image feature extraction
        features.visual_features[asset.id] = {
          dominant_colors: ['#FF0000', '#00FF00', '#0000FF'],
          complexity_score: Math.random(),
          has_faces: Math.random() > 0.5,
          has_text: Math.random() > 0.7
        };
      }
    }
    
    return features;
  }

  /**
   * Calculate visual complexity
   */
  calculateVisualComplexity(features) {
    if (!features.visual_features || Object.keys(features.visual_features).length === 0) {
      return 0.5;
    }
    
    const complexityScores = Object.values(features.visual_features)
      .map(vf => vf.complexity_score || 0.5);
    
    return complexityScores.reduce((a, b) => a + b) / complexityScores.length;
  }

  /**
   * Calculate cultural alignment
   */
  async calculateCulturalAlignment(features, targetCultures) {
    if (!targetCultures || targetCultures.length === 0) {
      return 0.8;
    }
    
    // Simulate cultural alignment calculation
    const alignmentScores = targetCultures.map(culture => {
      return 0.7 + Math.random() * 0.3;
    });
    
    return alignmentScores.reduce((a, b) => a + b) / alignmentScores.length;
  }

  /**
   * Generate optimization recommendations
   */
  async generateOptimizations({ campaign_id, current_scores, business_objectives }) {
    const optimizations = [];
    
    // Find areas for improvement
    const scoreEntries = Object.entries(current_scores || {})
      .sort((a, b) => a[1] - b[1]);
    
    // Generate recommendations for lowest scoring areas
    for (let i = 0; i < Math.min(3, scoreEntries.length); i++) {
      const [metric, score] = scoreEntries[i];
      
      optimizations.push({
        area: metric.replace(/_/g, ' ').replace('score', ''),
        current_score: score,
        target_score: Math.min(score + 0.2, 1.0),
        potential_impact: (0.2 * (1 - score)).toFixed(2),
        recommendation: this.getOptimizationRecommendation(metric, score),
        implementation_effort: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
        priority: i + 1,
        estimated_roi: Math.round((0.2 * (1 - score)) * 1000) / 10 + '%'
      });
    }
    
    return optimizations;
  }

  /**
   * Get optimization recommendation
   */
  getOptimizationRecommendation(metric, score) {
    const recommendations = {
      visual_complexity: score < 0.5 
        ? "Increase visual interest with varied elements while maintaining clarity"
        : "Simplify visual hierarchy to improve comprehension",
      message_clarity: "Refine core message to ensure immediate understanding",
      emotional_appeal: "Strengthen emotional storytelling and human connection",
      brand_prominence: "Optimize brand element placement for better recall",
      innovation_score: "Introduce unexpected creative elements or formats",
      attention_score: "Enhance visual hooks and engagement triggers",
      brand_recall_score: "Reinforce brand mnemonics and distinctive assets",
      cultural_alignment: "Adapt content to better resonate with target cultures"
    };
    
    return recommendations[metric] || "Optimize this metric for improved performance";
  }

  /**
   * Helper methods
   */
  
  detectAssetType(mimetype) {
    if (mimetype.startsWith('image/')) return 'image';
    if (mimetype.startsWith('video/')) return 'video';
    if (mimetype.startsWith('audio/')) return 'audio';
    if (mimetype.startsWith('text/')) return 'text';
    return 'other';
  }

  async getBrandHistory(brand) {
    // Simulate brand history lookup
    return {
      years_active: Math.floor(Math.random() * 20) + 5,
      csr_initiatives: Math.floor(Math.random() * 10),
      previous_awards: Math.floor(Math.random() * 15)
    };
  }

  async storeAnalysisResults(results) {
    // Store in database (implementation depends on your setup)
    console.log(`💾 Storing analysis results for ${results.analysis_id}`);
  }

  async getAnalysisResults(analysisId) {
    // Retrieve from database (implementation depends on your setup)
    console.log(`📤 Retrieving analysis results for ${analysisId}`);
    
    // Return mock data for now
    return {
      analysis_id: analysisId,
      status: 'completed',
      timestamp: new Date().toISOString()
    };
  }

  async listAnalyses(limit, offset) {
    // List analyses from database
    console.log(`📋 Listing analyses (limit: ${limit}, offset: ${offset})`);
    
    // Return mock data for now
    return {
      total: 100,
      items: []
    };
  }

  /**
   * Start the API server
   */
  start() {
    this.app.listen(this.port, () => {
      console.log(`🚀 JamPacked API Server running on port ${this.port}`);
      console.log(`📍 Health check: http://localhost:${this.port}/health`);
      console.log(`📍 API docs: http://localhost:${this.port}/api-docs`);
    });
  }
}




// Export modules
module.exports = {
  JamPackedAPI
};

// Start server if run directly
if (require.main === module) {
  const api = new JamPackedAPI();
  api.start();
}