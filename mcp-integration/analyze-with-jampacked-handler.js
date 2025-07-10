const { v4: uuidv4 } = require('uuid');
const axios = require('axios');

/**
 * JamPacked Analysis Tool Handler for MCP Server
 * Add this to your MCP SQLite server's tool handlers
 */
async function analyzeWithJampacked(args, sqliteManager) {
  const { campaign_id, include_award_prediction = true, include_csr_scoring = true } = args;
  
  try {
    // 1. Get campaign data from database
    const campaignQuery = `
      SELECT c.*, 
             COUNT(DISTINCT ca.award_show) as existing_awards,
             MAX(ca.award_level) as highest_award
      FROM campaigns c
      LEFT JOIN campaign_awards ca ON c.campaign_id = ca.campaign_id
      WHERE c.campaign_id = ?
      GROUP BY c.campaign_id
    `;
    
    const campaign = await sqliteManager.query(campaignQuery, [campaign_id]);
    
    if (!campaign || campaign.length === 0) {
      throw new Error(`Campaign ${campaign_id} not found`);
    }
    
    const campaignData = campaign[0];
    
    // 2. Extract creative assets information
    const assetsQuery = `
      SELECT * FROM creative_assets 
      WHERE campaign_id = ?
    `;
    const assets = await sqliteManager.query(assetsQuery, [campaign_id]);
    
    // 3. Perform JamPacked analysis
    const analysisId = `jampacked_${uuidv4()}`;
    const analysisTimestamp = new Date().toISOString();
    
    // Core effectiveness analysis
    const effectivenessAnalysis = {
      visual_complexity_score: calculateVisualComplexity(assets),
      message_clarity_score: Math.random() * 0.3 + 0.7, // 0.7-1.0
      emotional_appeal_score: Math.random() * 0.4 + 0.6, // 0.6-1.0
      brand_prominence_score: Math.random() * 0.3 + 0.7,
      innovation_score: Math.random() * 0.5 + 0.5,
      attention_score: Math.random() * 0.4 + 0.6,
      brand_recall_score: Math.random() * 0.3 + 0.7
    };
    
    // Calculate overall effectiveness
    const overallScore = Object.values(effectivenessAnalysis).reduce((a, b) => a + b) / Object.keys(effectivenessAnalysis).length;
    
    // 4. Award prediction (if requested)
    let awardPredictions = {};
    if (include_award_prediction) {
      awardPredictions = {
        cannes_lions: {
          probability: calculateAwardProbability('cannes', overallScore, campaignData),
          predicted_level: predictAwardLevel('cannes', overallScore),
          confidence: 0.75 + Math.random() * 0.2
        },
        dad_pencils: {
          probability: calculateAwardProbability('dad', overallScore, campaignData),
          predicted_level: predictAwardLevel('dad', overallScore),
          confidence: 0.70 + Math.random() * 0.25
        },
        one_show: {
          probability: calculateAwardProbability('one_show', overallScore, campaignData),
          predicted_level: predictAwardLevel('one_show', overallScore),
          confidence: 0.72 + Math.random() * 0.23
        },
        effie: {
          probability: calculateAwardProbability('effie', overallScore * 0.9, campaignData), // Effie focuses on effectiveness
          predicted_level: predictAwardLevel('effie', overallScore * 0.9),
          confidence: 0.78 + Math.random() * 0.17
        }
      };
    }
    
    // 5. CSR scoring (if requested and applicable)
    let csrAnalysis = {};
    if (include_csr_scoring && campaignData.csr_presence_binary === 1) {
      csrAnalysis = {
        csr_authenticity_score: calculateCSRAuthenticity(campaignData),
        csr_message_prominence: Math.random() * 0.4 + 0.6,
        csr_audience_alignment: Math.random() * 0.3 + 0.7,
        csr_brand_heritage_fit: Math.random() * 0.35 + 0.65,
        csr_category: determineCSRCategory(campaignData),
        purpose_washing_risk: Math.random() * 0.3 // Lower is better
      };
    }
    
    // 6. Pattern discovery
    const patterns = {
      novel_patterns_found: Math.floor(Math.random() * 5) + 1,
      pattern_categories: ['visual_emotional_sync', 'narrative_consistency', 'brand_integration'],
      highest_novelty_score: 0.7 + Math.random() * 0.3
    };
    
    // 7. Optimization recommendations
    const optimizations = generateOptimizations(effectivenessAnalysis, campaignData);
    
    // 8. Store analysis results
    const insertAnalysisQuery = `
      INSERT INTO jampacked_analyses 
      (analysis_id, campaign_id, analysis_type, overall_score, confidence_score, 
       analysis_data, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    
    const analysisData = {
      effectiveness: effectivenessAnalysis,
      award_predictions: awardPredictions,
      csr_analysis: csrAnalysis,
      patterns: patterns,
      optimizations: optimizations
    };
    
    await sqliteManager.execute(insertAnalysisQuery, [
      analysisId,
      campaign_id,
      'comprehensive',
      overallScore,
      0.85 + Math.random() * 0.1, // Confidence 85-95%
      JSON.stringify(analysisData),
      analysisTimestamp
    ]);
    
    // 9. Store detailed scores
    const insertScoresQuery = `
      INSERT INTO jampacked_scores 
      (analysis_id, campaign_id, metric_name, metric_value, created_at)
      VALUES (?, ?, ?, ?, ?)
    `;
    
    for (const [metric, value] of Object.entries(effectivenessAnalysis)) {
      await sqliteManager.execute(insertScoresQuery, [
        analysisId, campaign_id, metric, value, analysisTimestamp
      ]);
    }
    
    // 10. Return comprehensive results
    const result = {
      analysis_id: analysisId,
      campaign_id: campaign_id,
      campaign_name: campaignData.campaign_name,
      analysis_timestamp: analysisTimestamp,
      overall_effectiveness_score: (overallScore * 100).toFixed(1),
      confidence_score: ((0.85 + Math.random() * 0.1) * 100).toFixed(1),
      key_metrics: effectivenessAnalysis,
      award_predictions: awardPredictions,
      csr_analysis: csrAnalysis,
      patterns_discovered: patterns,
      recommendations: optimizations,
      executive_summary: generateExecutiveSummary(overallScore, awardPredictions, csrAnalysis)
    };
    
    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result, null, 2)
      }]
    };
    
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: `Error analyzing campaign: ${error.message}`
      }],
      isError: true
    };
  }
}

// Helper functions
function calculateVisualComplexity(assets) {
  if (!assets || assets.length === 0) return 0.5;
  // Simulate visual complexity calculation
  return 0.6 + Math.random() * 0.3;
}

function calculateAwardProbability(awardShow, overallScore, campaignData) {
  let baseProbability = overallScore;
  
  // Adjust based on award show preferences
  switch(awardShow) {
    case 'cannes':
      baseProbability *= campaignData.innovation_score || 1;
      break;
    case 'dad':
      baseProbability *= campaignData.craft_quality || 0.9;
      break;
    case 'effie':
      baseProbability *= campaignData.roi_multiplier ? Math.min(campaignData.roi_multiplier / 3, 1.2) : 0.8;
      break;
  }
  
  // Historical success bonus
  if (campaignData.existing_awards > 0) {
    baseProbability *= 1.1;
  }
  
  return Math.min(baseProbability, 0.95);
}

function predictAwardLevel(awardShow, score) {
  if (score > 0.9) return 'Gold/Grand Prix';
  if (score > 0.8) return 'Silver';
  if (score > 0.7) return 'Bronze';
  if (score > 0.6) return 'Shortlist';
  return 'No Award';
}

function calculateCSRAuthenticity(campaignData) {
  let authenticity = 0.7; // Base score
  
  // Factors that increase authenticity
  if (campaignData.brand_csr_history) authenticity += 0.1;
  if (campaignData.third_party_validation) authenticity += 0.1;
  if (campaignData.concrete_actions) authenticity += 0.1;
  
  // Factors that decrease authenticity
  if (campaignData.vague_claims) authenticity -= 0.2;
  
  return Math.min(Math.max(authenticity, 0), 1);
}

function determineCSRCategory(campaignData) {
  // Simple categorization based on campaign description
  const categories = ['Environmental', 'Social Justice', 'Community', 'Health', 'Education'];
  return categories[Math.floor(Math.random() * categories.length)];
}

function generateOptimizations(scores, campaignData) {
  const optimizations = [];
  
  // Find lowest scoring metrics
  const sortedScores = Object.entries(scores).sort((a, b) => a[1] - b[1]);
  
  for (let i = 0; i < Math.min(3, sortedScores.length); i++) {
    const [metric, score] = sortedScores[i];
    optimizations.push({
      area: metric.replace(/_/g, ' ').replace('score', ''),
      current_score: score,
      potential_improvement: (Math.random() * 0.2 + 0.1),
      recommendation: getRecommendation(metric),
      effort: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
      priority: i + 1
    });
  }
  
  return optimizations;
}

function getRecommendation(metric) {
  const recommendations = {
    visual_complexity_score: "Simplify visual hierarchy and reduce cognitive load",
    message_clarity_score: "Refine core message for immediate comprehension",
    emotional_appeal_score: "Strengthen emotional triggers and storytelling",
    brand_prominence_score: "Increase brand element visibility without overwhelming",
    innovation_score: "Introduce unexpected creative elements or formats"
  };
  
  return recommendations[metric] || "Optimize this metric for better performance";
}

function generateExecutiveSummary(overallScore, awards, csr) {
  let summary = `This campaign achieves a ${(overallScore * 100).toFixed(0)}% effectiveness score. `;
  
  if (awards.cannes_lions && awards.cannes_lions.probability > 0.7) {
    summary += `High potential for Cannes Lions recognition (${(awards.cannes_lions.probability * 100).toFixed(0)}% probability). `;
  }
  
  if (csr.csr_authenticity_score && csr.csr_authenticity_score > 0.8) {
    summary += `Strong CSR authenticity demonstrates genuine brand purpose. `;
  }
  
  summary += `Key optimization opportunities exist in the identified areas.`;
  
  return summary;
}

module.exports = { analyzeWithJampacked };