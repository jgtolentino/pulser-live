const sqlite3 = require('sqlite3').verbose();
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

/**
 * Analysis Versioning and Reproducibility System
 * Tracks analysis versions, parameters, and enables reproducible results
 */
class AnalysisVersioningSystem {
  constructor(dbPath) {
    this.dbPath = dbPath || '/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite';
    this.db = null;
  }

  /**
   * Initialize versioning system
   */
  async initialize() {
    this.db = new sqlite3.Database(this.dbPath);
    await this.createVersioningTables();
    console.log('✅ Analysis Versioning System initialized');
  }

  /**
   * Create versioning tables
   */
  async createVersioningTables() {
    const queries = [
      // Analysis versions table
      `CREATE TABLE IF NOT EXISTS analysis_versions (
        version_id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        parent_version_id TEXT,
        analysis_type TEXT NOT NULL,
        model_versions TEXT NOT NULL,
        parameters TEXT NOT NULL,
        input_hash TEXT NOT NULL,
        output_hash TEXT NOT NULL,
        results_summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_by TEXT,
        notes TEXT,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
      )`,
      
      // Model versions tracking
      `CREATE TABLE IF NOT EXISTS model_versions (
        model_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        model_type TEXT NOT NULL,
        checksum TEXT NOT NULL,
        training_date DATETIME,
        parameters TEXT,
        performance_metrics TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        deprecated BOOLEAN DEFAULT 0
      )`,
      
      // Analysis lineage tracking
      `CREATE TABLE IF NOT EXISTS analysis_lineage (
        lineage_id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        parent_analysis_id TEXT,
        lineage_type TEXT NOT NULL,
        transformation_applied TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Reproducibility snapshots
      `CREATE TABLE IF NOT EXISTS reproducibility_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        version_id TEXT NOT NULL,
        environment_data TEXT NOT NULL,
        dependencies TEXT NOT NULL,
        random_seeds TEXT,
        data_snapshot_hash TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Analysis diffs
      `CREATE TABLE IF NOT EXISTS analysis_diffs (
        diff_id TEXT PRIMARY KEY,
        version_id_a TEXT NOT NULL,
        version_id_b TEXT NOT NULL,
        diff_type TEXT NOT NULL,
        diff_data TEXT NOT NULL,
        significance_score REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Audit log
      `CREATE TABLE IF NOT EXISTS analysis_audit_log (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL,
        version_id TEXT,
        action TEXT NOT NULL,
        actor TEXT NOT NULL,
        details TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`
    ];

    for (const query of queries) {
      await this.runQuery(query);
    }

    // Create indexes for performance
    await this.createIndexes();
  }

  /**
   * Create indexes for performance
   */
  async createIndexes() {
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_versions_analysis ON analysis_versions(analysis_id)',
      'CREATE INDEX IF NOT EXISTS idx_versions_campaign ON analysis_versions(campaign_id)',
      'CREATE INDEX IF NOT EXISTS idx_versions_created ON analysis_versions(created_at)',
      'CREATE INDEX IF NOT EXISTS idx_lineage_analysis ON analysis_lineage(analysis_id)',
      'CREATE INDEX IF NOT EXISTS idx_audit_analysis ON analysis_audit_log(analysis_id)'
    ];

    for (const index of indexes) {
      await this.runQuery(index);
    }
  }

  /**
   * Create a new analysis version
   */
  async createVersion(analysisData) {
    const {
      analysis_id,
      campaign_id,
      analysis_type,
      parameters,
      results,
      model_info,
      created_by,
      notes
    } = analysisData;

    // Get current version number
    const currentVersion = await this.getCurrentVersion(analysis_id);
    const versionNumber = currentVersion ? currentVersion.version_number + 1 : 1;
    
    // Generate version ID
    const versionId = `v_${analysis_id}_${versionNumber}`;
    
    // Calculate hashes for reproducibility
    const inputHash = this.calculateHash({
      campaign_id,
      parameters,
      model_info
    });
    
    const outputHash = this.calculateHash(results);
    
    // Store version
    const versionData = {
      version_id: versionId,
      analysis_id,
      campaign_id,
      version_number: versionNumber,
      parent_version_id: currentVersion?.version_id || null,
      analysis_type,
      model_versions: JSON.stringify(model_info),
      parameters: JSON.stringify(parameters),
      input_hash: inputHash,
      output_hash: outputHash,
      results_summary: JSON.stringify(this.summarizeResults(results)),
      created_by: created_by || 'system',
      notes: notes || null
    };
    
    await this.insertVersion(versionData);
    
    // Create reproducibility snapshot
    await this.createReproducibilitySnapshot(versionId, analysisData);
    
    // Track lineage if parent exists
    if (currentVersion) {
      await this.trackLineage(analysis_id, versionId, currentVersion.version_id);
    }
    
    // Log action
    await this.logAction(analysis_id, versionId, 'version_created', created_by);
    
    return versionId;
  }

  /**
   * Get analysis version history
   */
  async getVersionHistory(analysisId) {
    const query = `
      SELECT 
        v.*,
        COUNT(DISTINCT l.lineage_id) as derivative_count
      FROM analysis_versions v
      LEFT JOIN analysis_lineage l ON v.version_id = l.parent_analysis_id
      WHERE v.analysis_id = ?
      GROUP BY v.version_id
      ORDER BY v.version_number DESC
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [analysisId], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Compare two versions
   */
  async compareVersions(versionIdA, versionIdB) {
    // Get both versions
    const [versionA, versionB] = await Promise.all([
      this.getVersion(versionIdA),
      this.getVersion(versionIdB)
    ]);
    
    if (!versionA || !versionB) {
      throw new Error('One or both versions not found');
    }
    
    // Calculate differences
    const diff = {
      parameter_changes: this.diffObjects(
        JSON.parse(versionA.parameters),
        JSON.parse(versionB.parameters)
      ),
      model_changes: this.diffObjects(
        JSON.parse(versionA.model_versions),
        JSON.parse(versionB.model_versions)
      ),
      result_changes: this.diffObjects(
        JSON.parse(versionA.results_summary),
        JSON.parse(versionB.results_summary)
      ),
      metadata: {
        version_a: versionIdA,
        version_b: versionIdB,
        time_difference: new Date(versionB.created_at) - new Date(versionA.created_at),
        created_by_same: versionA.created_by === versionB.created_by
      }
    };
    
    // Calculate significance score
    const significanceScore = this.calculateSignificance(diff);
    
    // Store diff
    await this.storeDiff(versionIdA, versionIdB, diff, significanceScore);
    
    return {
      diff,
      significance_score: significanceScore,
      recommendation: this.generateDiffRecommendation(diff, significanceScore)
    };
  }

  /**
   * Reproduce an analysis from a specific version
   */
  async reproduceAnalysis(versionId) {
    const version = await this.getVersion(versionId);
    if (!version) {
      throw new Error(`Version ${versionId} not found`);
    }
    
    const snapshot = await this.getReproducibilitySnapshot(versionId);
    if (!snapshot) {
      throw new Error(`Reproducibility snapshot for ${versionId} not found`);
    }
    
    // Prepare reproduction environment
    const reproductionConfig = {
      version_id: versionId,
      analysis_id: version.analysis_id,
      campaign_id: version.campaign_id,
      analysis_type: version.analysis_type,
      parameters: JSON.parse(version.parameters),
      model_versions: JSON.parse(version.model_versions),
      environment: JSON.parse(snapshot.environment_data),
      dependencies: JSON.parse(snapshot.dependencies),
      random_seeds: snapshot.random_seeds ? JSON.parse(snapshot.random_seeds) : null,
      original_input_hash: version.input_hash,
      original_output_hash: version.output_hash
    };
    
    // Log reproduction attempt
    await this.logAction(
      version.analysis_id,
      versionId,
      'reproduction_attempted',
      'system'
    );
    
    return reproductionConfig;
  }

  /**
   * Track model version
   */
  async trackModelVersion(modelInfo) {
    const {
      model_name,
      model_version,
      model_type,
      training_date,
      parameters,
      performance_metrics
    } = modelInfo;
    
    const modelId = `model_${model_name}_${model_version}`;
    const checksum = this.calculateHash(modelInfo);
    
    const query = `
      INSERT OR REPLACE INTO model_versions 
      (model_id, model_name, model_version, model_type, checksum, 
       training_date, parameters, performance_metrics)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      modelId,
      model_name,
      model_version,
      model_type,
      checksum,
      training_date,
      JSON.stringify(parameters),
      JSON.stringify(performance_metrics)
    ]);
    
    return modelId;
  }

  /**
   * Get analysis lineage tree
   */
  async getLineageTree(analysisId) {
    const query = `
      WITH RECURSIVE lineage_tree AS (
        SELECT 
          l.*,
          v.version_number,
          v.created_at,
          v.created_by,
          0 as depth
        FROM analysis_lineage l
        JOIN analysis_versions v ON l.version_id = v.version_id
        WHERE l.analysis_id = ?
        AND l.parent_analysis_id IS NULL
        
        UNION ALL
        
        SELECT 
          l.*,
          v.version_number,
          v.created_at,
          v.created_by,
          lt.depth + 1
        FROM analysis_lineage l
        JOIN analysis_versions v ON l.version_id = v.version_id
        JOIN lineage_tree lt ON l.parent_analysis_id = lt.analysis_id
      )
      SELECT * FROM lineage_tree
      ORDER BY depth, created_at
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [analysisId], (err, rows) => {
        if (err) reject(err);
        else resolve(this.buildLineageTree(rows));
      });
    });
  }

  /**
   * Search versions by criteria
   */
  async searchVersions(criteria) {
    const {
      campaign_id,
      analysis_type,
      created_by,
      date_from,
      date_to,
      min_score
    } = criteria;
    
    let query = `
      SELECT v.*, 
             COUNT(DISTINCT d.diff_id) as comparison_count
      FROM analysis_versions v
      LEFT JOIN analysis_diffs d ON v.version_id IN (d.version_id_a, d.version_id_b)
      WHERE 1=1
    `;
    
    const params = [];
    
    if (campaign_id) {
      query += ' AND v.campaign_id = ?';
      params.push(campaign_id);
    }
    
    if (analysis_type) {
      query += ' AND v.analysis_type = ?';
      params.push(analysis_type);
    }
    
    if (created_by) {
      query += ' AND v.created_by = ?';
      params.push(created_by);
    }
    
    if (date_from) {
      query += ' AND v.created_at >= ?';
      params.push(date_from);
    }
    
    if (date_to) {
      query += ' AND v.created_at <= ?';
      params.push(date_to);
    }
    
    query += ' GROUP BY v.version_id ORDER BY v.created_at DESC';
    
    return new Promise((resolve, reject) => {
      this.db.all(query, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Get audit trail for an analysis
   */
  async getAuditTrail(analysisId) {
    const query = `
      SELECT * FROM analysis_audit_log
      WHERE analysis_id = ?
      ORDER BY timestamp DESC
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [analysisId], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Helper methods
   */
  
  calculateHash(data) {
    const json = JSON.stringify(data, Object.keys(data).sort());
    return crypto.createHash('sha256').update(json).digest('hex');
  }
  
  summarizeResults(results) {
    return {
      overall_score: results.overall_score || null,
      key_metrics: Object.keys(results.key_metrics || {}),
      predictions_made: !!results.award_predictions,
      csr_analyzed: !!results.csr_analysis,
      patterns_found: results.patterns_discovered?.novel_patterns_found || 0,
      recommendations_count: results.recommendations?.length || 0
    };
  }
  
  diffObjects(objA, objB) {
    const diff = {
      added: {},
      removed: {},
      changed: {}
    };
    
    // Find added and changed
    for (const key in objB) {
      if (!(key in objA)) {
        diff.added[key] = objB[key];
      } else if (JSON.stringify(objA[key]) !== JSON.stringify(objB[key])) {
        diff.changed[key] = {
          from: objA[key],
          to: objB[key]
        };
      }
    }
    
    // Find removed
    for (const key in objA) {
      if (!(key in objB)) {
        diff.removed[key] = objA[key];
      }
    }
    
    return diff;
  }
  
  calculateSignificance(diff) {
    let score = 0;
    
    // Parameter changes
    const paramChanges = Object.keys(diff.parameter_changes.changed).length;
    score += paramChanges * 0.2;
    
    // Model changes (highly significant)
    const modelChanges = Object.keys(diff.model_changes.changed).length;
    score += modelChanges * 0.5;
    
    // Result changes
    const resultChanges = diff.result_changes;
    if (resultChanges.changed.overall_score) {
      const scoreDiff = Math.abs(
        resultChanges.changed.overall_score.from - 
        resultChanges.changed.overall_score.to
      );
      score += scoreDiff;
    }
    
    return Math.min(score, 1.0);
  }
  
  generateDiffRecommendation(diff, significanceScore) {
    if (significanceScore > 0.7) {
      return 'Significant changes detected. Review carefully before using results.';
    } else if (significanceScore > 0.3) {
      return 'Moderate changes detected. Results may vary from original.';
    } else {
      return 'Minor changes detected. Results should be consistent.';
    }
  }
  
  buildLineageTree(rows) {
    const tree = {};
    const nodeMap = {};
    
    // Build node map
    rows.forEach(row => {
      nodeMap[row.analysis_id] = {
        ...row,
        children: []
      };
    });
    
    // Build tree structure
    rows.forEach(row => {
      if (row.parent_analysis_id && nodeMap[row.parent_analysis_id]) {
        nodeMap[row.parent_analysis_id].children.push(nodeMap[row.analysis_id]);
      } else {
        tree[row.analysis_id] = nodeMap[row.analysis_id];
      }
    });
    
    return tree;
  }
  
  async getCurrentVersion(analysisId) {
    const query = `
      SELECT * FROM analysis_versions
      WHERE analysis_id = ?
      ORDER BY version_number DESC
      LIMIT 1
    `;
    
    return new Promise((resolve, reject) => {
      this.db.get(query, [analysisId], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }
  
  async getVersion(versionId) {
    const query = 'SELECT * FROM analysis_versions WHERE version_id = ?';
    
    return new Promise((resolve, reject) => {
      this.db.get(query, [versionId], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }
  
  async insertVersion(versionData) {
    const query = `
      INSERT INTO analysis_versions 
      (version_id, analysis_id, campaign_id, version_number, parent_version_id,
       analysis_type, model_versions, parameters, input_hash, output_hash,
       results_summary, created_by, notes)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      versionData.version_id,
      versionData.analysis_id,
      versionData.campaign_id,
      versionData.version_number,
      versionData.parent_version_id,
      versionData.analysis_type,
      versionData.model_versions,
      versionData.parameters,
      versionData.input_hash,
      versionData.output_hash,
      versionData.results_summary,
      versionData.created_by,
      versionData.notes
    ];
    
    await this.runQuery(query, params);
  }
  
  async createReproducibilitySnapshot(versionId, analysisData) {
    const snapshot = {
      snapshot_id: `snap_${versionId}`,
      analysis_id: analysisData.analysis_id,
      version_id: versionId,
      environment_data: JSON.stringify({
        node_version: process.version,
        platform: process.platform,
        arch: process.arch,
        timestamp: new Date().toISOString()
      }),
      dependencies: JSON.stringify(analysisData.dependencies || {}),
      random_seeds: JSON.stringify(analysisData.random_seeds || null),
      data_snapshot_hash: this.calculateHash(analysisData.input_data || {})
    };
    
    const query = `
      INSERT INTO reproducibility_snapshots 
      (snapshot_id, analysis_id, version_id, environment_data, 
       dependencies, random_seeds, data_snapshot_hash)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      snapshot.snapshot_id,
      snapshot.analysis_id,
      snapshot.version_id,
      snapshot.environment_data,
      snapshot.dependencies,
      snapshot.random_seeds,
      snapshot.data_snapshot_hash
    ]);
  }
  
  async getReproducibilitySnapshot(versionId) {
    const query = 'SELECT * FROM reproducibility_snapshots WHERE version_id = ?';
    
    return new Promise((resolve, reject) => {
      this.db.get(query, [versionId], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }
  
  async trackLineage(analysisId, versionId, parentVersionId) {
    const query = `
      INSERT INTO analysis_lineage 
      (lineage_id, analysis_id, version_id, parent_analysis_id, lineage_type)
      VALUES (?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      `lineage_${uuidv4()}`,
      analysisId,
      versionId,
      parentVersionId,
      'version_update'
    ]);
  }
  
  async storeDiff(versionIdA, versionIdB, diff, significanceScore) {
    const query = `
      INSERT INTO analysis_diffs 
      (diff_id, version_id_a, version_id_b, diff_type, diff_data, significance_score)
      VALUES (?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      `diff_${uuidv4()}`,
      versionIdA,
      versionIdB,
      'version_comparison',
      JSON.stringify(diff),
      significanceScore
    ]);
  }
  
  async logAction(analysisId, versionId, action, actor) {
    const query = `
      INSERT INTO analysis_audit_log 
      (analysis_id, version_id, action, actor, details)
      VALUES (?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      analysisId,
      versionId,
      action,
      actor || 'system',
      JSON.stringify({ timestamp: new Date().toISOString() })
    ]);
  }
  
  runQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }
}

// Export module
module.exports = {
  AnalysisVersioningSystem
};

// Example usage
if (require.main === module) {
  const versioning = new AnalysisVersioningSystem();
  
  async function demonstrateVersioning() {
    await versioning.initialize();
    
    // Create initial version
    const v1 = await versioning.createVersion({
      analysis_id: 'test_analysis_001',
      campaign_id: 'campaign_001',
      analysis_type: 'comprehensive',
      parameters: {
        include_awards: true,
        include_csr: true,
        confidence_threshold: 0.8
      },
      results: {
        overall_score: 0.85,
        key_metrics: {
          effectiveness: 0.9,
          innovation: 0.8
        }
      },
      model_info: {
        award_predictor: 'v1.2.0',
        csr_scorer: 'v1.1.0'
      },
      created_by: 'test_user'
    });
    
    console.log('✅ Created version:', v1);
    
    // Create updated version
    const v2 = await versioning.createVersion({
      analysis_id: 'test_analysis_001',
      campaign_id: 'campaign_001',
      analysis_type: 'comprehensive',
      parameters: {
        include_awards: true,
        include_csr: true,
        confidence_threshold: 0.85  // Changed parameter
      },
      results: {
        overall_score: 0.87,  // Different result
        key_metrics: {
          effectiveness: 0.92,
          innovation: 0.82
        }
      },
      model_info: {
        award_predictor: 'v1.2.1',  // Updated model
        csr_scorer: 'v1.1.0'
      },
      created_by: 'test_user'
    });
    
    console.log('✅ Created version:', v2);
    
    // Compare versions
    const comparison = await versioning.compareVersions(v1, v2);
    console.log('\n📊 Version comparison:');
    console.log(JSON.stringify(comparison, null, 2));
    
    // Get version history
    const history = await versioning.getVersionHistory('test_analysis_001');
    console.log('\n📜 Version history:');
    console.log(history);
    
    // Reproduce analysis
    const reproConfig = await versioning.reproduceAnalysis(v1);
    console.log('\n🔄 Reproduction config:');
    console.log(JSON.stringify(reproConfig, null, 2));
  }
  
  demonstrateVersioning().catch(console.error);
}