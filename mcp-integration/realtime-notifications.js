const WebSocket = require('ws');
const EventEmitter = require('events');
const sqlite3 = require('sqlite3').verbose();
const { v4: uuidv4 } = require('uuid');

/**
 * Real-time Notification System for Claude Desktop & Code Integration
 * Enables live communication between SQL-based Claude Desktop and Python-based Claude Code
 */
class RealtimeNotificationSystem extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      wsPort: config.wsPort || 8765,
      dbPath: config.dbPath || '/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite',
      heartbeatInterval: config.heartbeatInterval || 30000,
      reconnectDelay: config.reconnectDelay || 5000
    };
    
    this.wss = null;
    this.db = null;
    this.clients = new Map();
    this.subscriptions = new Map();
    this.messageQueue = [];
  }

  /**
   * Initialize the notification system
   */
  async initialize() {
    // Set up database connection
    this.db = new sqlite3.Database(this.config.dbPath);
    await this.createNotificationTables();
    
    // Set up WebSocket server
    this.wss = new WebSocket.Server({ port: this.config.wsPort });
    this.setupWebSocketHandlers();
    
    // Set up database triggers
    await this.setupDatabaseTriggers();
    
    // Start heartbeat
    this.startHeartbeat();
    
    console.log(`🚀 Realtime Notification System running on ws://localhost:${this.config.wsPort}`);
  }

  /**
   * Create notification tables
   */
  async createNotificationTables() {
    const queries = [
      `CREATE TABLE IF NOT EXISTS notifications (
        notification_id TEXT PRIMARY KEY,
        source_system TEXT NOT NULL,
        target_system TEXT,
        notification_type TEXT NOT NULL,
        payload TEXT,
        status TEXT DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        delivered_at DATETIME,
        acknowledged_at DATETIME
      )`,
      
      `CREATE TABLE IF NOT EXISTS notification_subscriptions (
        subscription_id TEXT PRIMARY KEY,
        client_id TEXT NOT NULL,
        system_type TEXT NOT NULL,
        notification_types TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      `CREATE TABLE IF NOT EXISTS system_status (
        system_id TEXT PRIMARY KEY,
        system_type TEXT NOT NULL,
        status TEXT NOT NULL,
        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
      )`
    ];

    for (const query of queries) {
      await this.runQuery(query);
    }
  }

  /**
   * Set up WebSocket handlers
   */
  setupWebSocketHandlers() {
    this.wss.on('connection', (ws, req) => {
      const clientId = uuidv4();
      console.log(`🔌 New client connected: ${clientId}`);
      
      // Store client connection
      this.clients.set(clientId, {
        ws,
        systemType: null,
        subscriptions: new Set(),
        lastActivity: Date.now()
      });
      
      // Send welcome message
      this.sendToClient(clientId, {
        type: 'welcome',
        clientId,
        timestamp: new Date().toISOString()
      });
      
      // Handle messages
      ws.on('message', (message) => {
        this.handleClientMessage(clientId, message);
      });
      
      // Handle close
      ws.on('close', () => {
        console.log(`👋 Client disconnected: ${clientId}`);
        this.handleClientDisconnect(clientId);
      });
      
      // Handle errors
      ws.on('error', (error) => {
        console.error(`❌ WebSocket error for ${clientId}:`, error);
      });
    });
  }

  /**
   * Handle incoming client messages
   */
  async handleClientMessage(clientId, message) {
    try {
      const data = JSON.parse(message);
      const client = this.clients.get(clientId);
      
      if (!client) return;
      
      client.lastActivity = Date.now();
      
      switch (data.type) {
        case 'identify':
          await this.handleIdentify(clientId, data);
          break;
          
        case 'subscribe':
          await this.handleSubscribe(clientId, data);
          break;
          
        case 'unsubscribe':
          await this.handleUnsubscribe(clientId, data);
          break;
          
        case 'notify':
          await this.handleNotify(clientId, data);
          break;
          
        case 'acknowledge':
          await this.handleAcknowledge(clientId, data);
          break;
          
        case 'query':
          await this.handleQuery(clientId, data);
          break;
          
        case 'heartbeat':
          this.sendToClient(clientId, { type: 'heartbeat_ack' });
          break;
          
        default:
          console.warn(`Unknown message type: ${data.type}`);
      }
    } catch (error) {
      console.error('Error handling message:', error);
      this.sendToClient(clientId, {
        type: 'error',
        error: error.message
      });
    }
  }

  /**
   * Handle client identification
   */
  async handleIdentify(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;
    
    client.systemType = data.systemType; // 'claude_desktop' or 'claude_code'
    client.systemInfo = data.systemInfo || {};
    
    // Update system status
    await this.updateSystemStatus(clientId, data.systemType, 'online');
    
    // Send confirmation
    this.sendToClient(clientId, {
      type: 'identified',
      systemType: data.systemType,
      timestamp: new Date().toISOString()
    });
    
    console.log(`✅ Client ${clientId} identified as ${data.systemType}`);
  }

  /**
   * Handle subscription requests
   */
  async handleSubscribe(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;
    
    const notificationTypes = Array.isArray(data.notificationTypes) 
      ? data.notificationTypes 
      : ['all'];
    
    // Add subscriptions
    notificationTypes.forEach(type => {
      client.subscriptions.add(type);
    });
    
    // Store in database
    await this.storeSubscription(clientId, client.systemType, notificationTypes);
    
    // Send confirmation
    this.sendToClient(clientId, {
      type: 'subscribed',
      notificationTypes,
      timestamp: new Date().toISOString()
    });
    
    // Send any pending notifications
    await this.sendPendingNotifications(clientId);
  }

  /**
   * Handle notification dispatch
   */
  async handleNotify(clientId, data) {
    const notification = {
      notification_id: data.notificationId || uuidv4(),
      source_system: data.sourceSystem,
      target_system: data.targetSystem,
      notification_type: data.notificationType,
      payload: JSON.stringify(data.payload),
      timestamp: new Date().toISOString()
    };
    
    // Store notification
    await this.storeNotification(notification);
    
    // Route to appropriate clients
    await this.routeNotification(notification);
    
    // Send confirmation to sender
    this.sendToClient(clientId, {
      type: 'notification_sent',
      notificationId: notification.notification_id,
      timestamp: notification.timestamp
    });
  }

  /**
   * Route notification to subscribed clients
   */
  async routeNotification(notification) {
    const targetClients = [];
    
    // Find matching clients
    for (const [clientId, client] of this.clients.entries()) {
      // Check if client matches target system
      if (notification.target_system && 
          client.systemType !== notification.target_system) {
        continue;
      }
      
      // Check if client is subscribed to this notification type
      if (client.subscriptions.has(notification.notification_type) ||
          client.subscriptions.has('all')) {
        targetClients.push(clientId);
      }
    }
    
    // Send to matching clients
    for (const clientId of targetClients) {
      const sent = this.sendToClient(clientId, {
        type: 'notification',
        notification: {
          id: notification.notification_id,
          type: notification.notification_type,
          source: notification.source_system,
          payload: JSON.parse(notification.payload),
          timestamp: notification.timestamp
        }
      });
      
      if (sent) {
        await this.markNotificationDelivered(notification.notification_id, clientId);
      }
    }
    
    // If no clients available, keep in queue
    if (targetClients.length === 0) {
      this.messageQueue.push(notification);
    }
  }

  /**
   * Handle notification acknowledgment
   */
  async handleAcknowledge(clientId, data) {
    const { notificationId } = data;
    
    await this.markNotificationAcknowledged(notificationId, clientId);
    
    this.sendToClient(clientId, {
      type: 'acknowledged',
      notificationId,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Handle query requests (for Claude Desktop SQL queries)
   */
  async handleQuery(clientId, data) {
    const { query, params = [] } = data;
    
    try {
      const results = await this.executeQuery(query, params);
      
      this.sendToClient(clientId, {
        type: 'query_result',
        queryId: data.queryId,
        results,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      this.sendToClient(clientId, {
        type: 'query_error',
        queryId: data.queryId,
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Set up database triggers for real-time updates
   */
  async setupDatabaseTriggers() {
    // Trigger for new analysis completions
    const analysisCompleteTrigger = `
      CREATE TRIGGER IF NOT EXISTS notify_analysis_complete
      AFTER INSERT ON jampacked_analyses
      BEGIN
        INSERT INTO notifications (
          notification_id,
          source_system,
          notification_type,
          payload
        ) VALUES (
          'notif_' || NEW.analysis_id,
          'jampacked_analysis',
          'analysis_complete',
          json_object(
            'analysis_id', NEW.analysis_id,
            'campaign_id', NEW.campaign_id,
            'overall_score', NEW.overall_score
          )
        );
      END;
    `;
    
    // Trigger for award predictions
    const awardPredictionTrigger = `
      CREATE TRIGGER IF NOT EXISTS notify_award_prediction
      AFTER INSERT ON award_predictions
      BEGIN
        INSERT INTO notifications (
          notification_id,
          source_system,
          notification_type,
          payload
        ) VALUES (
          'notif_' || NEW.prediction_id,
          'award_predictor',
          'prediction_complete',
          json_object(
            'prediction_id', NEW.prediction_id,
            'campaign_id', NEW.campaign_id
          )
        );
      END;
    `;
    
    await this.runQuery(analysisCompleteTrigger);
    await this.runQuery(awardPredictionTrigger);
    
    // Set up polling for new notifications
    setInterval(() => {
      this.checkForNewNotifications();
    }, 1000);
  }

  /**
   * Check for new notifications in database
   */
  async checkForNewNotifications() {
    const query = `
      SELECT * FROM notifications 
      WHERE status = 'pending' 
      ORDER BY created_at ASC 
      LIMIT 10
    `;
    
    this.db.all(query, async (err, notifications) => {
      if (err) {
        console.error('Error checking notifications:', err);
        return;
      }
      
      for (const notification of notifications) {
        await this.routeNotification(notification);
        await this.updateNotificationStatus(notification.notification_id, 'sent');
      }
    });
  }

  /**
   * Send pending notifications to a client
   */
  async sendPendingNotifications(clientId) {
    const client = this.clients.get(clientId);
    if (!client) return;
    
    // Check message queue
    const pending = this.messageQueue.filter(notif => {
      return (!notif.target_system || notif.target_system === client.systemType) &&
             (client.subscriptions.has(notif.notification_type) || 
              client.subscriptions.has('all'));
    });
    
    for (const notification of pending) {
      await this.routeNotification(notification);
    }
    
    // Clean up sent messages from queue
    this.messageQueue = this.messageQueue.filter(n => !pending.includes(n));
  }

  /**
   * Handle client disconnect
   */
  async handleClientDisconnect(clientId) {
    const client = this.clients.get(clientId);
    if (client && client.systemType) {
      await this.updateSystemStatus(clientId, client.systemType, 'offline');
    }
    
    this.clients.delete(clientId);
    this.emit('client_disconnected', clientId);
  }

  /**
   * Send message to specific client
   */
  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client || client.ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    
    try {
      client.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error(`Error sending to client ${clientId}:`, error);
      return false;
    }
  }

  /**
   * Broadcast message to all clients of a specific type
   */
  broadcast(systemType, message) {
    let sent = 0;
    
    for (const [clientId, client] of this.clients.entries()) {
      if (!systemType || client.systemType === systemType) {
        if (this.sendToClient(clientId, message)) {
          sent++;
        }
      }
    }
    
    return sent;
  }

  /**
   * Start heartbeat monitoring
   */
  startHeartbeat() {
    setInterval(() => {
      const now = Date.now();
      const timeout = this.config.heartbeatInterval * 2;
      
      for (const [clientId, client] of this.clients.entries()) {
        if (now - client.lastActivity > timeout) {
          console.log(`💔 Client ${clientId} timed out`);
          client.ws.terminate();
          this.handleClientDisconnect(clientId);
        } else {
          this.sendToClient(clientId, { type: 'heartbeat' });
        }
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Database helper methods
   */
  
  runQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }
  
  executeQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.all(query, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
  
  async storeNotification(notification) {
    const query = `
      INSERT INTO notifications 
      (notification_id, source_system, target_system, notification_type, payload)
      VALUES (?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      notification.notification_id,
      notification.source_system,
      notification.target_system,
      notification.notification_type,
      notification.payload
    ]);
  }
  
  async storeSubscription(clientId, systemType, notificationTypes) {
    const query = `
      INSERT OR REPLACE INTO notification_subscriptions 
      (subscription_id, client_id, system_type, notification_types)
      VALUES (?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      `sub_${clientId}`,
      clientId,
      systemType,
      JSON.stringify(notificationTypes)
    ]);
  }
  
  async updateSystemStatus(systemId, systemType, status) {
    const query = `
      INSERT OR REPLACE INTO system_status 
      (system_id, system_type, status, last_seen)
      VALUES (?, ?, ?, datetime('now'))
    `;
    
    await this.runQuery(query, [systemId, systemType, status]);
  }
  
  async updateNotificationStatus(notificationId, status) {
    const query = `
      UPDATE notifications 
      SET status = ? 
      WHERE notification_id = ?
    `;
    
    await this.runQuery(query, [status, notificationId]);
  }
  
  async markNotificationDelivered(notificationId, clientId) {
    const query = `
      UPDATE notifications 
      SET status = 'delivered', delivered_at = datetime('now')
      WHERE notification_id = ?
    `;
    
    await this.runQuery(query, [notificationId]);
  }
  
  async markNotificationAcknowledged(notificationId, clientId) {
    const query = `
      UPDATE notifications 
      SET status = 'acknowledged', acknowledged_at = datetime('now')
      WHERE notification_id = ?
    `;
    
    await this.runQuery(query, [notificationId]);
  }

  /**
   * Get system status
   */
  async getSystemStatus() {
    const query = `
      SELECT * FROM system_status 
      ORDER BY last_seen DESC
    `;
    
    return await this.executeQuery(query);
  }

  /**
   * Shutdown gracefully
   */
  async shutdown() {
    console.log('🛑 Shutting down notification system...');
    
    // Notify all clients
    this.broadcast(null, {
      type: 'system_shutdown',
      timestamp: new Date().toISOString()
    });
    
    // Close all connections
    for (const [clientId, client] of this.clients.entries()) {
      client.ws.close(1000, 'Server shutdown');
    }
    
    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
    }
    
    // Close database
    if (this.db) {
      this.db.close();
    }
    
    console.log('✅ Notification system shutdown complete');
  }
}

/**
 * Client library for connecting to the notification system
 */
class NotificationClient {
  constructor(systemType, wsUrl = 'ws://localhost:8765') {
    this.systemType = systemType;
    this.wsUrl = wsUrl;
    this.ws = null;
    this.clientId = null;
    this.connected = false;
    this.reconnectAttempts = 0;
    this.eventHandlers = new Map();
  }
  
  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      
      this.ws.on('open', () => {
        console.log('✅ Connected to notification system');
        this.connected = true;
        this.reconnectAttempts = 0;
        
        // Identify ourselves
        this.send({
          type: 'identify',
          systemType: this.systemType,
          systemInfo: {
            version: '1.0.0',
            platform: process.platform
          }
        });
        
        resolve();
      });
      
      this.ws.on('message', (data) => {
        this.handleMessage(JSON.parse(data));
      });
      
      this.ws.on('close', () => {
        console.log('❌ Disconnected from notification system');
        this.connected = false;
        this.attemptReconnect();
      });
      
      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      });
    });
  }
  
  handleMessage(message) {
    switch (message.type) {
      case 'welcome':
        this.clientId = message.clientId;
        break;
        
      case 'notification':
        this.emit('notification', message.notification);
        break;
        
      case 'query_result':
        this.emit(`query_${message.queryId}`, message.results);
        break;
        
      case 'heartbeat':
        this.send({ type: 'heartbeat' });
        break;
    }
  }
  
  subscribe(notificationTypes) {
    this.send({
      type: 'subscribe',
      notificationTypes
    });
  }
  
  notify(targetSystem, notificationType, payload) {
    this.send({
      type: 'notify',
      sourceSystem: this.systemType,
      targetSystem,
      notificationType,
      payload
    });
  }
  
  acknowledge(notificationId) {
    this.send({
      type: 'acknowledge',
      notificationId
    });
  }
  
  async query(sql, params = []) {
    const queryId = uuidv4();
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Query timeout'));
      }, 30000);
      
      this.once(`query_${queryId}`, (results) => {
        clearTimeout(timeout);
        resolve(results);
      });
      
      this.send({
        type: 'query',
        queryId,
        query: sql,
        params
      });
    });
  }
  
  send(message) {
    if (this.connected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
  
  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }
  
  once(event, handler) {
    const wrappedHandler = (...args) => {
      handler(...args);
      this.off(event, wrappedHandler);
    };
    this.on(event, wrappedHandler);
  }
  
  off(event, handler) {
    if (this.eventHandlers.has(event)) {
      const handlers = this.eventHandlers.get(event);
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }
  
  emit(event, ...args) {
    if (this.eventHandlers.has(event)) {
      const handlers = this.eventHandlers.get(event);
      handlers.forEach(handler => handler(...args));
    }
  }
  
  attemptReconnect() {
    if (this.reconnectAttempts < 5) {
      this.reconnectAttempts++;
      console.log(`🔄 Attempting reconnect (${this.reconnectAttempts}/5)...`);
      
      setTimeout(() => {
        this.connect().catch(err => {
          console.error('Reconnect failed:', err);
        });
      }, 5000 * this.reconnectAttempts);
    }
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Export modules
module.exports = {
  RealtimeNotificationSystem,
  NotificationClient
};

// Example usage
if (require.main === module) {
  // Start server
  const server = new RealtimeNotificationSystem();
  server.initialize().then(() => {
    console.log('🚀 Notification server ready');
    
    // Example: Create Claude Desktop client
    const desktopClient = new NotificationClient('claude_desktop');
    desktopClient.connect().then(() => {
      desktopClient.subscribe(['analysis_complete', 'prediction_complete']);
      
      desktopClient.on('notification', (notification) => {
        console.log('📬 Desktop received:', notification);
      });
    });
    
    // Example: Create Claude Code client
    const codeClient = new NotificationClient('claude_code');
    codeClient.connect().then(() => {
      codeClient.subscribe(['all']);
      
      // Simulate sending a notification
      setTimeout(() => {
        codeClient.notify('claude_desktop', 'analysis_complete', {
          campaign_id: 'test_001',
          score: 0.85
        });
      }, 2000);
    });
  });
  
  // Graceful shutdown
  process.on('SIGINT', async () => {
    await server.shutdown();
    process.exit(0);
  });
}