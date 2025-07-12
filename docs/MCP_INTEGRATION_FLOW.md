# MCP Integration Flow Diagrams

## 1. Overall Architecture

```mermaid
graph TB
    subgraph "Claude Interfaces"
        CD[Claude Desktop<br/>SQL Interface]
        CC[Claude Code<br/>Python/API]
    end
    
    subgraph "MCP Layer"
        MCP[MCP SQLite Server<br/>node index.js]
        DB[(SQLite Database<br/>database.sqlite)]
    end
    
    subgraph "JamPacked Intelligence"
        JP[JamPacked Suite<br/>Python Backend]
        CI[Creative Intelligence<br/>DAIVID-like]
        MA[Multimodal Analysis<br/>Quilt.AI-like]
        PD[Pattern Discovery<br/>Evolutionary]
        AW[Award Recognition<br/>Tracking]
        CSR[CSR Analysis<br/>Authenticity]
    end
    
    CD <-->|SQL Queries| MCP
    CC <-->|Python API| JP
    MCP <--> DB
    JP <--> DB
    
    JP --> CI
    JP --> MA
    JP --> PD
    JP --> AW
    JP --> CSR
```

## 2. Data Flow Example

```mermaid
sequenceDiagram
    participant User
    participant ClaudeCode as Claude Code
    participant JamPacked
    participant SQLite as SQLite DB
    participant ClaudeDesktop as Claude Desktop
    
    User->>ClaudeCode: Upload creative materials
    ClaudeCode->>JamPacked: analyze_campaign_via_mcp()
    
    JamPacked->>JamPacked: Extract 200+ variables
    Note over JamPacked: Visual features<br/>Text analysis<br/>Award prediction<br/>CSR scoring
    
    JamPacked->>SQLite: INSERT INTO jampacked_creative_analysis
    JamPacked->>SQLite: INSERT INTO jampacked_pattern_discoveries
    JamPacked->>SQLite: INSERT INTO jampacked_cultural_insights
    
    JamPacked-->>ClaudeCode: Return campaign_id & results
    ClaudeCode-->>User: Analysis complete
    
    User->>ClaudeDesktop: Query results
    ClaudeDesktop->>SQLite: SELECT * FROM jampacked_creative_analysis
    SQLite-->>ClaudeDesktop: Return results
    ClaudeDesktop-->>User: Display insights
```

## 3. Variable Extraction Pipeline

```mermaid
graph LR
    subgraph "Input Materials"
        IMG[Images]
        TXT[Text]
        VID[Videos]
        AUD[Audio]
    end
    
    subgraph "Feature Extraction"
        VF[Visual Features<br/>Complexity, Color, Faces]
        TF[Text Features<br/>Sentiment, Urgency, Readability]
        AF[Audio Features<br/>Tempo, Energy, Voice]
        MF[Multimodal Fusion]
    end
    
    subgraph "Intelligence Analysis"
        ATT[Attention Prediction]
        EMO[Emotion Analysis]
        BR[Brand Recall]
        AWD[Award Potential]
        CSR[CSR Authenticity]
    end
    
    subgraph "Storage"
        DB[(MCP SQLite<br/>200+ Variables)]
    end
    
    IMG --> VF
    TXT --> TF
    VID --> VF
    VID --> AF
    AUD --> AF
    
    VF --> MF
    TF --> MF
    AF --> MF
    
    MF --> ATT
    MF --> EMO
    MF --> BR
    MF --> AWD
    MF --> CSR
    
    ATT --> DB
    EMO --> DB
    BR --> DB
    AWD --> DB
    CSR --> DB
```

## 4. Cross-Interface Workflow

```mermaid
stateDiagram-v2
    [*] --> ClaudeCode: Start Analysis
    
    ClaudeCode --> Processing: Run JamPacked Analysis
    Processing --> Database: Store Results
    
    Database --> ClaudeDesktop: Query Results
    Database --> ClaudeCode: Get Insights
    
    ClaudeDesktop --> Report: Generate Business Report
    ClaudeCode --> DeepDive: Advanced Analysis
    
    Report --> [*]: Complete
    DeepDive --> Database: Update Findings
    DeepDive --> [*]: Complete
```

## 5. Award & CSR Analysis Flow

```mermaid
graph TD
    subgraph "Award Recognition"
        A1[Campaign Data] --> A2[Award Database<br/>Cannes, D&AD, One Show]
        A2 --> A3[Prestige Scoring]
        A3 --> A4[Time Decay]
        A4 --> A5[Category Diversity]
        A5 --> AS[Award Score: 0-1]
    end
    
    subgraph "CSR Analysis"
        C1[Content Analysis] --> C2[Theme Classification]
        C2 --> C3[Prominence Measurement]
        C3 --> C4[Authenticity Check]
        C4 --> C5[Audience Alignment]
        C5 --> CS[CSR Score: 0-1]
    end
    
    AS --> DB[(SQLite DB)]
    CS --> DB
    
    DB --> CD[Claude Desktop<br/>Business Insights]
    DB --> CC[Claude Code<br/>Deep Analysis]
```

## 6. Real-time Integration Benefits

```mermaid
graph TB
    subgraph "Traditional Approach"
        T1[Analyze in Tool A] --> T2[Export Results]
        T2 --> T3[Import to Tool B]
        T3 --> T4[Re-analyze]
        T4 --> T5[Export Again]
        style T1 fill:#ffcccc
        style T2 fill:#ffcccc
        style T3 fill:#ffcccc
        style T4 fill:#ffcccc
        style T5 fill:#ffcccc
    end
    
    subgraph "JamPacked MCP Integration"
        J1[Analyze in Claude Code] --> DB[(Shared SQLite)]
        DB --> J2[Query in Claude Desktop]
        DB --> J3[Continue in Claude Code]
        style J1 fill:#ccffcc
        style DB fill:#ccffcc
        style J2 fill:#ccffcc
        style J3 fill:#ccffcc
    end
    
    T5 -.->|"Time: Hours<br/>Errors: High"| T1
    J2 & J3 -.->|"Time: Seconds<br/>Errors: None"| DB
```

## Key Integration Points

### 1. **Database Location**
```
/Users/pulser/Documents/GitHub/mcp-sqlite-server/data/database.sqlite
```

### 2. **MCP Server**
```
/Users/pulser/Documents/GitHub/mcp-sqlite-server/dist/index.js
```

### 3. **JamPacked Integration**
```python
from jampacked_sqlite_integration import analyze_campaign_via_mcp
```

### 4. **SQL Access**
```sql
SELECT * FROM jampacked_creative_analysis;
SELECT * FROM jampacked_pattern_discoveries;
SELECT * FROM jampacked_cultural_insights;
SELECT * FROM jampacked_optimizations;
```

This architecture enables seamless, real-time collaboration between Claude Desktop and Claude Code while leveraging the full power of JamPacked's creative intelligence capabilities!