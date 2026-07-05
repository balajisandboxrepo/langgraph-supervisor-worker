# Technical Architecture

## System Overview

This document describes the architecture of the LangGraph + Prefect orchestration system.

### Data Flow

```mermaid
%%{init: {'flowchart': {'curve': 'linear', 'rankSpacing': 150, 'nodeSpacing': 100}, 'theme': 'base', 'primaryColor': '#ffffff', 'primaryBorderColor': '#333333', 'lineColor': '#FF6B6B', 'tertiaryLineColor': '#FF6B6B', 'background': '#f5f5f5', 'mainBkg': '#ffffff', 'secondBkg': '#f0f0f0', 'tertiaryBkg': '#ffffff'}}%%
graph LR
    A["<b>Webhooks / API</b>"]
    B["<b>Enterprise Applications</b>"]
    
    subgraph Prefect ["<b>Prefect Orchestration Layer</b>"]
        PF["<b>Flow: Orchestrator</b>"]
        T1["Task: Ingest & Preprocess"]
        T2["<b>Task: Run LangGraph</b>"]
        T3["Task: Postprocess & Notify"]
        PF --> T1 --> T2 --> T3
    end
    
    subgraph LangGraph ["<b>LangGraph Agentic Layer</b>"]
        Start(["<b>START</b>"])
        State["<b>State Manager</b>"]
        Sup["<b>Supervisor</b>"]
        W1["<b>Worker:</b><br/>DocumentExtract"]
        W2["<b>Worker:</b><br/>Object Detection"]
        W3["<b>Worker:</b><br/>Classification"]
        End(["<b>END</b>"])
        
        Start --> State --> Sup
        Sup --> W1
        Sup --> W2
        Sup --> W3
        W1 --> End
        W2 --> End
        W3 --> End
    end
    
    subgraph Ext ["<b>External Services & LLMs</b>"]
        APIcalls["<b>API Calls</b>"]
        VectorDB["<b>Vector DB</b><br/>FAISS"]
        LLM["<b>LLM</b><br/>Qwen / OpenAI"]
        Slack["<b>Slack</b><br/>Email"]
    end
    
    A --> PF
    B --> T1
    T2 --> Start
    W1 -.-> APIcalls
    W2 -.-> VectorDB
    Sup -.-> LLM
    T3 --> Slack
    
    classDef prefect fill:#2b5c8f,stroke:#fff,stroke-width:3px,color:#fff,font-size:16px;
    classDef langgraph fill:#138a72,stroke:#fff,stroke-width:3px,color:#fff,font-size:16px;
    classDef external fill:#f39c12,stroke:#fff,stroke-width:3px,color:#000,font-size:14px;
    classDef source fill:#555,stroke:#fff,stroke-width:2px,color:#fff,font-size:14px;
    classDef core fill:#d32f2f,stroke:#fff,stroke-width:4px,color:#fff,font-size:16px;
    
    class PF,T1,T2,T3 prefect;
    class Start,State,Sup,W1,W2,W3,End langgraph;
    class APIcalls,VectorDB,LLM,Slack external;
    class A,B source;
    class Sup core;
```

## Component Descriptions

### Data Sources
- **Webhooks / API**: External event triggers
- **Enterprise Applications**: Data ingestion sources

### Prefect Orchestration Layer
- **Flow**: Main orchestrator managing the workflow
- **Task 1**: Data ingestion and preprocessing
- **Task 2**: Invokes LangGraph agents
- **Task 3**: Postprocessing and notifications

### LangGraph Agentic Layer
- **Supervisor**: Routes tasks to appropriate workers
- **Workers**: Execute specialized tasks (Document Extraction, Object Detection, Classification)
- **State Manager**: Maintains conversation and processing state

### External Services
- **Vector DB (FAISS)**: Semantic search and embeddings
- **LLM (Qwen/OpenAI)**: Language model inference
- **API Calls**: Third-party integrations
- **Notifications**: Slack/Email alerts
