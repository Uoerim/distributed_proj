# Distributed LLM Load Balancing System

## Folder Structure

```
Root_Folder/
├── README.md
├── client/                    # Simulates 1000 concurrent users sending requests
├── common/                    # Shared models (Request, Response classes)
├── lb/                        # Load balancer
├── master/                    # Master scheduler
├── workers/                   # GPU worker nodes 
├── llm/                       # LLM inference engine
├── rag/                       # RAG module - retrieves context from knowledge base
├── ragkb/                     # Knowledge base storage
└── tests/                     # Test suite
```
