# Distributed LLM Load Balancing System

A distributed inference system for handling 1000+ concurrent LLM requests across multiple worker nodes. The system implements three load balancing strategies (Round Robin, Least Connections, Load-Aware) and integrates Retrieval-Augmented Generation (RAG) to enrich LLM responses.

## Architecture

- **Load Balancer** — Routes incoming requests across worker nodes using configurable strategies
- **Master Scheduler** — Coordinates request distribution and monitors worker health
- **GPU Workers** — Process LLM inference tasks with RAG context retrieval
- **RAG Module** — Retrieves relevant documents to augment LLM prompts
- **Client Layer** — Simulates concurrent users sending requests

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally (or accessible via network)
- A local LLM model (default: `llama3.2:1b`)

## Installation

```bash
# Clone or extract the project
cd distributed_proj

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your Ollama host and model details
```

## Configuration

Edit `.env` to configure:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
```

If running multiple Ollama instances on different hosts:

```env
OLLAMA_HOST_0=http://localhost:11434
OLLAMA_HOST_1=http://another-host:11434
```

## Running the System

### Basic usage (default: 100 users, Round Robin)

```bash
python main.py
```

### Custom configuration

```bash
# 500 concurrent users with Least Connections strategy
python main.py --strategy least_connections --users 500

# Load-Aware routing with 300 users
python main.py --strategy load_aware --users 300

# Run all three strategies back-to-back and compare results
python main.py --compare-strategies --users 200
```

### Load Testing with Failure Simulation

```bash
# Kill Worker 1 after 5 seconds to test fault tolerance
python main.py --users 300 --kill-worker 1 --kill-after 5

# Kill worker without recovery
python main.py --users 300 --kill-worker 0 --no-recovery
```

### Available CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--strategy` | round_robin | Load balancing strategy: `round_robin`, `least_connections`, `load_aware` |
| `--users` | 100 | Number of simulated concurrent users |
| `--max-concurrent` | 50 | Max requests in-flight simultaneously |
| `--compare-strategies` | - | Run all 3 strategies sequentially and compare |
| `--kill-worker` | - | Worker index to terminate mid-test (for fault tolerance testing) |
| `--kill-after` | 5 | Seconds before worker termination |
| `--no-recovery` | - | Killed worker is NOT auto-recovered |

## System Behavior

### Load Balancing Strategies

**Round Robin** — Distributes requests equally across workers in order.

**Least Connections** — Routes to the worker with fewest active requests.

**Load-Aware** — Considers worker load and response time history.

### RAG Integration

Each worker retrieves relevant context from `rag/kb/documents.json` before querying the LLM. Context is appended to the user prompt for better response quality.

### Fault Tolerance

When a worker fails or is killed:
- The system detects the failure
- Failed requests are requeued to available workers
- System continues accepting new requests
- Failed worker is automatically recovered (unless `--no-recovery` is set)

## Project Structure

```
distributed_proj/
├── main.py                 # Entry point
├── client/                 # User simulation and load generation
├── master/                 # Scheduler and task coordination
├── workers/                # GPU worker nodes
├── lb/                     # Load balancer implementation
├── llm/                    # LLM inference layer (Ollama integration)
├── rag/                    # RAG retriever and knowledge base
├── common/                 # Shared models and utilities
├── tests/                  # Unit tests
└── .env.example            # Environment template
```

## Metrics

The system tracks:
- **Latency** — End-to-end request processing time
- **Throughput** — Requests processed per second
- **Worker Utilization** — CPU/memory usage per worker
- **Success Rate** — Percentage of completed requests

Metrics are printed to console and can be exported for analysis.

## Performance Notes

- Expects Ollama to be running and accessible at the configured host
- First request to each model will take longer (cold start)
- RAG retrieval uses TF-IDF ranking; performance scales with knowledge base size
- Network latency between load balancer and workers is included in measurements

## Troubleshooting

**Connection refused to Ollama:**
```bash
# Verify Ollama is running
curl http://localhost:11434/api/models

# Check .env configuration
cat .env
```

**Workers failing to start:**
```bash
# Ensure required model is available
ollama pull llama3.2:1b

# Check logs for detailed errors
python main.py 2>&1 | head -50
```

**No documents in RAG knowledge base:**
```bash
# Verify KB file exists and is populated
ls -la rag/kb/documents.json
wc -l rag/kb/documents.json
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- Load Balancing Algorithms: Round Robin, Least Connections
- Retrieval-Augmented Generation: TF-IDF ranking with cosine similarity

## Team

Team 35 || CSE 354 - Distributed Computing, Ain Shams University
