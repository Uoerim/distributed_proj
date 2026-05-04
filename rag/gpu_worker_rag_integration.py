import os
import time

import requests
from dotenv import load_dotenv

from rag_retriever import RAGRetriever

load_dotenv()


class GPUWorkerWithRAG:
    def __init__(self, worker_id: int, kb_path: str = "rag/kb/documents.json"):
        self.id = worker_id
        self.rag = RAGRetriever(kb_path)
        self.processed_count = 0
        self.total_latency = 0.0

    def process(self, request):
        start_time = time.time()

        query = request.get('query', '')
        request_id = request.get('id')

        try:
            context = self.rag.retrieve(query, top_k=3)

            result = self._run_llm_inference(query, context)

            latency = time.time() - start_time
            self.processed_count += 1
            self.total_latency += latency

            return {
                "id": request_id,
                "result": result,
                "latency": latency,
                "worker_id": self.id,
                "context_used": len(context) > 0
            }

        except Exception as e:
            latency = time.time() - start_time
            return {
                "id": request_id,
                "result": f"Error: {str(e)}",
                "latency": latency,
                "worker_id": self.id,
                "error": True
            }

    def _run_llm_inference(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)

        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model       = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",  # required when OLLAMA_HOST is an ngrok URL
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]

    def _build_prompt(self, query: str, context: str) -> str:
        if context:
            return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            return f"Question: {query}\n\nAnswer:"

    def get_stats(self) -> dict:
        avg_latency = self.total_latency / max(1, self.processed_count)
        return {
            "worker_id": self.id,
            "requests_processed": self.processed_count,
            "avg_latency": avg_latency,
            "total_latency": self.total_latency
        }


class RAGStats:
    def __init__(self):
        self.query_count = 0
        self.retrieval_times = []
        self.cache_hits = 0

    def log_retrieval(self, retrieval_time: float, cache_hit: bool = False):
        self.query_count += 1
        self.retrieval_times.append(retrieval_time)
        if cache_hit:
            self.cache_hits += 1

    def get_avg_retrieval_time(self) -> float:
        if not self.retrieval_times:
            return 0.0
        return sum(self.retrieval_times) / len(self.retrieval_times)

    def get_cache_hit_rate(self) -> float:
        if self.query_count == 0:
            return 0.0
        return self.cache_hits / self.query_count