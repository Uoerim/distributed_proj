import json
import os
from typing import List, Dict, Tuple
import math

class RAGRetriever:
    def __init__(self, kb_path: str = "rag/kb/documents.json"):
        self.kb_path = kb_path
        self.documents = []
        self.word_index = {}
        self.doc_freqs = {}
        self.idf_cache = {}
        
        self._load_kb()
        self._build_indices()

    def _load_kb(self):
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"KB not found at {self.kb_path}")
        
        with open(self.kb_path, 'r') as f:
            data = json.load(f)
        
        self.documents = data if isinstance(data, list) else data.get('documents', [])
        
        if not self.documents:
            raise ValueError("No documents in KB")

    def _build_indices(self):
        for doc_id, doc in enumerate(self.documents):
            text = doc.get('text', '').lower()
            words = set(text.split())
            
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = []
                self.word_index[word].append(doc_id)
            
            self.doc_freqs[doc_id] = len(text.split())

    def _idf(self, word: str) -> float:
        if word in self.idf_cache:
            return self.idf_cache[word]
        
        doc_count = len([d for d in self.documents if word in d.get('text', '').lower()])
        if doc_count == 0:
            idf_val = 0
        else:
            idf_val = math.log(len(self.documents) / doc_count)
        
        self.idf_cache[word] = idf_val
        return idf_val

    def _keyword_score(self, query: str, doc_id: int) -> float:
        query_words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
        doc_text = self.documents[doc_id].get('text', '').lower()
        
        score = 0.0
        for word in query_words:
            if word in doc_text:
                idf = self._idf(word)
                tf = doc_text.count(word) / max(1, self.doc_freqs.get(doc_id, 1))
                score += tf * (idf + 1.0)
        
        return score

    def _embedding_score(self, query: str, doc_id: int) -> float:
        q_words = set([w.strip().lower() for w in query.split() if len(w.strip()) > 2])
        d_words = set(self.documents[doc_id].get('text', '').lower().split())
        
        if not q_words or not d_words:
            return 0.0
        
        intersection = len(q_words & d_words)
        union = len(q_words | d_words)
        
        jaccard = intersection / union if union > 0 else 0
        return jaccard

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not query or not query.strip():
            return ""
        
        scores = {}
        for doc_id in range(len(self.documents)):
            kw_score = self._keyword_score(query, doc_id)
            emb_score = self._embedding_score(query, doc_id)
            
            combined = (0.6 * kw_score) + (0.4 * emb_score)
            if combined > 0:
                scores[doc_id] = combined
        
        if not scores:
            return self.documents[0].get('text', '') if self.documents else ""
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [self.documents[doc_id].get('text', '') for doc_id, _ in ranked[:top_k]]
        
        return "\n---\n".join(top_docs)

    def retrieve_with_scores(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        scores = {}
        for doc_id in range(len(self.documents)):
            kw_score = self._keyword_score(query, doc_id)
            emb_score = self._embedding_score(query, doc_id)
            combined = (0.6 * kw_score) + (0.4 * emb_score)
            if combined > 0:
                scores[doc_id] = combined
        
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[doc_id].get('text', ''), score) for doc_id, score in ranked]

    def get_kb_size(self) -> int:
        return len(self.documents)
