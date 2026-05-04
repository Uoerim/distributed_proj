import math
from typing import List, Dict, Set

class TextPreprocessor:
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'same', 'so',
            'than', 'too', 'very', 'as', 'if', 'from', 'by', 'with', 'about'
        }
    
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]
    
    def preprocess(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        return self.remove_stopwords(tokens)

class ScoringUtils:
    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        if score < min_val:
            return min_val
        if score > max_val:
            return max_val
        return score
    
    @staticmethod
    def combine_scores(keyword: float, semantic: float, kw_weight: float = 0.6) -> float:
        sem_weight = 1.0 - kw_weight
        combined = (keyword * kw_weight) + (semantic * sem_weight)
        return ScoringUtils.normalize_score(combined)
    
    @staticmethod
    def exponential_decay(score: float, decay_factor: float = 0.95) -> float:
        return score * decay_factor

class BM25Scorer:
    def __init__(self, documents: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(len(d.get('text', '').split()) for d in documents) / max(1, len(documents))
        self.idf_cache = {}
        self._build_idf()
    
    def _build_idf(self):
        doc_count = len(self.documents)
        word_docs = {}
        
        for doc in self.documents:
            words = set(doc.get('text', '').lower().split())
            for word in words:
                word_docs[word] = word_docs.get(word, 0) + 1
        
        for word, count in word_docs.items():
            self.idf_cache[word] = math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0)
    
    def score_document(self, query: str, doc_idx: int) -> float:
        query_words = query.lower().split()
        doc_text = self.documents[doc_idx].get('text', '').lower()
        doc_words = doc_text.split()
        doc_len = len(doc_words)
        
        score = 0.0
        for word in query_words:
            if not word:
                continue
            
            word_freq = doc_text.count(word)
            idf = self.idf_cache.get(word, 0.0)
            
            numerator = word_freq * (self.k1 + 1)
            denominator = word_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return score

def rank_documents(scores: Dict[int, float], top_k: int = 3) -> List[int]:
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_scores[:top_k]]
