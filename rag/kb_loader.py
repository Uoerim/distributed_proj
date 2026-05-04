import json
import os
from typing import List, Dict
from rag_utils import TextPreprocessor

class KBLoader:
    def __init__(self, kb_path: str = "rag/kb/documents.json"):
        self.kb_path = kb_path
        self.documents = []
        self.processed_docs = {}
        self.metadata = {}
    
    def load(self) -> List[Dict]:
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"KB file not found: {self.kb_path}")
        
        with open(self.kb_path, 'r') as f:
            self.documents = json.load(f)
        
        if not isinstance(self.documents, list):
            self.documents = self.documents.get('documents', [])
        
        return self.documents
    
    def preprocess_documents(self) -> Dict[int, List[str]]:
        preprocessor = TextPreprocessor()
        
        for doc in self.documents:
            doc_id = doc.get('id', self.documents.index(doc))
            text = doc.get('text', '')
            
            tokens = preprocessor.preprocess(text)
            self.processed_docs[doc_id] = tokens
        
        return self.processed_docs
    
    def get_document_by_id(self, doc_id: int) -> Dict:
        for doc in self.documents:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> List[Dict]:
        return self.documents
    
    def get_processed_tokens(self, doc_id: int) -> List[str]:
        return self.processed_docs.get(doc_id, [])
    
    def save_metadata(self, metadata_path: str = "rag/kb/metadata.json"):
        metadata = {
            'total_documents': len(self.documents),
            'preprocessed': len(self.processed_docs),
            'sample_tokens': {str(k): v[:10] for k, v in list(self.processed_docs.items())[:3]}
        }
        
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def initialize_kb(kb_path: str = "rag/kb/documents.json") -> tuple:
    loader = KBLoader(kb_path)
    docs = loader.load()
    loader.preprocess_documents()
    loader.save_metadata()
    
    return loader, docs
