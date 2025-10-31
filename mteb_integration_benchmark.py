#!/usr/bin/env python3
"""
MTEB-Style Integration Benchmark for Local Embedding Models
Comprehensive benchmark following MTEB leaderboard methodology for retrieval tasks.
Tests local models against real academic papers with intensive evaluation.

MODES:
------
1. TITLE + ABSTRACT Mode (default)
   - Embeddings: title + abstract of each paper
   - Queries: based on title, abstract, category, author, year
   - Documents: ~100-200 papers
   - Usage: python mteb_integration_benchmark.py
   
2. FULL TEXT Mode (new)
   - Embeddings: complete full text split into chunks
   - Downloads: actual PDF files from arXiv
   - Text Extraction: uses PyPDF2 or pdfplumber
   - Chunking: intelligent sentence-based chunking (512 char chunks)
   - Queries: extracted from chunk content
   - Documents: thousands of chunks from papers
   - Usage: python mteb_integration_benchmark.py --full-text

INSTALLATION REQUIREMENTS:
--------------------------
For Full Text mode, install additional dependencies:
  pip install PyPDF2 pdfplumber

EXAMPLES:
---------
# Run standard benchmark with title+abstract
python mteb_integration_benchmark.py

# Run full-text benchmark with custom model
python mteb_integration_benchmark.py --full-text --model "Qwen/Qwen3-Embedding-0.6B"

# Run with custom dataset size
python mteb_integration_benchmark.py --full-text --dataset-size 50 --test-queries 25

# Specify custom cache directory
python mteb_integration_benchmark.py --full-text --cache-dir ~/.cache/zotero_benchmark

PERFORMANCE NOTES:
------------------
- Full-text mode downloads actual PDFs (~5-20 MB per paper)
- PDF extraction takes additional time (5-30 seconds per paper)
- More chunks = more embeddings to generate (1000s vs 100s)
- Results are cached to avoid re-downloading
- Full-text provides more realistic evaluation of embedding quality

METRICS:
--------
- Precision@k: exact chunk retrieval accuracy at top-k
- Recall@k: chunk retrieval coverage at top-k  
- NDCG@k: ranking quality considering position
- MRR: mean reciprocal rank of first relevant result
- MAP: mean average precision across all queries
- Embedding time: time to generate all embeddings
- Search time: time to perform retrieval across all queries
"""

import sys
import time
import json
import csv
import re
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import numpy as np
from dataclasses import dataclass, field
import tempfile
import os
import argparse
from urllib.parse import urlparse
import gzip
import pickle
import math

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from zotero_mcp.chroma_client import LocalEmbeddingFunction, create_chroma_client
import chromadb.utils.embedding_functions

DEFAULT_QUERY_FILE = Path(__file__).parent / "benchmarks" / "queries" / "semantic_retrieval_queries.json"
DEFAULT_MODELS = ["chromadb-default", "Qwen/Qwen3-Embedding-0.6B"]
DEFAULT_MODES = ["abstract", "full_text"]
SCOREBOARD_METRICS = ["accuracy", "precision", "recall", "mrr", "ndcg", "map_score"]


def slugify_model_name(name: str) -> str:
    """Create filesystem-safe slug from model name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return slug or "model"


def _init_bucket() -> Dict[str, Any]:
    return {
        "queries": 0,
        "queries_with_relevance": 0,
        "hit_at_1_count": 0,
        "hit_at_5_count": 0,
        "hit_at_10_count": 0,
        "best_rank_sum": 0.0,
        "best_rank_hits": 0,
    }


def _update_bucket(bucket: Dict[str, Any], stat: Dict[str, Any]):
    bucket["queries"] += 1
    if stat.get("has_relevance"):
        bucket["queries_with_relevance"] += 1
    if stat.get("top1_hit"):
        bucket["hit_at_1_count"] += 1
    if stat.get("top5_hit"):
        bucket["hit_at_5_count"] += 1
    if stat.get("top10_hit"):
        bucket["hit_at_10_count"] += 1
    best_rank = stat.get("best_rank")
    if best_rank is not None:
        bucket["best_rank_sum"] += best_rank
        bucket["best_rank_hits"] += 1


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    queries = bucket.get("queries", 0)
    if queries > 0:
        bucket["hit_at_1"] = bucket["hit_at_1_count"] / queries
        bucket["hit_at_5"] = bucket["hit_at_5_count"] / queries
        bucket["hit_at_10"] = bucket["hit_at_10_count"] / queries
    else:
        bucket["hit_at_1"] = bucket["hit_at_5"] = bucket["hit_at_10"] = 0.0
    
    hits = bucket.get("best_rank_hits", 0)
    if hits > 0:
        bucket["mean_best_rank"] = bucket["best_rank_sum"] / hits
    else:
        bucket["mean_best_rank"] = None
    
    # Remove intermediate counters
    for key in ["hit_at_1_count", "hit_at_5_count", "hit_at_10_count", "best_rank_sum", "best_rank_hits"]:
        bucket.pop(key, None)
    
    return bucket


@dataclass
class MTEBResult:
    """MTEB-style benchmark results."""
    model_name: str
    dataset_name: str
    task_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    map_score: float
    embedding_time: float
    search_time: float
    total_documents: int
    total_queries: int
    success_rate: float
    use_full_text: bool = False
    per_query_stats: List[Dict[str, Any]] = field(default_factory=list)


def compute_query_breakdown(results: List[MTEBResult]) -> Dict[str, Any]:
    breakdown: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for result in results:
        mode = "full_text" if result.use_full_text else "abstract"
        model = result.model_name
        model_entry = breakdown.setdefault(mode, {}).setdefault(model, {
            "overall": _init_bucket(),
            "by_category": {},
            "by_difficulty": {}
        })
        
        for stat in result.per_query_stats:
            _update_bucket(model_entry["overall"], stat)
            
            category = stat.get("category") or "uncategorized"
            cat_bucket = model_entry["by_category"].setdefault(category, _init_bucket())
            _update_bucket(cat_bucket, stat)
            
            difficulty = stat.get("difficulty")
            if difficulty is None:
                diff_key = "unspecified"
            else:
                diff_key = str(difficulty)
            diff_bucket = model_entry["by_difficulty"].setdefault(diff_key, _init_bucket())
            _update_bucket(diff_bucket, stat)
    
    # Finalize ratios
    for mode_dict in breakdown.values():
        for model_dict in mode_dict.values():
            model_dict["overall"] = _finalize_bucket(model_dict["overall"])
            model_dict["by_category"] = {
                category: _finalize_bucket(bucket)
                for category, bucket in model_dict["by_category"].items()
            }
            model_dict["by_difficulty"] = {
                difficulty: _finalize_bucket(bucket)
                for difficulty, bucket in model_dict["by_difficulty"].items()
            }
    
    return breakdown


class TextChunker:
    """Split documents into meaningful chunks for embeddings."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target characters per chunk
            overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_id: str = "") -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source_id: Source document ID
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Split by sentences first
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence) + 2  # +2 for '. '
            
            # If adding this sentence would exceed chunk_size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    'text': chunk_text,
                    'source': source_id,
                    'char_count': len(chunk_text)
                })
                
                # Keep last sentence for overlap
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append({
                'text': chunk_text,
                'source': source_id,
                'char_count': len(chunk_text)
            })
        
        return chunks


class AcademicPaperDataset:
    """Download and manage real academic paper datasets for benchmarking."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mteb_benchmark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_chunker = TextChunker()
    
    def download_arxiv_pdf(self, arxiv_url: str) -> Optional[str]:
        """
        Download full text PDF from arXiv.
        
        Args:
            arxiv_url: arXiv paper URL or ID
            
        Returns:
            Path to downloaded PDF or None if failed
        """
        try:
            # Extract arxiv ID
            if 'arxiv.org' in arxiv_url:
                arxiv_id = arxiv_url.split('/abs/')[-1]
            else:
                arxiv_id = arxiv_url
            
            # Create PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Download PDF
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            
            # Save to cache
            pdf_path = self.cache_dir / f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            return str(pdf_path)
        except Exception as e:
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF using PyPDF or similar.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            import PyPDF2
            
            text = []
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            
            return ' '.join(text)
        except ImportError:
            # Fallback: try with pdfplumber
            try:
                import pdfplumber
                text = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                return ' '.join(text)
            except ImportError:
                print("⚠️  No PDF extraction library available (install PyPDF2 or pdfplumber)")
                return None
    
    def fetch_paper_by_arxiv_id(self, arxiv_id: str, use_full_text: bool = False) -> Optional[Dict]:
        """Fetch a specific paper by arXiv ID."""
        cache_file = self.cache_dir / f"arxiv_paper_{arxiv_id.replace('/', '_')}_fulltext_{use_full_text}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                paper = json.load(f)
                return paper if isinstance(paper, dict) else None
        
        try:
            # Fetch single paper from arXiv API
            url = f"http://export.arxiv.org/api/query"
            params = {
                'id_list': arxiv_id,
                'max_results': 1
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_xml(response.text, use_full_text)
            if papers:
                paper = papers[0]
                with open(cache_file, 'w') as f:
                    json.dump(paper, f, indent=2)
                return paper
        except Exception as e:
            print(f"⚠️  Error fetching paper {arxiv_id}: {e}")
        
        return None
    
    def get_arxiv_papers(self, categories: List[str] = None, max_papers: int = 200, use_full_text: bool = False, required_arxiv_ids: Optional[List[str]] = None) -> List[Dict]:
        """Download real papers from arXiv for benchmarking.
        
        Args:
            categories: List of arXiv categories to fetch from
            max_papers: Maximum number of papers to fetch
            use_full_text: Whether to download full text PDFs
            required_arxiv_ids: Optional list of arXiv IDs that must be included
        """
        if categories is None:
            categories = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV', 'cs.IR', 'cs.NE', 'stat.ML']
        
        papers = []
        paper_urls_seen = set()
        
        # First, fetch any required papers specified by arXiv ID
        if required_arxiv_ids:
            print(f"📥 Fetching {len(required_arxiv_ids)} required papers from query file...")
            for arxiv_id in required_arxiv_ids:
                paper = self.fetch_paper_by_arxiv_id(arxiv_id, use_full_text)
                if paper:
                    paper_url = paper.get("url", "").strip().lower()
                    if paper_url and paper_url not in paper_urls_seen:
                        papers.append(paper)
                        paper_urls_seen.add(paper_url)
                        print(f"   ✓ Fetched: {paper.get('title', arxiv_id)[:60]}...")
        
        # Then fetch additional papers from categories to reach max_papers
        remaining = max_papers - len(papers)
        if remaining > 0:
            total_categories = len(categories)
            for idx, category in enumerate(categories):
                if remaining <= 0:
                    break
                categories_left = total_categories - idx
                per_category = max(1, math.ceil(remaining / categories_left))
                fetched = self._fetch_arxiv_category(category, per_category, use_full_text)
                
                # Deduplicate by URL
                for paper in fetched:
                    paper_url = paper.get("url", "").strip().lower()
                    if paper_url and paper_url not in paper_urls_seen:
                        papers.append(paper)
                        paper_urls_seen.add(paper_url)
                        remaining -= 1
                        if remaining <= 0:
                            break
        
        return papers[:max_papers]
    
    def _fetch_arxiv_category(self, category: str, max_papers: int, use_full_text: bool = False) -> List[Dict]:
        """Fetch real papers from arXiv API."""
        cache_file = self.cache_dir / f"arxiv_{category}_{max_papers}_fulltext_{use_full_text}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        print(f"📡 Fetching real papers from arXiv category: {category}")
        
        # Use arXiv API to get real papers
        papers = self._fetch_from_arxiv_api(category, max_papers, use_full_text)
        
        with open(cache_file, 'w') as f:
            json.dump(papers, f, indent=2)
        
        return papers
    
    def _fetch_from_arxiv_api(self, category: str, max_papers: int, use_full_text: bool = False) -> List[Dict]:
        """Fetch papers from arXiv API."""
        try:
            # arXiv API endpoint
            url = f"http://export.arxiv.org/api/query"
            params = {
                'search_query': f'cat:{category}',
                'start': 0,
                'max_results': max_papers,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_xml(response.text, use_full_text)
            return papers[:max_papers]
            
        except Exception as e:
            print(f"⚠️  Error fetching from arXiv API: {e}")
            print("📝 Falling back to synthetic papers...")
            return self._generate_synthetic_papers(category, max_papers, use_full_text)
    
    def _parse_arxiv_xml(self, xml_content: str, use_full_text: bool = False) -> List[Dict]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET
        
        papers = []
        root = ET.fromstring(xml_content)
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            try:
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                
                # Get authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name = author.find('{http://www.w3.org/2005/Atom}name').text.strip()
                    authors.append(name)
                
                # Get published date
                published = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()
                year = published.split('-')[0]
                
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()
                
                paper = {
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "year": int(year),
                    "category": "arXiv",
                    "url": arxiv_id,
                    "arxiv_id": arxiv_id.split('/abs/')[-1]
                }
                
                # Download and extract full text if requested
                if use_full_text:
                    print(f"   Downloading full text for: {title[:50]}...")
                    pdf_path = self.download_arxiv_pdf(arxiv_id)
                    if pdf_path:
                        full_text = self.extract_text_from_pdf(pdf_path)
                        if full_text:
                            paper["full_text"] = full_text
                            paper["full_text_chunks"] = self.text_chunker.chunk_text(full_text, paper["arxiv_id"])
                
                papers.append(paper)
            except Exception as e:
                continue
        
        return papers
    
    def _generate_synthetic_papers(self, category: str, count: int, use_full_text: bool = False) -> List[Dict]:
        """Generate high-quality synthetic academic papers for testing."""
        base_papers = {
            'cs.AI': [
                {
                    "title": "Deep Reinforcement Learning for Autonomous Navigation",
                    "abstract": "We present a novel approach to autonomous navigation using deep reinforcement learning algorithms. Our method combines Q-learning with neural networks to achieve robust path planning in dynamic environments. We demonstrate significant improvements over traditional methods on both simulated and real-world scenarios.",
                    "authors": ["John Smith", "Jane Doe"],
                    "year": 2023,
                    "category": "Artificial Intelligence"
                },
                {
                    "title": "Multi-Agent Systems for Distributed Problem Solving",
                    "abstract": "This paper explores multi-agent systems and their applications in distributed problem solving. We propose a new coordination mechanism that improves efficiency in collaborative tasks. Our experimental results show substantial performance gains in complex multi-agent environments.",
                    "authors": ["Alice Johnson", "Bob Wilson"],
                    "year": 2023,
                    "category": "Artificial Intelligence"
                }
            ],
            'cs.CL': [
                {
                    "title": "Transformer Architecture for Natural Language Understanding",
                    "abstract": "We investigate the effectiveness of transformer architectures in natural language understanding tasks. Our experiments show significant improvements in semantic parsing and question answering. The proposed model achieves state-of-the-art results on multiple benchmark datasets.",
                    "authors": ["Sarah Chen", "Mike Brown"],
                    "year": 2023,
                    "category": "Natural Language Processing"
                },
                {
                    "title": "Cross-Lingual Transfer Learning in Neural Machine Translation",
                    "abstract": "This work presents a novel approach to cross-lingual transfer learning for neural machine translation. We demonstrate improved performance across multiple language pairs. Our method reduces the need for parallel training data while maintaining translation quality.",
                    "authors": ["David Lee", "Emma Davis"],
                    "year": 2023,
                    "category": "Natural Language Processing"
                }
            ],
            'cs.LG': [
                {
                    "title": "Federated Learning with Differential Privacy",
                    "abstract": "We propose a new framework for federated learning that incorporates differential privacy guarantees. Our approach maintains model utility while ensuring user privacy. We prove theoretical privacy bounds and demonstrate empirical results on real-world datasets.",
                    "authors": ["Lisa Wang", "Tom Anderson"],
                    "year": 2023,
                    "category": "Machine Learning"
                },
                {
                    "title": "Graph Neural Networks for Molecular Property Prediction",
                    "abstract": "This paper presents graph neural networks for predicting molecular properties. We achieve state-of-the-art results on several benchmark datasets. Our approach leverages attention mechanisms and message passing for improved molecular representation learning.",
                    "authors": ["Maria Garcia", "James Taylor"],
                    "year": 2023,
                    "category": "Machine Learning"
                }
            ],
            'cs.CV': [
                {
                    "title": "Self-Supervised Learning for Computer Vision",
                    "abstract": "We explore self-supervised learning methods for computer vision tasks. Our approach reduces the need for labeled data while maintaining high performance. We demonstrate effectiveness across multiple vision tasks including classification and detection.",
                    "authors": ["Alex Kim", "Sophie White"],
                    "year": 2023,
                    "category": "Computer Vision"
                },
                {
                    "title": "3D Object Detection in Point Clouds",
                    "abstract": "This work presents a new method for 3D object detection in point cloud data. We achieve improved accuracy and efficiency compared to existing approaches. Our method handles sparse and noisy point cloud data effectively.",
                    "authors": ["Ryan Park", "Anna Green"],
                    "year": 2023,
                    "category": "Computer Vision"
                }
            ],
            'cs.IR': [
                {
                    "title": "Neural Information Retrieval with Dense Representations",
                    "abstract": "We propose a new approach to information retrieval using dense neural representations. Our method improves both precision and recall in document search tasks. We demonstrate effectiveness on large-scale retrieval benchmarks.",
                    "authors": ["Kevin Liu", "Rachel Brown"],
                    "year": 2023,
                    "category": "Information Retrieval"
                },
                {
                    "title": "Cross-Modal Retrieval for Multimedia Content",
                    "abstract": "This paper presents cross-modal retrieval techniques for multimedia content. We demonstrate effective search across text, images, and audio modalities. Our approach learns joint representations for improved cross-modal understanding.",
                    "authors": ["Daniel Kim", "Laura Wilson"],
                    "year": 2023,
                    "category": "Information Retrieval"
                }
            ],
            'cs.NE': [
                {
                    "title": "Spiking Neural Networks for Efficient Computing",
                    "abstract": "We present spiking neural networks for energy-efficient computing. Our approach mimics biological neural processing for improved efficiency. We demonstrate significant energy savings while maintaining computational performance.",
                    "authors": ["Neural Expert", "Bio Researcher"],
                    "year": 2023,
                    "category": "Neural Networks"
                }
            ],
            'stat.ML': [
                {
                    "title": "Bayesian Optimization for Hyperparameter Tuning",
                    "abstract": "We propose Bayesian optimization methods for efficient hyperparameter tuning. Our approach reduces the computational cost of model selection. We demonstrate effectiveness across multiple machine learning algorithms.",
                    "authors": ["Stats Expert", "ML Researcher"],
                    "year": 2023,
                    "category": "Statistical Machine Learning"
                }
            ]
        }
        
        papers = base_papers.get(category, [])
        result = []
        for i in range(count):
            if i < len(papers):
                result.append(papers[i])
            else:
                # Create variations of existing papers
                base_paper = papers[i % len(papers)]
                variation = base_paper.copy()
                variation["title"] = f"{base_paper['title']} (Variation {i // len(papers) + 1})"
                variation["year"] = base_paper["year"] + (i // len(papers))
                result.append(variation)
        
        return result

class MTEBRetrievalBenchmark:
    """MTEB-style retrieval benchmark for embedding models."""
    
    def __init__(self, cache_dir: str = None, analysis_config: Optional[Dict[str, Any]] = None):
        self.dataset_manager = AcademicPaperDataset(cache_dir)
        self.results = []
        self.analysis_config = analysis_config or {}
        self.analysis_records: List[Dict[str, Any]] = []
    
    @staticmethod
    def _normalize_key(value: str) -> str:
        """Normalize text keys for matching."""
        return ' '.join(value.lower().split())
    
    @staticmethod
    def _extract_arxiv_id(url_or_id: str) -> Optional[str]:
        """Extract normalized arXiv ID from URL or ID string.
        
        Examples:
            "http://arxiv.org/abs/2510.13804v1" -> "2510.13804v1"
            "https://arxiv.org/abs/2510.13804" -> "2510.13804"
            "2510.13804v1" -> "2510.13804v1"
        """
        if not url_or_id:
            return None
        text = url_or_id.strip()
        # Extract arXiv ID from URL patterns
        if 'arxiv.org' in text.lower():
            if '/abs/' in text:
                arxiv_id = text.split('/abs/')[-1]
            elif '/pdf/' in text:
                arxiv_id = text.split('/pdf/')[-1].replace('.pdf', '')
            else:
                return None
        else:
            # Assume it's already an arXiv ID
            arxiv_id = text
        
        # Remove query parameters and fragments
        arxiv_id = arxiv_id.split('?')[0].split('#')[0]
        return arxiv_id.strip()
    
    @staticmethod
    def _extract_arxiv_ids_from_queries(query_records: Optional[List[Dict]]) -> List[str]:
        """Extract all unique arXiv IDs from query records."""
        if not query_records:
            return []
        
        arxiv_ids = set()
        for record in query_records:
            for doc in record.get("relevant_documents", []):
                url = doc.get("url")
                if url:
                    arxiv_id = MTEBRetrievalBenchmark._extract_arxiv_id(url)
                    if arxiv_id:
                        arxiv_ids.add(arxiv_id)
        
        return sorted(list(arxiv_ids))
    
    def _build_external_queries(
        self,
        papers: List[Dict],
        query_records: List[Dict],
        limit: Optional[int] = None,
        expand_to_chunks: bool = False,
        paper_to_chunks: Optional[Dict[int, List[int]]] = None
    ) -> List[Dict]:
        """
        Build query structures from an external query specification.
        
        Args:
            papers: Dataset papers to search over.
            query_records: Raw query specifications loaded from file.
            limit: Optional maximum number of queries to return.
            expand_to_chunks: If True, map relevant documents to all of their chunk indices.
            paper_to_chunks: Mapping of paper index to list of chunk indices (required when expand_to_chunks).
        
        Returns:
            A list of query dictionaries compatible with downstream evaluation.
        """
        if not query_records:
            return []
        
        url_to_index = {}
        title_to_index = {}
        arxiv_id_to_index = {}
        
        for idx, paper in enumerate(papers):
            url = paper.get("url")
            if url:
                normalized_url = url.strip().lower()
                url_to_index[normalized_url] = idx
                # Also try to extract and index by arXiv ID
                arxiv_id = self._extract_arxiv_id(url)
                if arxiv_id:
                    arxiv_id_to_index[arxiv_id] = idx
            # Also check arxiv_id field if it exists
            arxiv_id_direct = paper.get("arxiv_id")
            if arxiv_id_direct:
                arxiv_id_to_index[arxiv_id_direct] = idx
            
            title = paper.get("title")
            if title:
                title_to_index[self._normalize_key(title)] = idx
        
        queries = []
        missing_entries = []
        
        for record in query_records:
            query_text = record.get("query")
            if not query_text:
                continue
            
            relevant_items: Dict[int, float] = {}
            for doc in record.get("relevant_documents", []):
                score = float(doc.get("relevance", 1.0))
                doc_index = None
                
                # Try matching by URL first
                url = doc.get("url")
                if url:
                    normalized_url = url.strip().lower()
                    doc_index = url_to_index.get(normalized_url)
                    
                    # If no direct match, try matching by arXiv ID
                    if doc_index is None:
                        arxiv_id = self._extract_arxiv_id(url)
                        if arxiv_id:
                            doc_index = arxiv_id_to_index.get(arxiv_id)
                
                # Try matching by title if URL didn't work
                if doc_index is None:
                    title = doc.get("title")
                    if title:
                        doc_index = title_to_index.get(self._normalize_key(title))
                
                if doc_index is None:
                    missing_entries.append((record.get("id"), doc))
                    continue
                
                if expand_to_chunks:
                    if paper_to_chunks is None:
                        raise ValueError("paper_to_chunks is required when expand_to_chunks is True.")
                    for chunk_idx in paper_to_chunks.get(doc_index, []):
                        relevant_items[chunk_idx] = score
                else:
                    relevant_items[doc_index] = score
            
            # Allow negative control queries with no relevant documents but warn once.
            if not relevant_items:
                missing_entries.append((record.get("id"), {"reason": "no matched documents"}))
            
            queries.append({
                "id": record.get("id"),
                "query": query_text,
                "category": record.get("category"),
                "difficulty": record.get("difficulty"),
                "relevant_items": relevant_items
            })
            
            if limit is not None and len(queries) >= limit:
                break
        
        if missing_entries:
            print("⚠️  Some query targets could not be matched:")
            for qid, doc in missing_entries[:5]:
                print(f"   - Query {qid}: missing {doc.get('title') or doc.get('url') or doc.get('reason')}")
            if len(missing_entries) > 5:
                print(f"   ... {len(missing_entries) - 5} more missing entries.")
        
        return queries
    
    def _collect_analysis(
        self,
        model_name: str,
        mode: str,
        queries: List[Dict],
        query_results: List[List[int]],
        relevant_sets: List[Dict[int, float]],
        documents_metadata: List[Dict[str, Any]]
    ):
        """Collect per-query analysis data when enabled."""
        top_k = max(0, int(self.analysis_config.get("top_k", 0)))
        if top_k <= 0 or not documents_metadata:
            return
        
        for query, rankings, relevants in zip(queries, query_results, relevant_sets):
            entry: Dict[str, Any] = {
                "model": model_name,
                "mode": mode,
                "query_id": query.get("id"),
                "query": query.get("query"),
                "relevant_items": [
                    {
                        **documents_metadata[doc_idx],
                        "document_index": doc_idx,
                        "relevance": float(score)
                    }
                    for doc_idx, score in relevants.items()
                    if doc_idx < len(documents_metadata)
                ],
                "top_documents": []
            }
            
            for rank, doc_idx in enumerate(rankings[:top_k], start=1):
                if doc_idx >= len(documents_metadata):
                    continue
                doc_meta = {
                    **documents_metadata[doc_idx],
                    "document_index": doc_idx,
                    "rank": rank,
                    "is_relevant": doc_idx in relevants,
                    "relevance_score": float(relevants.get(doc_idx, 0.0))
                }
                entry["top_documents"].append(doc_meta)
            
            self.analysis_records.append(entry)
    
    def save_analysis(self, path: Path):
        """Persist collected analysis data to disk."""
        if not self.analysis_records:
            print("ℹ️  No analysis data collected.")
            return
        
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "analysis": self.analysis_records,
            "top_k": int(self.analysis_config.get("top_k", 0))
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"📝 Analysis data written to {path}")
    
    def create_retrieval_queries(self, papers: List[Dict], max_queries: Optional[int] = None) -> List[Dict]:
        """Create comprehensive retrieval queries from paper content with graded relevance."""
        queries = []
        if max_queries is None or max_queries > len(papers):
            max_queries = len(papers)
        
        # Pre-compute attribute lookups across full candidate pool
        category_to_indices: Dict[str, List[int]] = defaultdict(list)
        year_to_indices: Dict[int, List[int]] = defaultdict(list)
        author_to_indices: Dict[str, List[int]] = defaultdict(list)
        
        for idx, paper in enumerate(papers):
            category = paper.get("category")
            if category:
                category_to_indices[category.lower()].append(idx)
            year = paper.get("year")
            if isinstance(year, int):
                year_to_indices[year].append(idx)
            for author in paper.get("authors", []):
                if author:
                    author_to_indices[author].append(idx)
        
        # Create queries based on paper content
        for i, paper in enumerate(papers[:max_queries]):
            # Title-based query
            queries.append({
                "query": paper["title"],
                "query_type": "title_match",
                "relevant_items": {i: 1.0}
            })
            
            # Abstract-based query (first sentence)
            abstract_sentences = paper["abstract"].split('.')
            if len(abstract_sentences) > 1:
                queries.append({
                    "query": abstract_sentences[0].strip(),
                    "query_type": "abstract_match",
                    "relevant_items": {i: 0.8}
                })
            
            # Category-based query
            category = paper.get("category")
            if category:
                category_key = category.lower()
                relevant_indices = category_to_indices.get(category_key, [])
                if relevant_indices:
                    queries.append({
                        "query": f"papers about {category_key}",
                        "query_type": "category_match",
                        "relevant_items": {doc_idx: 0.6 for doc_idx in relevant_indices}
                    })
            
            # Author-based query
            if paper.get("authors") and len(paper["authors"]) > 0:
                primary_author = paper["authors"][0]
                relevant_indices = author_to_indices.get(primary_author, [])
                if relevant_indices:
                    queries.append({
                        "query": f"papers by {primary_author}",
                        "query_type": "author_match",
                        "relevant_items": {doc_idx: 0.7 for doc_idx in relevant_indices}
                    })
            
            # Year-based query
            year = paper.get("year")
            if isinstance(year, int):
                relevant_indices = year_to_indices.get(year, [])
                if relevant_indices:
                    queries.append({
                        "query": f"papers from {year}",
                        "query_type": "year_match",
                        "relevant_items": {doc_idx: 0.5 for doc_idx in relevant_indices}
                    })
        
        return queries
    
    def create_full_text_retrieval_queries(self, papers: List[Dict]) -> List[Dict]:
        """Create retrieval queries from full text chunks."""
        queries = []
        chunk_index = 0
        
        for paper_idx, paper in enumerate(papers):
            if "full_text_chunks" not in paper:
                continue
            
            chunks = paper["full_text_chunks"]
            for chunk in chunks:
                current_chunk_id = chunk_index
                chunk_index += 1
                
                # Extract key phrases from chunk for queries
                text = chunk['text']
                sentences = text.split('. ')
                
                # First sentence query
                if sentences:
                    queries.append({
                        "query": sentences[0].strip(),
                        "query_type": "chunk_content",
                        "relevant_items": {current_chunk_id: 0.9},
                        "paper_index": paper_idx
                    })
                
                # Title + chunk query (more specific)
                queries.append({
                    "query": f"{paper['title']} {sentences[0].strip() if sentences else ''}",
                    "query_type": "title_chunk_match",
                    "relevant_items": {current_chunk_id: 0.8},
                    "paper_index": paper_idx
                })
        
        return queries
    
    def calculate_mteb_metrics(
        self,
        query_results: List[List[int]],
        relevant_sets: List[Dict[int, float]],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> Dict[str, float]:
        """Calculate MTEB-style retrieval metrics with graded relevance."""
        if not query_results or not relevant_sets:
            return {}
        
        metrics: Dict[str, float] = {}
        max_k = max(k_values)
        
        precision_scores: Dict[int, List[float]] = {k: [] for k in k_values}
        recall_scores: Dict[int, List[float]] = {k: [] for k in k_values}
        ndcg_scores: Dict[int, List[float]] = {k: [] for k in k_values}
        
        map_scores: List[float] = []
        mrr_scores: List[float] = []
        
        for results, relevants in zip(query_results, relevant_sets):
            relevant_items = {doc_id: score for doc_id, score in relevants.items() if score > 0}
            if not relevant_items:
                for k in k_values:
                    precision_scores[k].append(0.0)
                    recall_scores[k].append(0.0)
                    ndcg_scores[k].append(0.0)
                map_scores.append(0.0)
                mrr_scores.append(0.0)
                continue
            
            relevant_count = len(relevant_items)
            top_results = results[:max_k]
            ranked_gains = [relevant_items.get(doc_id, 0.0) for doc_id in top_results]
            ideal_gains_sorted = sorted(relevant_items.values(), reverse=True)
            
            for k in k_values:
                top_k_gains = ranked_gains[:k]
                hits = sum(1 for gain in top_k_gains if gain > 0)
                precision_scores[k].append(hits / k)
                recall_scores[k].append(hits / relevant_count if relevant_count else 0.0)
                
                dcg = sum(gain / np.log2(idx + 2) for idx, gain in enumerate(top_k_gains))
                ideal_k_gains = ideal_gains_sorted[:k]
                idcg = sum(gain / np.log2(idx + 2) for idx, gain in enumerate(ideal_k_gains))
                ndcg_scores[k].append((dcg / idcg) if idcg > 0 else 0.0)
            
            # Mean Average Precision
            hits_so_far = 0
            precision_sum = 0.0
            first_relevant_rank: Optional[int] = None
            for rank, doc_id in enumerate(results, start=1):
                if doc_id in relevant_items:
                    hits_so_far += 1
                    precision_sum += hits_so_far / rank
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                if hits_so_far == relevant_count:
                    # All relevant items have been seen; remaining ranks can't improve MAP/MRR
                    break
            
            average_precision = (precision_sum / relevant_count) if relevant_count else 0.0
            map_scores.append(average_precision)
            if first_relevant_rank is not None:
                mrr_scores.append(1.0 / first_relevant_rank)
            else:
                mrr_scores.append(0.0)
        
        for k in k_values:
            metrics[f'precision@{k}'] = float(np.mean(precision_scores[k])) if precision_scores[k] else 0.0
            metrics[f'recall@{k}'] = float(np.mean(recall_scores[k])) if recall_scores[k] else 0.0
            metrics[f'ndcg@{k}'] = float(np.mean(ndcg_scores[k])) if ndcg_scores[k] else 0.0
        
        metrics['map'] = float(np.mean(map_scores)) if map_scores else 0.0
        metrics['mrr'] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        
        return metrics
    
    def run_mteb_benchmark(self, 
                          model_name: str, 
                          embedding_function, 
                          dataset_size: int = 100,
                          test_queries: int = 50,
                          external_queries: Optional[List[Dict]] = None) -> MTEBResult:
        """Run comprehensive MTEB-style benchmark."""
        print(f"\n🧪 Running MTEB-style benchmark for {model_name}")
        print("=" * 60)
        
        # 1. Load academic papers dataset
        print("📚 Loading academic papers dataset...")
        # Extract arXiv IDs from external queries if provided
        required_arxiv_ids = None
        if external_queries:
            required_arxiv_ids = self._extract_arxiv_ids_from_queries(external_queries)
            if required_arxiv_ids:
                print(f"   Found {len(required_arxiv_ids)} unique papers referenced in queries")
        
        papers = self.dataset_manager.get_arxiv_papers(
            max_papers=dataset_size,
            required_arxiv_ids=required_arxiv_ids
        )
        print(f"   Loaded {len(papers)} papers")
        
        # 2. Create retrieval queries
        print("🔍 Creating retrieval queries...")
        if external_queries:
            queries = self._build_external_queries(papers, external_queries, limit=test_queries)
        else:
            queries = self.create_retrieval_queries(papers, max_queries=test_queries)
            queries = [q for q in queries if q.get("relevant_items")]
        print(f"   Created {len(queries)} queries")
        
        if not queries:
            print("⚠️  No queries available after filtering; skipping benchmark.")
            empty_result = MTEBResult(
                model_name=model_name,
                dataset_name="academic_papers",
                task_type="retrieval",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mrr=0.0,
                ndcg=0.0,
                map_score=0.0,
                embedding_time=0.0,
                search_time=0.0,
                total_documents=len(papers),
                total_queries=0,
                success_rate=0.0
            )
            return empty_result
        
        # 3. Generate embeddings for documents
        print("⚡ Generating document embeddings...")
        doc_texts = [f"{paper['title']} {paper['abstract']}" for paper in papers]
        doc_metadata = []
        for idx, paper in enumerate(papers):
            doc_metadata.append({
                "paper_index": idx,
                "title": paper.get("title"),
                "url": paper.get("url"),
                "category": paper.get("category"),
                "year": paper.get("year"),
                "abstract_preview": paper.get("abstract", "")[:200]
            })
        
        start_time = time.time()
        doc_embeddings = embedding_function(doc_texts)
        embedding_time = time.time() - start_time
        
        print(f"   Generated {len(doc_embeddings)} embeddings in {embedding_time:.3f}s")
        print(f"   Embedding dimension: {len(doc_embeddings[0])}")
        
        # 4. Generate embeddings for queries
        print("🔍 Generating query embeddings...")
        query_texts = [query["query"] for query in queries]
        
        start_time = time.time()
        query_embeddings = embedding_function(query_texts)
        query_embedding_time = time.time() - start_time
        
        print(f"   Generated {len(query_embeddings)} query embeddings in {query_embedding_time:.3f}s")
        
        # 5. Perform retrieval
        print("🔎 Performing retrieval...")
        start_time = time.time()
        
        query_results = []
        relevant_sets = [query["relevant_items"] for query in queries]
        
        for i, query_embedding in enumerate(query_embeddings):
            # Calculate similarities
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Get ranked results
            ranked_indices = np.argsort(similarities)[::-1]  # Descending order
            query_results.append(ranked_indices.tolist())
        
        search_time = time.time() - start_time
        print(f"   Completed retrieval in {search_time:.3f}s")
        
        # 6. Calculate MTEB metrics
        print("📊 Calculating MTEB metrics...")
        metrics = self.calculate_mteb_metrics(query_results, relevant_sets)
        
        # Optional detailed analysis
        if queries and self.analysis_config.get("top_k", 0):
            self._collect_analysis(
                model_name=model_name,
                mode="abstract",
                queries=queries,
                query_results=query_results,
                relevant_sets=relevant_sets,
                documents_metadata=doc_metadata
            )
        
        # Per-query stats
        per_query_stats = []
        for query, rankings, relevants in zip(queries, query_results, relevant_sets):
            best_rank = None
            for rank, doc_idx in enumerate(rankings, start=1):
                if doc_idx in relevants:
                    best_rank = rank
                    break
            stat = {
                "query_id": query.get("id"),
                "query_text": query.get("query"),
                "category": query.get("category"),
                "difficulty": query.get("difficulty"),
                "mode": "abstract",
                "has_relevance": bool(relevants),
                "best_rank": best_rank,
                "top1_hit": best_rank is not None and best_rank <= 1,
                "top5_hit": best_rank is not None and best_rank <= 5,
                "top10_hit": best_rank is not None and best_rank <= 10,
                "relevant_count": len(relevants),
            }
            per_query_stats.append(stat)
        
        # 7. Create result
        result = MTEBResult(
            model_name=model_name,
            dataset_name="academic_papers",
            task_type="retrieval",
            accuracy=metrics.get('precision@1', 0.0),
            precision=metrics.get('precision@5', 0.0),
            recall=metrics.get('recall@5', 0.0),
            f1_score=2 * (metrics.get('precision@5', 0.0) * metrics.get('recall@5', 0.0)) / 
                     (metrics.get('precision@5', 0.0) + metrics.get('recall@5', 0.0)) if 
                     (metrics.get('precision@5', 0.0) + metrics.get('recall@5', 0.0)) > 0 else 0.0,
            mrr=metrics.get('mrr', 0.0),
            ndcg=metrics.get('ndcg@10', 0.0),
            map_score=metrics.get('map', 0.0),
            embedding_time=embedding_time + query_embedding_time,
            search_time=search_time,
            total_documents=len(papers),
            total_queries=len(queries),
            success_rate=metrics.get('precision@5', 0.0),
            per_query_stats=per_query_stats
        )
        
        # 8. Display results
        print(f"\n📈 MTEB Benchmark Results for {model_name}:")
        print(f"   Accuracy (P@1): {result.accuracy:.3f}")
        print(f"   Precision@5: {result.precision:.3f}")
        print(f"   Recall@5: {result.recall:.3f}")
        print(f"   F1-Score: {result.f1_score:.3f}")
        print(f"   MRR: {result.mrr:.3f}")
        print(f"   NDCG@10: {result.ndcg:.3f}")
        print(f"   MAP: {result.map_score:.3f}")
        print(f"   Embedding time: {result.embedding_time:.3f}s")
        print(f"   Search time: {result.search_time:.3f}s")
        
        # Display detailed metrics
        print(f"\n📊 Detailed MTEB Metrics:")
        for k in [1, 3, 5, 10, 20]:
            if f'precision@{k}' in metrics:
                print(f"   P@{k}: {metrics[f'precision@{k}']:.3f}")
        for k in [1, 3, 5, 10, 20]:
            if f'ndcg@{k}' in metrics:
                print(f"   NDCG@{k}: {metrics[f'ndcg@{k}']:.3f}")
        
        return result

    def run_full_text_benchmark(self, 
                               model_name: str, 
                               embedding_function, 
                               dataset_size: int = 100,
                               test_queries: int = 50,
                               external_queries: Optional[List[Dict]] = None) -> MTEBResult:
        """Run MTEB-style benchmark using full text of papers."""
        print(f"\n🧪 Running MTEB-style benchmark (FULL TEXT mode) for {model_name}")
        print("=" * 60)
        
        # 1. Load academic papers dataset with full text
        print("📚 Loading academic papers dataset with full text...")
        # Extract arXiv IDs from external queries if provided
        required_arxiv_ids = None
        if external_queries:
            required_arxiv_ids = self._extract_arxiv_ids_from_queries(external_queries)
            if required_arxiv_ids:
                print(f"   Found {len(required_arxiv_ids)} unique papers referenced in queries")
        
        papers = self.dataset_manager.get_arxiv_papers(
            max_papers=dataset_size,
            use_full_text=True,
            required_arxiv_ids=required_arxiv_ids
        )
        
        # Filter papers that have full text chunks
        papers_with_fulltext = [p for p in papers if "full_text_chunks" in p and p["full_text_chunks"]]
        print(f"   Loaded {len(papers)} papers, {len(papers_with_fulltext)} with full text")
        
        if not papers_with_fulltext:
            print("⚠️  No papers with full text available. Using fallback to title+abstract mode.")
            return self.run_mteb_benchmark(model_name, embedding_function, dataset_size, test_queries)
        
        # 2. Create retrieval queries from full text chunks
        print("🔍 Creating retrieval queries from full text chunks...")
        if external_queries:
            # We'll build queries after chunk mapping is available
            raw_query_records = external_queries
            queries = None
        else:
            queries = self.create_full_text_retrieval_queries(papers_with_fulltext[:test_queries])
            queries = [q for q in queries if q.get("relevant_items")]
            covered_chunks = {idx for query in queries for idx in query["relevant_items"].keys()}
            print(f"   Created {len(queries)} queries covering {len(covered_chunks)} chunks")
        
        # 3. Generate embeddings for all chunks
        print("⚡ Generating embeddings for document chunks...")
        chunk_texts = []
        chunk_metadata: List[Dict[str, Any]] = []
        paper_to_chunks: Dict[int, List[int]] = defaultdict(list)
        for paper_idx, paper in enumerate(papers_with_fulltext):
            if "full_text_chunks" in paper:
                for local_chunk_idx, chunk in enumerate(paper["full_text_chunks"]):
                    chunk_id = len(chunk_texts)
                    chunk_texts.append(chunk['text'])
                    paper_to_chunks[paper_idx].append(chunk_id)
                    chunk_metadata.append({
                        "paper_index": paper_idx,
                        "paper_title": paper.get("title"),
                        "paper_url": paper.get("url"),
                        "chunk_index": local_chunk_idx,
                        "source": chunk.get("source"),
                        "char_count": chunk.get("char_count"),
                        "chunk_preview": chunk['text'][:200]
                    })
        
        start_time = time.time()
        doc_embeddings = embedding_function(chunk_texts)
        embedding_time = time.time() - start_time
        
        print(f"   Generated {len(doc_embeddings)} chunk embeddings in {embedding_time:.3f}s")
        print(f"   Embedding dimension: {len(doc_embeddings[0])}")
        print(f"   Total text processed: {sum(len(ct) for ct in chunk_texts)} characters")
        
        # 4. Generate embeddings for queries
        print("🔍 Generating query embeddings...")
        if external_queries:
            queries = self._build_external_queries(
                papers_with_fulltext,
                raw_query_records,
                limit=test_queries,
                expand_to_chunks=True,
                paper_to_chunks=paper_to_chunks
            )
            covered_chunks = {idx for query in queries for idx in query["relevant_items"].keys()}
            print(f"   Loaded {len(queries)} external queries covering {len(covered_chunks)} chunks")
        
        if not queries:
            print("⚠️  No queries available after filtering; skipping benchmark.")
            empty_result = MTEBResult(
                model_name=model_name,
                dataset_name="academic_papers_fulltext",
                task_type="retrieval_fulltext",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mrr=0.0,
                ndcg=0.0,
                map_score=0.0,
                embedding_time=0.0,
                search_time=0.0,
                total_documents=len(chunk_texts),
                total_queries=0,
                success_rate=0.0,
                use_full_text=True
            )
            return empty_result
        
        query_texts = [query["query"] for query in queries]
        
        start_time = time.time()
        query_embeddings = embedding_function(query_texts)
        query_embedding_time = time.time() - start_time
        
        print(f"   Generated {len(query_embeddings)} query embeddings in {query_embedding_time:.3f}s")
        
        # 5. Perform retrieval
        print("🔎 Performing retrieval on chunks...")
        start_time = time.time()
        
        query_results = []
        relevant_sets = [query["relevant_items"] for query in queries]
        
        for i, query_embedding in enumerate(query_embeddings):
            # Calculate similarities with all chunks
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Get ranked results (chunk indices)
            ranked_indices = np.argsort(similarities)[::-1]  # Descending order
            query_results.append(ranked_indices.tolist())
        
        search_time = time.time() - start_time
        print(f"   Completed retrieval in {search_time:.3f}s")
        
        # 6. Calculate MTEB metrics
        print("📊 Calculating MTEB metrics...")
        metrics = self.calculate_mteb_metrics(query_results, relevant_sets)
        
        # Optional detailed analysis
        if queries and self.analysis_config.get("top_k", 0):
            self._collect_analysis(
                model_name=model_name,
                mode="full_text",
                queries=queries,
                query_results=query_results,
                relevant_sets=relevant_sets,
                documents_metadata=chunk_metadata
            )
        
        per_query_stats = []
        for query, rankings, relevants in zip(queries, query_results, relevant_sets):
            best_rank = None
            for rank, doc_idx in enumerate(rankings, start=1):
                if doc_idx in relevants:
                    best_rank = rank
                    break
            stat = {
                "query_id": query.get("id"),
                "query_text": query.get("query"),
                "category": query.get("category"),
                "difficulty": query.get("difficulty"),
                "mode": "full_text",
                "has_relevance": bool(relevants),
                "best_rank": best_rank,
                "top1_hit": best_rank is not None and best_rank <= 1,
                "top5_hit": best_rank is not None and best_rank <= 5,
                "top10_hit": best_rank is not None and best_rank <= 10,
                "relevant_count": len(relevants),
            }
            per_query_stats.append(stat)
        
        # 7. Create result
        result = MTEBResult(
            model_name=model_name,
            dataset_name="academic_papers_fulltext",
            task_type="retrieval_fulltext",
            accuracy=metrics.get('precision@1', 0.0),
            precision=metrics.get('precision@5', 0.0),
            recall=metrics.get('recall@5', 0.0),
            f1_score=2 * (metrics.get('precision@5', 0.0) * metrics.get('recall@5', 0.0)) / 
                     (metrics.get('precision@5', 0.0) + metrics.get('recall@5', 0.0)) if 
                     (metrics.get('precision@5', 0.0) + metrics.get('recall@5', 0.0)) > 0 else 0.0,
            mrr=metrics.get('mrr', 0.0),
            ndcg=metrics.get('ndcg@10', 0.0),
            map_score=metrics.get('map', 0.0),
            embedding_time=embedding_time + query_embedding_time,
            search_time=search_time,
            total_documents=len(chunk_texts),
            total_queries=len(queries),
            success_rate=metrics.get('precision@5', 0.0),
            use_full_text=True,
            per_query_stats=per_query_stats
        )
        
        # 8. Display results
        print(f"\n📈 MTEB Benchmark Results (FULL TEXT) for {model_name}:")
        print(f"   Accuracy (P@1): {result.accuracy:.3f}")
        print(f"   Precision@5: {result.precision:.3f}")
        print(f"   Recall@5: {result.recall:.3f}")
        print(f"   F1-Score: {result.f1_score:.3f}")
        print(f"   MRR: {result.mrr:.3f}")
        print(f"   NDCG@10: {result.ndcg:.3f}")
        print(f"   MAP: {result.map_score:.3f}")
        print(f"   Embedding time: {result.embedding_time:.3f}s")
        print(f"   Search time: {result.search_time:.3f}s")
        print(f"   Document chunks: {result.total_documents}")
        
        # Display detailed metrics
        print(f"\n📊 Detailed MTEB Metrics:")
        for k in [1, 3, 5, 10, 20]:
            if f'precision@{k}' in metrics:
                print(f"   P@{k}: {metrics[f'precision@{k}']:.3f}")
        for k in [1, 3, 5, 10, 20]:
            if f'ndcg@{k}' in metrics:
                print(f"   NDCG@{k}: {metrics[f'ndcg@{k}']:.3f}")
        
        return result


def run_mteb_integration_benchmark(
    model_names: List[str],
    modes: List[str],
    dataset_size: int = 100,
    test_queries: int = 50,
    cache_dir: Optional[str] = None,
    query_records: Optional[List[Dict]] = None,
    analysis_top_k: int = 0,
    output_dir: Optional[Path] = None,
) -> List[MTEBResult]:
    """Run comprehensive MTEB-style integration benchmark for one or more models."""
    base_dir = Path(__file__).parent
    output_dir = Path(output_dir) if output_dir else base_dir / "benchmarks" / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 MTEB-Style Integration Benchmark for Local Embedding Models")
    print("=" * 80)
    print(f"Results directory: {output_dir}")
    if query_records:
        print(f"🔖 Loaded external query set with {len(query_records)} entries")
    print(f"Evaluating modes: {', '.join(modes)}")
    print(f"Benchmarking models: {', '.join(model_names)}")
    
    results: List[MTEBResult] = []
    summary_rows: List[Dict[str, Any]] = []
    
    for mode in modes:
        use_full_text = mode == "full_text"
        mode_label = "FULL TEXT" if use_full_text else "TITLE + ABSTRACT"
        print(f"\n{'=' * 80}")
        print(f"🧪 Mode: {mode_label}")
        print(f"{'=' * 80}")
        
        for model_identifier in model_names:
            is_default_model = model_identifier == "chromadb-default"
            display_name = "Default ChromaDB (all-MiniLM-L6-v2)" if is_default_model else model_identifier
            print(f"\n--- Running {display_name} ---")
            
            analysis_path = None
            analysis_config = None
            if analysis_top_k > 0:
                analysis_path = output_dir / "analysis" / mode / f"{slugify_model_name(display_name)}.json"
                analysis_config = {
                    "top_k": analysis_top_k,
                    "output_path": analysis_path
                }
            
            benchmark = MTEBRetrievalBenchmark(cache_dir=cache_dir, analysis_config=analysis_config)
            
            try:
                if is_default_model:
                    embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
                else:
                    embedding_function = LocalEmbeddingFunction(model_name=model_identifier)
                
                if use_full_text:
                    result = benchmark.run_full_text_benchmark(
                        model_name=display_name,
                        embedding_function=embedding_function,
                        dataset_size=dataset_size,
                        test_queries=test_queries,
                        external_queries=query_records
                    )
                else:
                    result = benchmark.run_mteb_benchmark(
                        model_name=display_name,
                        embedding_function=embedding_function,
                        dataset_size=dataset_size,
                        test_queries=test_queries,
                        external_queries=query_records
                    )
                result.use_full_text = use_full_text
                results.append(result)
                
                summary_rows.append({
                    "mode": mode,
                    "model": display_name,
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                    "mrr": result.mrr,
                    "ndcg": result.ndcg,
                    "map_score": result.map_score,
                    "embedding_time": result.embedding_time,
                    "search_time": result.search_time,
                    "total_documents": result.total_documents,
                    "total_queries": result.total_queries,
                    "success_rate": result.success_rate
                })
                
                if analysis_path:
                    benchmark.save_analysis(analysis_path)
            except Exception as exc:
                print(f"❌ Error testing {display_name} ({mode}): {exc}")
                import traceback
                traceback.print_exc()
                continue
    
    if not summary_rows:
        print("⚠️  No successful benchmark runs were recorded.")
        return results
    
    summary_rows = sorted(summary_rows, key=lambda r: (r["mode"], r["model"]))
    
    # Persist summary outputs
    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n📝 Summary written to {summary_json_path}")
    
    summary_csv_path = output_dir / "summary.csv"
    csv_fields = [
        "mode", "model", "accuracy", "precision", "recall", "f1_score",
        "mrr", "ndcg", "map_score", "embedding_time", "search_time",
        "total_documents", "total_queries", "success_rate"
    ]
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"📝 Summary CSV written to {summary_csv_path}")
    
    # Pretty scoreboard
    print("\n🏆 Benchmark Scoreboard")
    for mode in modes:
        mode_rows = [row for row in summary_rows if row["mode"] == mode]
        if not mode_rows:
            continue
        mode_label = "Full Text" if mode == "full_text" else "Title + Abstract"
        print(f"\n{mode_label} Results")
        print("-" * 60)
        best_values: Dict[str, float] = {
            metric: max((row[metric] for row in mode_rows), default=0.0)
            for metric in SCOREBOARD_METRICS
        }
        
        header = f"{'Model':30} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6} {'MAP':>6}"
        print(header)
        print("-" * len(header))
        for row in mode_rows:
            def fmt(metric: str) -> str:
                value = row[metric]
                formatted = f"{value:.3f}"
                if abs(value - best_values[metric]) < 1e-6:
                    formatted = f"{formatted}*"
                return formatted.rjust(6 + (1 if "*" in formatted else 0))
            
            line = (
                f"{row['model'][:30]:30} "
                f"{fmt('accuracy')} {fmt('precision')} {fmt('recall')} "
                f"{fmt('mrr')} {fmt('ndcg')} {fmt('map_score')}"
            )
            print(line)
        print("-" * len(header))
    
    # Query breakdown
    query_breakdown = compute_query_breakdown(results)
    breakdown_path = output_dir / "query_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump(query_breakdown, f, indent=2)
    print(f"\n📝 Query breakdown written to {breakdown_path}")
    
    for mode, models_dict in query_breakdown.items():
        mode_label = "Full Text" if mode == "full_text" else "Title + Abstract"
        print(f"\n🔍 {mode_label} category hit@5 (top 3 per model)")
        for model_name, stats in models_dict.items():
            categories = stats.get("by_category", {})
            if not categories:
                continue
            top_categories = sorted(
                categories.items(),
                key=lambda item: item[1].get("hit_at_5", 0.0),
                reverse=True
            )[:3]
            print(f"  {model_name}:")
            for category, data in top_categories:
                hit5 = data.get("hit_at_5", 0.0)
                queries = data.get("queries", 0)
                print(f"    {category}: hit@5={hit5:.2f} ({queries} queries)")
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="MTEB-style benchmark for local embedding models")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of additional embedding models to benchmark (HuggingFace model ids)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Shortcut for supplying a single additional model (equivalent to --models MODEL)"
    )
    parser.add_argument(
        "--dataset-size", 
        type=int, 
        default=100,
        help="Number of papers in dataset (default: 100)"
    )
    parser.add_argument(
        "--test-queries", 
        type=int, 
        default=50,
        help="Number of test queries (default: 50)"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        help="Cache directory for downloaded papers"
    )
    parser.add_argument(
        "--full-text", 
        action="store_true", 
        help="(Deprecated) Run only the full-text benchmark mode"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=str(DEFAULT_QUERY_FILE),
        help=f"Path to a JSON file containing external queries with relevance annotations (default: {DEFAULT_QUERY_FILE})"
    )
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=5,
        help="Collect per-query diagnostics for the top-K retrieved documents (set to 0 to disable)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to write benchmark outputs (defaults to benchmarks/results/<timestamp>)"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["abstract", "full_text"],
        help="Benchmark modes to run (default: both abstract and full_text)"
    )
    parser.add_argument(
        "--no-default-model",
        action="store_true",
        help="Skip benchmarking the ChromaDB default embedding function"
    )
    
    args = parser.parse_args()
    
    query_path = Path(args.query_file).expanduser()
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    with open(query_path) as f:
        query_payload = json.load(f)
    if isinstance(query_payload, dict) and "queries" in query_payload:
        query_records = query_payload["queries"]
    elif isinstance(query_payload, list):
        query_records = query_payload
    else:
        raise ValueError("Query file must contain either a list of queries or an object with a 'queries' field.")
    
    # Determine models to benchmark
    models: List[str] = []
    if args.models:
        models.extend(args.models)
    if args.model:
        models.append(args.model)
    if not args.no_default_model:
        models = ["chromadb-default"] + [m for m in models if m != "chromadb-default"]
    else:
        models = [m for m in models if m != "chromadb-default"]
    if not models:
        models = DEFAULT_MODELS.copy()
        if args.no_default_model:
            models = [m for m in models if m != "chromadb-default"]
    if not models:
        raise ValueError("No embedding models specified for benchmarking.")
    # Deduplicate while preserving order
    unique_models: List[str] = []
    for name in models:
        if name not in unique_models:
            unique_models.append(name)
    models = unique_models
    
    # Determine benchmark modes
    if args.modes:
        modes = args.modes
    elif args.full_text:
        modes = ["full_text"]
    else:
        modes = DEFAULT_MODES.copy()
    modes = [mode for mode in modes if mode in {"abstract", "full_text"}]
    if not modes:
        raise ValueError("No valid benchmark modes specified.")
    
    analysis_top_k = max(0, args.analysis_top_k)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    
    run_mteb_integration_benchmark(
        model_names=models,
        modes=modes,
        dataset_size=args.dataset_size,
        test_queries=args.test_queries,
        cache_dir=args.cache_dir,
        query_records=query_records,
        analysis_top_k=analysis_top_k,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
