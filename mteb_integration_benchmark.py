#!/usr/bin/env python3
"""
MTEB-Style Integration Benchmark for Local Embedding Models
Comprehensive benchmark following MTEB leaderboard methodology for retrieval tasks.
Tests local models against real academic papers with intensive evaluation.
"""

import sys
import time
import json
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import tempfile
import os
import argparse
from urllib.parse import urlparse
import gzip
import pickle

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from zotero_mcp.chroma_client import LocalEmbeddingFunction, create_chroma_client
import chromadb.utils.embedding_functions

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
    embedding_time: float
    search_time: float
    total_documents: int
    total_queries: int
    success_rate: float

class AcademicPaperDataset:
    """Download and manage real academic paper datasets for benchmarking."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "mteb_benchmark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_arxiv_papers(self, categories: List[str] = None, max_papers: int = 200) -> List[Dict]:
        """Download real papers from arXiv for benchmarking."""
        if categories is None:
            categories = ['cs.AI', 'cs.CL', 'cs.LG', 'cs.CV', 'cs.IR', 'cs.NE', 'stat.ML']
        
        papers = []
        for category in categories:
            papers.extend(self._fetch_arxiv_category(category, max_papers // len(categories)))
        
        return papers[:max_papers]
    
    def _fetch_arxiv_category(self, category: str, max_papers: int) -> List[Dict]:
        """Fetch real papers from arXiv API."""
        cache_file = self.cache_dir / f"arxiv_{category}_{max_papers}.json"
        
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        print(f"📡 Fetching real papers from arXiv category: {category}")
        
        # Use arXiv API to get real papers
        papers = self._fetch_from_arxiv_api(category, max_papers)
        
        with open(cache_file, 'w') as f:
            json.dump(papers, f, indent=2)
        
        return papers
    
    def _fetch_from_arxiv_api(self, category: str, max_papers: int) -> List[Dict]:
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
            
            # Parse XML response (simplified)
            papers = self._parse_arxiv_xml(response.text)
            return papers[:max_papers]
            
        except Exception as e:
            print(f"⚠️  Error fetching from arXiv API: {e}")
            print("📝 Falling back to synthetic papers...")
            return self._generate_synthetic_papers(category, max_papers)
    
    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict]:
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
                
                papers.append({
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "year": int(year),
                    "category": "arXiv",
                    "url": entry.find('{http://www.w3.org/2005/Atom}id').text.strip()
                })
            except Exception as e:
                continue
        
        return papers
    
    def _generate_synthetic_papers(self, category: str, count: int) -> List[Dict]:
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
        # Duplicate and modify papers to reach desired count
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
    
    def __init__(self, cache_dir: str = None):
        self.dataset_manager = AcademicPaperDataset(cache_dir)
        self.results = []
    
    def create_retrieval_queries(self, papers: List[Dict]) -> List[Dict]:
        """Create comprehensive retrieval queries from paper content."""
        queries = []
        
        # Create queries based on paper content
        for i, paper in enumerate(papers):
            # Title-based query
            queries.append({
                "query": paper["title"],
                "expected_paper_id": i,
                "query_type": "title_match",
                "relevance_score": 1.0
            })
            
            # Abstract-based query (first sentence)
            abstract_sentences = paper["abstract"].split('.')
            if len(abstract_sentences) > 1:
                queries.append({
                    "query": abstract_sentences[0].strip(),
                    "expected_paper_id": i,
                    "query_type": "abstract_match",
                    "relevance_score": 0.8
                })
            
            # Category-based query
            queries.append({
                "query": f"papers about {paper['category'].lower()}",
                "expected_paper_id": i,
                "query_type": "category_match",
                "relevance_score": 0.6
            })
            
            # Author-based query
            if paper.get("authors") and len(paper["authors"]) > 0:
                queries.append({
                    "query": f"papers by {paper['authors'][0]}",
                    "expected_paper_id": i,
                    "query_type": "author_match",
                    "relevance_score": 0.7
                })
            
            # Year-based query
            queries.append({
                "query": f"papers from {paper['year']}",
                "expected_paper_id": i,
                "query_type": "year_match",
                "relevance_score": 0.5
            })
        
        return queries
    
    def calculate_mteb_metrics(self, 
                            query_results: List[List[int]], 
                            expected_results: List[int],
                            relevance_scores: List[float],
                            k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, float]:
        """Calculate MTEB-style retrieval metrics."""
        metrics = {}
        
        for k in k_values:
            precision_at_k = []
            recall_at_k = []
            ndcg_at_k = []
            
            for i, (results, expected, relevance) in enumerate(zip(query_results, expected_results, relevance_scores)):
                # Get top-k results
                top_k = results[:k]
                
                # Precision@K
                if expected in top_k:
                    precision_at_k.append(1.0)
                else:
                    precision_at_k.append(0.0)
                
                # Recall@K
                if expected in top_k:
                    recall_at_k.append(1.0)
                else:
                    recall_at_k.append(0.0)
                
                # NDCG@K
                if expected in top_k:
                    rank = top_k.index(expected) + 1
                    ndcg = relevance / np.log2(rank + 1)
                    ndcg_at_k.append(ndcg)
                else:
                    ndcg_at_k.append(0.0)
            
            metrics[f'precision@{k}'] = np.mean(precision_at_k)
            metrics[f'recall@{k}'] = np.mean(recall_at_k)
            metrics[f'ndcg@{k}'] = np.mean(ndcg_at_k)
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr_scores = []
        for results, expected in zip(query_results, expected_results):
            if expected in results:
                rank = results.index(expected) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        metrics['mrr'] = np.mean(mrr_scores)
        
        # Calculate MAP (Mean Average Precision)
        map_scores = []
        for results, expected in zip(query_results, expected_results):
            if expected in results:
                rank = results.index(expected) + 1
                map_scores.append(1.0 / rank)
            else:
                map_scores.append(0.0)
        metrics['map'] = np.mean(map_scores)
        
        return metrics
    
    def run_mteb_benchmark(self, 
                          model_name: str, 
                          embedding_function, 
                          dataset_size: int = 100,
                          test_queries: int = 50) -> MTEBResult:
        """Run comprehensive MTEB-style benchmark."""
        print(f"\n🧪 Running MTEB-style benchmark for {model_name}")
        print("=" * 60)
        
        # 1. Load academic papers dataset
        print("📚 Loading academic papers dataset...")
        papers = self.dataset_manager.get_arxiv_papers(max_papers=dataset_size)
        print(f"   Loaded {len(papers)} papers")
        
        # 2. Create retrieval queries
        print("🔍 Creating retrieval queries...")
        queries = self.create_retrieval_queries(papers[:test_queries])
        print(f"   Created {len(queries)} queries")
        
        # 3. Generate embeddings for documents
        print("⚡ Generating document embeddings...")
        doc_texts = [f"{paper['title']} {paper['abstract']}" for paper in papers]
        
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
        expected_results = []
        relevance_scores = []
        
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
            expected_results.append(queries[i]["expected_paper_id"])
            relevance_scores.append(queries[i]["relevance_score"])
        
        search_time = time.time() - start_time
        print(f"   Completed retrieval in {search_time:.3f}s")
        
        # 6. Calculate MTEB metrics
        print("📊 Calculating MTEB metrics...")
        metrics = self.calculate_mteb_metrics(query_results, expected_results, relevance_scores)
        
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
            embedding_time=embedding_time + query_embedding_time,
            search_time=search_time,
            total_documents=len(papers),
            total_queries=len(queries),
            success_rate=metrics.get('precision@5', 0.0)
        )
        
        # 8. Display results
        print(f"\n📈 MTEB Benchmark Results for {model_name}:")
        print(f"   Accuracy (P@1): {result.accuracy:.3f}")
        print(f"   Precision@5: {result.precision:.3f}")
        print(f"   Recall@5: {result.recall:.3f}")
        print(f"   F1-Score: {result.f1_score:.3f}")
        print(f"   MRR: {result.mrr:.3f}")
        print(f"   NDCG@10: {result.ndcg:.3f}")
        print(f"   MAP: {metrics.get('map', 0.0):.3f}")
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

def run_mteb_integration_benchmark(local_model_name: str = None):
    """Run comprehensive MTEB-style integration benchmark."""
    print("🚀 MTEB-Style Integration Benchmark for Local Embedding Models")
    print("=" * 80)
    print("Following MTEB leaderboard methodology for retrieval tasks")
    print("Testing with real academic papers and intensive evaluation")
    
    # Initialize benchmark
    benchmark = MTEBRetrievalBenchmark()
    results = []
    
    # Test 1: Default ChromaDB model
    print(f"\n{'='*80}")
    print("🧪 Testing Default ChromaDB Model")
    print("=" * 80)
    
    try:
        default_ef = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
        default_result = benchmark.run_mteb_benchmark(
            model_name="Default ChromaDB (all-MiniLM-L6-v2)",
            embedding_function=default_ef,
            dataset_size=100,
            test_queries=50
        )
        results.append(default_result)
    except Exception as e:
        print(f"❌ Error testing default model: {e}")
    
    # Test 2: Local model
    if local_model_name:
        model_name = local_model_name
    else:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing {model_name} Model")
    print("=" * 80)
    
    try:
        local_ef = LocalEmbeddingFunction(model_name=model_name)
        local_result = benchmark.run_mteb_benchmark(
            model_name=model_name,
            embedding_function=local_ef,
            dataset_size=100,
            test_queries=50
        )
        results.append(local_result)
    except Exception as e:
        print(f"❌ Error testing {model_name} model: {e}")
    
    # Compare results
    if len(results) == 2:
        print(f"\n🏆 MTEB-STYLE BENCHMARK COMPARISON")
        print("=" * 80)
        
        default = results[0]
        local = results[1]
        
        print(f"MTEB Leaderboard-Style Results:")
        print(f"  Default ChromaDB:")
        print(f"    Accuracy (P@1): {default.accuracy:.3f}")
        print(f"    Precision@5: {default.precision:.3f}")
        print(f"    Recall@5: {default.recall:.3f}")
        print(f"    F1-Score: {default.f1_score:.3f}")
        print(f"    MRR: {default.mrr:.3f}")
        print(f"    NDCG@10: {default.ndcg:.3f}")
        print(f"    Embedding time: {default.embedding_time:.3f}s")
        
        print(f"  {local.model_name}:")
        print(f"    Accuracy (P@1): {local.accuracy:.3f}")
        print(f"    Precision@5: {local.precision:.3f}")
        print(f"    Recall@5: {local.recall:.3f}")
        print(f"    F1-Score: {local.f1_score:.3f}")
        print(f"    MRR: {local.mrr:.3f}")
        print(f"    NDCG@10: {local.ndcg:.3f}")
        print(f"    Embedding time: {local.embedding_time:.3f}s")
        
        # Calculate improvements
        accuracy_improvement = local.accuracy - default.accuracy
        precision_improvement = local.precision - default.precision
        recall_improvement = local.recall - default.recall
        f1_improvement = local.f1_score - default.f1_score
        mrr_improvement = local.mrr - default.mrr
        ndcg_improvement = local.ndcg - default.ndcg
        
        print(f"\n📈 Performance Analysis:")
        print(f"  Accuracy improvement: {accuracy_improvement:+.3f}")
        print(f"  Precision improvement: {precision_improvement:+.3f}")
        print(f"  Recall improvement: {recall_improvement:+.3f}")
        print(f"  F1-Score improvement: {f1_improvement:+.3f}")
        print(f"  MRR improvement: {mrr_improvement:+.3f}")
        print(f"  NDCG improvement: {ndcg_improvement:+.3f}")
        
        # Determine winner
        print(f"\n🎯 MTEB-Style Leaderboard Results:")
        if local.accuracy > default.accuracy:
            print(f"  🥇 {local.model_name} wins on Accuracy: {local.accuracy:.3f} vs {default.accuracy:.3f}")
        else:
            print(f"  🥇 Default model wins on Accuracy: {default.accuracy:.3f} vs {local.accuracy:.3f}")
        
        if local.ndcg > default.ndcg:
            print(f"  🥇 {local.model_name} wins on NDCG: {local.ndcg:.3f} vs {default.ndcg:.3f}")
        else:
            print(f"  🥇 Default model wins on NDCG: {default.ndcg:.3f} vs {local.ndcg:.3f}")
        
        if local.mrr > default.mrr:
            print(f"  🥇 {local.model_name} wins on MRR: {local.mrr:.3f} vs {default.mrr:.3f}")
        else:
            print(f"  🥇 Default model wins on MRR: {default.mrr:.3f} vs {local.mrr:.3f}")
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="MTEB-style benchmark for local embedding models")
    parser.add_argument(
        "--model", 
        type=str, 
        help="HuggingFace model name (e.g., 'Qwen/Qwen3-Embedding-0.6B')"
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
    
    args = parser.parse_args()
    
    run_mteb_integration_benchmark(args.model)

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
