#!/usr/bin/env python3
"""
CI-Ready MTEB Benchmark for Local Embedding Models
Optimized for continuous integration with caching and parallel execution.
"""

import sys
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse
import subprocess
import tempfile

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def run_ci_benchmark(model_name: str = None, cache_dir: str = None, output_file: str = None):
    """Run benchmark optimized for CI environments."""
    print("🚀 CI-Ready MTEB Benchmark")
    print("=" * 50)
    
    # Set up environment
    if cache_dir:
        os.environ['MTEB_CACHE_DIR'] = cache_dir
    
    # Run the main benchmark
    cmd = [
        sys.executable, 
        "mteb_integration_benchmark.py",
        "--dataset-size", "50",
        "--test-queries", "25"
    ]
    
    if model_name:
        cmd.extend(["--model", model_name])
    
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"❌ Benchmark failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        print("✅ Benchmark completed successfully")
        print(result.stdout)
        
        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            print(f"📄 Results saved to {output_file}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return False

def main():
    """Main function for CI benchmark."""
    parser = argparse.ArgumentParser(description="CI-ready MTEB benchmark")
    parser.add_argument("--model", type=str, help="Model name to test")
    parser.add_argument("--cache-dir", type=str, help="Cache directory")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    success = run_ci_benchmark(
        model_name=args.model,
        cache_dir=args.cache_dir,
        output_file=args.output
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
