"""
Benchmark comparing Embedding Inference with Cache OFF (real model speed) vs Cache ON.
Evaluates FastEmbed ('BAAI/bge-small-en-v1.5') across batch sizes [1, 8, 16, 32, 64].
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.document_processing.document_chunk import DocumentChunk


def run_benchmark():
    batch_sizes = [1, 8, 16, 32, 64]
    trials = 5

    # 128 unique test chunks matching exact original benchmark passage strings
    test_chunks = [
        DocumentChunk(
            chunk_id=f"chunk_{i}",
            content=(
                f"Experimental passage number {i} evaluating the computational latency "
                f"and vector density of transformer embeddings under ONNX Runtime execution."
            ),
            source_file="synthetic.txt",
            source_type="txt",
            chunk_index=i,
            metadata={},
        )
        for i in range(128)
    ]

    print("=" * 70)
    print("EMBEDDING INFERENCE BENCHMARK: CACHE OFF (REAL MODEL INFERENCE)")
    print("=" * 70)

    embedder_nocache = EmbeddingGenerator(use_cache=False)
    # Warmup model
    embedder_nocache.generate_embeddings(test_chunks[:4])

    nocache_results = {}
    for bs in batch_sizes:
        sub_chunks = test_chunks[:bs]
        latencies = []
        for _ in range(trials):
            t0 = time.perf_counter()
            embedder_nocache.generate_embeddings(sub_chunks)
            lat = time.perf_counter() - t0
            latencies.append(lat)

        mean_lat_sec = statistics.mean(latencies)
        total_lat_ms = mean_lat_sec * 1000.0
        per_chunk_ms = total_lat_ms / bs
        throughput = bs / mean_lat_sec

        nocache_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "total_latency_ms": round(total_lat_ms, 2),
            "per_chunk_latency_ms": round(per_chunk_ms, 3),
            "throughput_chunks_per_sec": round(throughput, 2),
        }
        print(
            f"Batch {bs:2d}: Total {total_lat_ms:6.2f} ms | "
            f"Per-chunk {per_chunk_ms:6.3f} ms | "
            f"Throughput {throughput:7.2f} chunks/sec"
        )

    # Calculate speedup relative to batch 1
    base_nocache_tp = nocache_results["batch_1"]["throughput_chunks_per_sec"]
    for k, v in nocache_results.items():
        v["speedup_factor"] = round(v["throughput_chunks_per_sec"] / base_nocache_tp, 2)

    print("\n" + "=" * 70)
    print("EMBEDDING INFERENCE BENCHMARK: CACHE ON (CACHE HIT REPLAY)")
    print("=" * 70)

    # Use a temporary cache db to guarantee controlled cache hits
    import tempfile
    from src.embeddings.embed_cache import EmbedCache

    temp_db = Path(tempfile.gettempdir()) / "bench_embed_cache.db"
    if temp_db.exists():
        temp_db.unlink(missing_ok=True)

    cache = EmbedCache(db_path=str(temp_db))
    embedder_cached = EmbeddingGenerator(use_cache=True)
    embedder_cached._cache = cache

    # Populate cache first with all chunks
    embedder_cached.generate_embeddings(test_chunks[:64])

    cached_results = {}
    for bs in batch_sizes:
        sub_chunks = test_chunks[:bs]
        latencies = []
        for _ in range(trials):
            t0 = time.perf_counter()
            embedder_cached.generate_embeddings(sub_chunks)
            lat = time.perf_counter() - t0
            latencies.append(lat)

        mean_lat_sec = statistics.mean(latencies)
        total_lat_ms = mean_lat_sec * 1000.0
        per_chunk_ms = total_lat_ms / bs
        throughput = bs / mean_lat_sec

        cached_results[f"batch_{bs}"] = {
            "batch_size": bs,
            "total_latency_ms": round(total_lat_ms, 2),
            "per_chunk_latency_ms": round(per_chunk_ms, 3),
            "throughput_chunks_per_sec": round(throughput, 2),
        }
        print(
            f"Batch {bs:2d} (Cached): Total {total_lat_ms:6.2f} ms | "
            f"Per-chunk {per_chunk_ms:6.3f} ms | "
            f"Throughput {throughput:9.2f} chunks/sec"
        )

    base_cached_tp = cached_results["batch_1"]["throughput_chunks_per_sec"]
    for k, v in cached_results.items():
        v["speedup_factor"] = round(v["throughput_chunks_per_sec"] / base_cached_tp, 2)

    cache.close()
    if temp_db.exists():
        temp_db.unlink(missing_ok=True)

    out = {
        "cache_off_real_inference": nocache_results,
        "cache_on_cached_hits": cached_results,
    }

    out_path = ROOT_DIR / "tests" / "benchmark_embedding_cache_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nBenchmark comparison saved to:", out_path)
    return out


if __name__ == "__main__":
    run_benchmark()
