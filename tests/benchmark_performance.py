import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm.llm_router import LLMRouter
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.memory.local_memory import LocalMemoryLayer
from src.generation.rag_v2 import RAGGeneratorV2

def benchmark():
    results = {}
    
    # 1. Init Memory
    print("Initializing Memory...")
    start = time.time()
    db_path = "./tests/benchmark_data/memory.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # clean up previous if exists
    if os.path.exists(db_path):
        os.remove(db_path)
    memory = LocalMemoryLayer(db_path=db_path)
    memory.set_performance_mode("quality")
    results["init_memory_sec"] = time.time() - start

    # 2. Init LLM
    print("Initializing LLM Router...")
    start = time.time()
    llm = LLMRouter(ollama_model="qwen2.5:7b", auto_start=True)
    # Wait for LLM to be available
    for i in range(30):
        if llm.health_check()["ollama"]["available"]:
            break
        time.sleep(1)
    results["init_llm_sec"] = time.time() - start
    
    if not llm.ollama_available:
        print("ERROR: Ollama is not available. Ensure qwen2.5:7b is installed.")
        results["error"] = "Ollama not available"
        with open("tests/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)
        sys.exit(1)
        
    llm.set_model("qwen2.5:7b")
    
    # 3. LLM Generation
    print("Benchmarking LLM Generation...")
    start = time.time()
    response = llm.generate("Write a short paragraph about artificial intelligence.", max_tokens=100)
    end = time.time()
    results["llm_generate_sec"] = end - start
    tokens = response.usage.get("completion_tokens", 0)
    if tokens > 0:
        results["llm_tps"] = tokens / (end - start)
    else:
        results["llm_tps"] = 0
    results["llm_tokens_generated"] = tokens
    
    # 4. Init Embeddings
    print("Initializing Embedding Generator...")
    start = time.time()
    embedder = EmbeddingGenerator()
    # 5. Benchmarking Embeddings
    print("Benchmarking Embeddings...")
    from src.document_processing.document_chunk import DocumentChunk
    
    texts = []
    for i in range(100):
        texts.append(DocumentChunk(
            chunk_id=f"test_chunk_{i}",
            content="This is a test sentence number " + str(i),
            source_file="benchmark.pdf",
            source_type="Document",
            page_number=1,
            chunk_index=i,
            start_char=0,
            end_char=0,
            metadata={}
        ))
        
    start = time.time()
    embeddings = embedder.generate_embeddings(texts)
    end = time.time()
    results["embedding_100_chunks_sec"] = end - start
    results["embedding_chunks_per_sec"] = 100 / (end - start)
    
    # 6. Init Milvus
    print("Initializing Milvus Vector DB...")
    start = time.time()
    vdb_path = f"./tests/benchmark_data/carnetlm_{int(time.time())}.db"
    vdb = MilvusVectorDB(db_path=vdb_path, collection_name="benchmark", embedding_dim=embedder.get_embedding_dimension())
    vdb.create_index(use_binary_quantization=False)
    results["init_milvus_sec"] = time.time() - start
    
    # 7. Benchmarking Milvus Insertion
    print("Benchmarking Milvus Insertion...")
    start = time.time()
    vdb.insert_embeddings(embeddings, notebook_id=1)
    end = time.time()
    results["milvus_insert_100_chunks_sec"] = end - start
    
    # 8. Benchmarking Vector Search
    print("Benchmarking Vector Search...")
    query_emb = embedder.generate_query_embedding("test sentence 50")
    start = time.time()
    search_results = vdb.search(query_emb.tolist(), limit=10, notebook_id=1)
    end = time.time()
    results["milvus_search_sec"] = end - start
    
    # 9. Benchmarking RAG Generator V2
    print("Benchmarking RAG Generator V2...")
    start = time.time()
    rag = RAGGeneratorV2(llm_router=llm, embedding_generator=embedder, vector_db=vdb, memory=memory)
    results["init_rag_sec"] = time.time() - start
    
    # Add fake sources to memory so it doesn't fail early
    source_id = memory.save_source({"name": "benchmark.pdf", "type": "Document", "index_status": "ready"}, notebook_id=1)
    
    print("Benchmarking End-to-End RAG Query...")
    start = time.time()
    rag_result = rag.generate_response("What is test sentence 50?", notebook_id=1)
    end = time.time()
    results["rag_end_to_end_query_sec"] = end - start
    results["rag_retrieval_count"] = rag_result.retrieval_count
    
    # Cleanup
    llm.close()
    vdb.close()
    memory.close()
    
    out_path = Path("tests/benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("Benchmark complete. Results saved to tests/benchmark_results.json")

if __name__ == "__main__":
    benchmark()
