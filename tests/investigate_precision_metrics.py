"""
Investigation script for Information Retrieval (IR) metrics and Precision@5.
Compares:
1. Original synthetic benchmark (why Precision@5 was reported as 1.0000).
2. Ground-truth passage retrieval with unique targets (single-target IR evaluation).
3. Realistic IR evaluation with challenging queries (Lexical vs Semantic vs Hybrid).
"""

import sys
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.document_processing.document_chunk import DocumentChunk
from src.vector_database.milvus_vector_db import MilvusVectorDB
from src.generation.hybrid_search import BM25Index, reciprocal_rank_fusion


def run_original_synthetic_setup():
    print("=" * 70)
    print("PART 1: ANALYZING ORIGINAL SYNTHETIC IR BENCHMARK SETUP")
    print("=" * 70)

    temp_dir = Path(tempfile.mkdtemp())
    vdb_path = str(temp_dir / "milvus_orig.db")

    embedder = EmbeddingGenerator(use_cache=True)
    vdb = MilvusVectorDB(db_path=vdb_path, collection_name="orig_bench", embedding_dim=384)
    vdb.create_index(use_binary_quantization=False)

    domains = [
        ("Quantum Computing", "Qubits, quantum superposition, entanglement, and Shor's prime factorization algorithm."),
        ("Neuroscience", "Synaptic plasticity, long-term potentiation, neurotransmitter receptor densities in hippocampus."),
        ("Distributed Systems", "Raft consensus, Paxos protocol, Byzantine fault tolerance, and eventual consistency models."),
        ("Cardiology", "Myocardial infarction, ventricular tachycardia, electrocardiogram ST elevation, and coronary angioplasty."),
        ("Aerodynamics", "Navier-Stokes equations, boundary layer turbulence, supersonic shock waves, and airfoil lift drag ratios.")
    ]

    chunks = []
    bm25_docs = []
    idx = 0
    for d_name, d_desc in domains:
        for j in range(30):
            c_id = f"c_{idx}"
            text = f"[{d_name} Section {j}] {d_desc} Additional elaboration on topic {j}: empirical validation and mathematical modeling."
            chunks.append(DocumentChunk(
                chunk_id=c_id,
                content=text,
                source_file=f"{d_name.lower().replace(' ', '_')}.pdf",
                source_type="Document",
                page_number=(j // 5) + 1,
                chunk_index=j,
                metadata={"topic": d_name}
            ))
            bm25_docs.append({"id": c_id, "content": text})
            idx += 1

    embs = embedder.generate_embeddings(chunks)
    vdb.insert_embeddings(embs, notebook_id=1)

    bm25 = BM25Index()
    bm25.build_index(bm25_docs)

    eval_queries = [
        {"query": "prime factorization with quantum qubits", "domain": "quantum"},
        {"query": "superposition and Shor algorithm quantum processors", "domain": "quantum"},
        {"query": "synaptic plasticity and long-term potentiation in hippocampus", "domain": "synaptic"},
        {"query": "neurotransmitter receptor densities in brain neural networks", "domain": "neuroscience"},
        {"query": "Raft consensus and Paxos protocol state machine replication", "domain": "raft"},
        {"query": "Byzantine fault tolerance and eventual consistency in distributed systems", "domain": "byzantine"},
        {"query": "myocardial infarction and ventricular tachycardia electrocardiogram", "domain": "cardiology"},
        {"query": "coronary angioplasty and ST elevation diagnosis in cardiology", "domain": "coronary"},
        {"query": "Navier-Stokes equations for boundary layer turbulence", "domain": "navier"},
        {"query": "supersonic shock waves and airfoil aerodynamic lift drag ratios", "domain": "aerodynamics"},
    ]

    def is_match(hit, domain):
        text = (hit.get("content", "") + " " + str(hit.get("citation", {}).get("source_file", "")) + " " + str(hit.get("metadata", {}))).lower()
        return domain.lower() in text

    for method in ["dense_only", "bm25_only", "hybrid_rrf"]:
        p5_list = []
        hit5_list = []
        mrr5_list = []
        for item in eval_queries:
            q = item["query"]
            dom = item["domain"]
            if method == "dense_only":
                q_vec = embedder.generate_query_embedding(q).tolist()
                hits = vdb.search(q_vec, limit=5, notebook_id=1)
            elif method == "bm25_only":
                pairs = bm25.search(q, k=5)
                hits = [bm25.get_document(i) for i, _ in pairs if bm25.get_document(i)]
            else:
                q_vec = embedder.generate_query_embedding(q).tolist()
                d_res = vdb.search(q_vec, limit=5, notebook_id=1)
                s_hits = bm25.search(q, k=5)
                fused = reciprocal_rank_fusion(d_res, s_hits, bm25_docs=bm25, k=60)
                hits = fused[:5]

            # top-5 evaluation
            top_5 = hits[:5]
            matches = [is_match(h, dom) for h in top_5]
            hit5 = 1.0 if any(matches) else 0.0
            p5 = sum(matches) / 5.0
            rr = 0.0
            for r_idx, m in enumerate(matches):
                if m:
                    rr = 1.0 / (r_idx + 1)
                    break
            hit5_list.append(hit5)
            p5_list.append(p5)
            mrr5_list.append(rr)

        avg_hit5 = sum(hit5_list) / len(hit5_list)
        avg_mrr5 = sum(mrr5_list) / len(mrr5_list)
        avg_p5 = sum(p5_list) / len(p5_list)
        print(f"[{method:10s}] Hit Rate@5: {avg_hit5:.4f} | MRR@5: {avg_mrr5:.4f} | Precision@5: {avg_p5:.4f}")

    vdb.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_realistic_ir_setup():
    print("\n" + "=" * 70)
    print("PART 2: REALISTIC PASSAGE RETRIEVAL EVALUATION")
    print("=" * 70)
    print("Evaluating specific fact retrieval with unique target chunks & distractors.\n")

    temp_dir = Path(tempfile.mkdtemp())
    vdb_path = str(temp_dir / "milvus_real.db")

    embedder = EmbeddingGenerator(use_cache=False)
    vdb = MilvusVectorDB(db_path=vdb_path, collection_name="real_bench", embedding_dim=384)
    vdb.create_index(use_binary_quantization=False)

    # 10 specific target topics with 1 distinct gold chunk + 4 distractor chunks each (50 total chunks)
    # Plus hard negatives and vocabulary mismatch cases
    corpus_data = [
        # Domain 1: Quantum Computing (Chunks 0..4)
        ("chunk_0", "Shor's algorithm uses quantum Fourier transform to achieve polynomial time prime factorization of integers.", "target_shor"),
        ("chunk_1", "Grover's algorithm provides quadratic speedup for unstructured database search in O(sqrt(N)) time.", "distractor_quantum"),
        ("chunk_2", "Quantum key distribution relies on BB84 protocol using polarized single photon states.", "distractor_quantum"),
        ("chunk_3", "Surface codes provide topological quantum error correction with a fault-tolerance threshold near 1%.", "distractor_quantum"),
        ("chunk_4", "Transmon superconducting qubits operate in the microwave frequency domain cooled to 15 millikelvin.", "distractor_quantum"),

        # Domain 2: Distributed Systems (Chunks 5..9)
        ("chunk_5", "The Raft consensus algorithm decomposes state machine replication into leader election, log replication, and safety.", "target_raft"),
        ("chunk_6", "Paxos guarantees consensus under asynchronous network partitions but requires multi-phase commit rounds.", "distractor_dist"),
        ("chunk_7", "Vector clocks capture causal ordering of distributed events across partially ordered asynchronous nodes.", "distractor_dist"),
        ("chunk_8", "Consistent hashing with virtual nodes minimizes key redistribution when cluster topology changes.", "distractor_dist"),
        ("chunk_9", "Two-phase locking protocol enforces conflict serializability in distributed database transactions.", "distractor_dist"),

        # Domain 3: Machine Learning / NLP (Chunks 10..14)
        ("chunk_10", "FlashAttention reduces memory overhead by tiling softmax computation across GPU SRAM blocks.", "target_flashatt"),
        ("chunk_11", "LoRA low-rank adaptation freezes pretrained model weights and injects trainable rank decomposition matrices.", "distractor_nlp"),
        ("chunk_12", "Rotary position embeddings RoPE encode token relative positions via complex rotation matrices.", "distractor_nlp"),
        ("chunk_13", "Direct Preference Optimization DPO eliminates reinforcement learning reward model fitting by policy reparameterization.", "distractor_nlp"),
        ("chunk_14", "Speculative decoding accelerates autoregressive generation using a small draft model verified by the target model.", "distractor_nlp"),

        # Domain 4: Security / Systems (Chunks 15..19)
        ("chunk_15", "CVE-2024-38077 is a remote code execution vulnerability in Windows Remote Desktop Licensing Service.", "target_cve"),
        ("chunk_16", "Address Space Layout Randomization ASLR randomizes memory addresses of stack, heap, and libraries.", "distractor_sec"),
        ("chunk_17", "Rowhammer attacks induce bit flips in adjacent DRAM memory rows via rapid alternating voltage pulses.", "distractor_sec"),
        ("chunk_18", "Control Flow Guard CFG verifies indirect call targets against a valid bitmap compiled by MSVC.", "distractor_sec"),
        ("chunk_19", "Spectre speculative execution side channel leaks secret cache data across speculative branch mispredictions.", "distractor_sec"),

        # Domain 5: Cardiology / Medicine (Chunks 20..24)
        ("chunk_20", "Cardiac troponin I is the biomarker of choice for detecting acute myocardial infarction necrosis.", "target_troponin"),
        ("chunk_21", "Atrial fibrillation is characterized by irregularly irregular R-R intervals and absent P waves on ECG.", "distractor_med"),
        ("chunk_22", "Coronary artery bypass grafting CABG restores blood flow past occluded coronary arteries.", "distractor_med"),
        ("chunk_23", "Ventricular tachycardia produces wide QRS complexes exceeding 120 milliseconds at rates above 100 bpm.", "distractor_med"),
        ("chunk_24", "Beta-blockers decrease myocardial oxygen demand by reducing heart rate and contractility.", "distractor_med"),
    ]

    import hashlib

    chunks = []
    bm25_docs = []
    target_id_map = {}

    for idx, (label, content, query_tag) in enumerate(corpus_data):
        c_hash = hashlib.md5(content.encode()).hexdigest()
        c_id = f"chunk_{c_hash}"
        target_id_map[query_tag] = c_id

        chunks.append(DocumentChunk(
            chunk_id=c_id,
            content=content,
            source_file=f"{label}.txt",
            source_type="Document",
            chunk_index=idx,
            metadata={"label": label, "query_tag": query_tag}
        ))
        bm25_docs.append({"id": c_id, "content": content})

    embs = embedder.generate_embeddings(chunks)
    vdb.insert_embeddings(embs, notebook_id=1)

    bm25 = BM25Index()
    bm25.build_index(bm25_docs)

    # Test Queries categorized by type:
    queries = [
        # Exact Identifier / Acronym (BM25 favored)
        {"q": "CVE-2024-38077 Windows RDL vulnerability", "target": target_id_map["target_cve"], "type": "exact_keyword"},
        {"q": "BB84 protocol polarized photon QKD", "target": target_id_map["distractor_quantum"], "type": "exact_keyword"},

        # Semantic Paraphrase without exact keyword overlap (Dense favored)
        {"q": "How to speed up transformer attention memory without saving entire attention matrix", "target": target_id_map["target_flashatt"], "type": "semantic_paraphrase"},
        {"q": "Protocol where nodes elect a single authority to manage replicated logs", "target": target_id_map["target_raft"], "type": "semantic_paraphrase"},
        {"q": "Biomarker blood test indicating heart muscle cell necrosis after heart attack", "target": target_id_map["target_troponin"], "type": "semantic_paraphrase"},
        {"q": "Algorithm that breaks integer cryptography into factors using quantum Fourier", "target": target_id_map["target_shor"], "type": "semantic_paraphrase"},

        # Mixed conceptual + terminology
        {"q": "hardware bit flip vulnerability from hammering memory chips", "target": target_id_map["distractor_sec"], "type": "mixed"},
        {"q": "ECG diagnostic features of atrial fibrillation irregular rhythm", "target": target_id_map["distractor_med"], "type": "mixed"},
        {"q": "fine-tuning LLMs by injecting small low rank parameter matrices", "target": target_id_map["distractor_nlp"], "type": "mixed"},
        {"q": "bypassing RL reward model in language model alignment", "target": target_id_map["distractor_nlp"], "type": "mixed"},
    ]

    methods = ["dense_only", "bm25_only", "hybrid_rrf"]
    results = {}

    for method in methods:
        hit1_list, hit3_list, hit5_list = [], [], []
        mrr5_list = []
        p5_list = []  # Single-target precision: 1/5 if target in top 5, else 0

        for item in queries:
            q = item["q"]
            target = item["target"]

            if method == "dense_only":
                q_vec = embedder.generate_query_embedding(q).tolist()
                hits = vdb.search(q_vec, limit=5, notebook_id=1)
            elif method == "bm25_only":
                pairs = bm25.search(q, k=5)
                hits = [bm25.get_document(i) for i, _ in pairs if bm25.get_document(i)]
            else:
                q_vec = embedder.generate_query_embedding(q).tolist()
                d_res = vdb.search(q_vec, limit=5, notebook_id=1)
                s_hits = bm25.search(q, k=5)
                fused = reciprocal_rank_fusion(d_res, s_hits, bm25_docs=bm25, k=60)
                hits = fused[:5]

            retrieved_ids = [h.get("id", "") for h in hits[:5]]

            # Hit Rate
            hit1 = 1.0 if target in retrieved_ids[:1] else 0.0
            hit3 = 1.0 if target in retrieved_ids[:3] else 0.0
            hit5 = 1.0 if target in retrieved_ids[:5] else 0.0

            # MRR
            rr = 0.0
            if target in retrieved_ids:
                rank = retrieved_ids.index(target) + 1
                rr = 1.0 / rank

            # Single-target Precision@5: |{target} ∩ Top-5| / 5
            # Since there is exactly 1 relevant passage in the corpus for the fact:
            p5 = 1.0 / 5.0 if target in retrieved_ids[:5] else 0.0

            hit1_list.append(hit1)
            hit3_list.append(hit3)
            hit5_list.append(hit5)
            mrr5_list.append(rr)
            p5_list.append(p5)

        n = len(queries)
        results[method] = {
            "hit@1": sum(hit1_list) / n,
            "hit@3": sum(hit3_list) / n,
            "hit@5": sum(hit5_list) / n,
            "mrr@5": sum(mrr5_list) / n,
            "p@5_single_target": sum(p5_list) / n,
        }

        print(f"[{method.upper():11s}] Hit@1: {results[method]['hit@1']:.2f} | "
              f"Hit@3: {results[method]['hit@3']:.2f} | "
              f"Hit@5: {results[method]['hit@5']:.2f} | "
              f"MRR@5: {results[method]['mrr@5']:.4f} | "
              f"Single-Target P@5: {results[method]['p@5_single_target']:.4f}")

    vdb.close()
    shutil.rmtree(temp_dir, ignore_errors=True)
    return results


if __name__ == "__main__":
    run_original_synthetic_setup()
    run_realistic_ir_setup()
