Lecture Note: KV-Cache Prefix Caching &amp; Trie-Based Routing





1. The Core Problem: Why Do We Need Prefix Caching?

When Large Language Models (LLMs) generate text token-by-token (autoregressively), they compute and store Key-Value (KV) tensors for every token in the prompt and response. This is called the KV-Cache.

In production, millions of users often send identical or heavily overlapping prefixes:

- System Prompts: "You are a helpful AI assistant. Always respond in Markdown..." (sent with every single user request).
- Few-Shot Examples: The same 5 examples showing how to format JSON.
- RAG (Retrieval-Augmented Generation): The same ingested company manual injected into prompts for hundreds of employees.

Without caching, the GPU recomputes the exact same KV tensors for these shared prefixes over and over again, wasting massive amounts of compute cycles and GPU memory. KV-Cache Prefix Caching solves this by saving those tensors once and reusing them across requests.

2. The Solution: The Prefix Trie (Radix Tree)

To find matching prefixes instantly across thousands of incoming requests, we organize our cached prompt tokens into a Trie (Prefix Tree):

- The Root: Represents an empty sequence (Node ID 0).
- Edges: Represent individual token IDs.
- Nodes: Represent a valid, cached prefix state in memory, assigned a unique cache_node ID.

When a new request arrives, we walk down the Trie token-by-token until we hit a missing token.

- The longest matched path gives us our reusable cache nodes.
- The remaining tokens form the uncached suffix, which must be computed from scratch.

3. Potential Real-World Applications

1. Enterprise Chatbots: Sharing heavy system instructions and compliance guardrails across thousands of simultaneous employee chats.
2. Coding Assistants (e.g., Copilot): Sharing large repository contexts or codebases that multiple developers are querying against.
3. Multi-Turn Conversations: Reusing the entire conversation history from turns 1 to $N$ when turn $N+1$ arrives, avoiding re-processing past turns.
4. Agentic Workflows &amp; Tool Use: Reusing shared tool-definition prompts that are prepended to complex agent chains.

4. How Is It Implemented in Production? (CPU vs. GPU)

In production-grade inference engines (such as vLLM, SGLang, or TensorRT-LLM), the architecture cleanly splits responsibilities between the CPU and the GPU:

A. The Metadata Index (The Trie) → CPU Implementation

- Where it lives: Main system memory (RAM) on the CPU.
- Why: The Trie structure involves pointer-chasing, hash map lookups, string/token traversals, and thread locks when concurrent requests arrive. CPUs excel at this control-plane logic. Running the Trie match lookup on the CPU takes microseconds and keeps the GPU unburdened.



Because production serving engines typically implement the control plane and block management in **C++** for performance, while exposing interfaces in **Python**, both the **C++ struct definitions** and their **Python equivalents** are outlined below.
### 1. C++ Implementation (Systems / Inference Engine Level)

In a high-performance C++ backend, these structs manage the Trie topology and map logical token paths to physical GPU memory blocks (via PagedAttention). 



```
#include &lt;unordered_map&gt;
#include &lt;vector&gt;
#include &lt;memory&gt;
#include &lt;cstdint&gt;

// Represents a node in the Prefix Trie (Radix Tree)
struct TrieNode {
    int64_t node_id;                                // Unique identifier for the cache node (0 = root)
    std::vector&lt;int32_t&gt; token_sequence;            // Token slice represented by this node/edge
    int32_t physical_block_id;                      // Assigned physical GPU block ID in HBM
    std::unordered_map&lt;int32_t, std::shared_ptr&lt;TrieNode&gt;&gt; children; // Map: token_id -&gt; child node
    int32_t reference_count;                        // Active requests referencing this node (for LRU/eviction)
    bool is_cached;                                 // True if KV tensors are fully computed and stored in GPU memory

    // Constructor
    TrieNode(int64_t id, int32_t block_id = -1) 
        : node_id(id), physical_block_id(block_id), reference_count(0), is_cached(false) {}
};

// Represents a physical memory block in GPU High-Bandwidth Memory (HBM) via PagedAttention
struct PhysicalBlock {
    int32_t block_id;                               // Unique physical block index in the GPU memory pool
    std::vector&lt;int32_t&gt; token_ids;                 // Tokens stored in this physical block (e.g., block size = 16)
    int32_t ref_count;                              // Reference count for memory management &amp; caching
    bool is_free;                                   // Allocation flag

    PhysicalBlock(int32_t id) : block_id(id), ref_count(0), is_free(true) {}
};

// Execution plan handed from the CPU scheduler to the GPU kernel executor
struct KVRequestExecutionPlan {
    int64_t request_id;                             // Unique request identifier
    int64_t matched_cache_node_id;                  // Deepest matched cache node ID from the Trie
    int32_t matched_token_count;                    // Number of tokens successfully found in the cache
    std::vector&lt;int32_t&gt; physical_block_indices;    // Physical GPU block table (reused prefix blocks + newly allocated blocks)
    std::vector&lt;int32_t&gt; uncached_suffix_tokens;    // Remaining tokens that must be computed from scratch by the GPU
};
```


### 2. Python Implementation (Application / Router Level)

If you are implementing the control-plane scheduler or state tracker in Python, standard classes or `dataclasses` are used: 



```
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrieNode:
    """Represents a single node in the KV-cache Prefix Trie."""
    node_id: int                                    # Unique identifier for the cache node (0 = root)
    physical_block_id: int = -1                     # Assigned physical GPU memory block ID
    children: Dict[int, "TrieNode"] = field(default_factory=dict)  # Map: token_id -&gt; child node
    reference_count: int = 0                        # Active request counter for memory management
    is_cached: bool = False                         # Whether KV tensors are computed and stored in GPU HBM


@dataclass
class KVRequestExecutionPlan:
    """Contains the scheduling decision for a single incoming inference request."""
    request_id: int
    matched_cache_node_id: int                      # The deepest matched cache node ID from the Trie
    matched_token_count: int                        # Number of tokens successfully matched in the cache
    physical_block_indices: List[int]               # Combined GPU block table (reused prefix blocks + new allocations)
    uncached_suffix_tokens: List[int]               # Tokens that must undergo full forward-pass computation
```



B. The Actual KV-Cache Storage → GPU Implementation

- Where it lives: High-Bandwidth Memory (HBM) on the GPU (often managed via PagedAttention blocks, e.g., 16 or 32 tokens per block).
- Why: The actual tensor data needs to be instantly accessible by CUDA kernels during matrix multiplication.

C. The Production Execution Flow

1. Request Arrival: A user request hits the inference router.
2. CPU Lookup: The CPU-based scheduler runs our Trie prefix-matching algorithm against the incoming token sequence.
3. Block Table Mapping: If the Trie finds a match up to token length $K$ (e.g., hitting cache_node 4), it retrieves the physical GPU block indices associated with those nodes.
4. Execution: The CPU hands the execution plan to the GPU: "For this request, do not compute attention for blocks corresponding to tokens 1 through $K$. Just map them to existing GPU memory blocks [102, 105, 112], and only allocate new blocks for the uncached_suffix."

5. Principal's Summary

Prefix caching transforms LLM inference economics. By turning prompt overlapping into a graph traversal problem (via a Trie) and mapping it to decoupled CPU routing and GPU paged memory, modern serving engines slash Time-to-First-Token (TTFT) latency by up to 50–80% on prompt-heavy workloads.