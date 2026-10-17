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