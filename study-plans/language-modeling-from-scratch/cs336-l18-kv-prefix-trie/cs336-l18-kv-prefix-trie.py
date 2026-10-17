class TrieNode:
    """Represents a single node in the KV-cache Trie."""
    def __init__(self):
        self.children = { } # Maps token_id (int) -> TrieNode
        self.node_id = 0 # Node id is 0 for root and incremented subsequently for new nodes 

class KVCacheTrie:
    """Manages the prefix tree insertion and prefix search for KV-caching."""
    def __init__(self):
        self.root = TrieNode()
        self.next_node_id = 1 # The node_id for the next node to be inserted to the Trie

    def insert(self, sequence: list[int]):
        """Inserts a cached sequence into the trie, assigning incremental node IDs."""
        current = self.root
        for token in sequence:
            if token not in current.children:
                new_node = TrieNode()
                new_node.node_id = self.next_node_id
                # Update the next_node_id
                self.next_node_id += 1
                current.children[token] = new_node
            current = current.children[token]

    def search_prefix(self, request: list[int]) -> dict:
        """Finds the longest matching prefix for a request sequence."""
        current = self.root
        matched_length = 0
        cache_node = 0

        for token in request:
            if token in current.children:
                current = current.children[token]
                matched_length += 1
                cache_node = current.node_id
            else:
                break

        uncached_suffix = request[matched_length:]

        return {
            "matched_length": matched_length,
            "cache_node": cache_node,
            "uncached_suffix": uncached_suffix
        }

def kv_prefix_trie(cached_sequences: list, requests: list) -> dict:
    """
    Returns a dict with matches: records of matched_length, cache_node, and uncached_suffix.
    """
    trie = KVCacheTrie()

    # Step 1 : Build the Trie by inserting cached sequences in input order
    for seq in cached_sequences:
        trie.insert(seq)

    # Step 2 : Evaluate each request against the constructed KV-cache Trie
    matches = [trie.search_prefix(req) for req in requests]

    return {"matches": matches}
