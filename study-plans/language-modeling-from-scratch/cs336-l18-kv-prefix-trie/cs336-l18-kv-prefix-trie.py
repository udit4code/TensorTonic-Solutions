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

class RadixNode:
    """A compressed incoming edge and branches keyed by their first token.

    Each token retains its original prefix ID so partial-edge matches return
    the same cache_node as the token trie. IDs need not be consecutive when
    an existing leaf is extended after other branches have been inserted.
    """

    def __init__(self, tokens: list[int], prefix_ids: list[int]):
        self.tokens = tokens
        self.prefix_ids = prefix_ids
        self.children = {}

    @property
    def node_id(self) -> int:
        """ID of the prefix at the end of this edge; the root is zero."""
        return self.prefix_ids[-1] if self.prefix_ids else 0


class KVCacheRadixTree:
    """Path-compressed trie with the same results and prefix IDs as KVCacheTrie."""

    def __init__(self):
        self.root = RadixNode([], [])
        self.next_node_id = 1

    def insert(self, sequence: list[int]):
        """Insert a cached sequence, splitting an edge only when paths diverge."""
        current = self.root
        position = 0
        while position < len(sequence):
            token = sequence[position]
            child = current.children.get(token)
            if child is None:
                # Copy the suffix so caller mutations cannot change the tree.
                suffix = sequence[position:]
                prefix_ids = list(range(self.next_node_id, self.next_node_id + len(suffix)))
                if current is not self.root and not current.children:
                    # Extend a leaf instead of introducing a unary branch.
                    current.tokens.extend(suffix)
                    current.prefix_ids.extend(prefix_ids)
                else:
                    current.children[token] = RadixNode(suffix, prefix_ids)
                self.next_node_id += len(suffix)
                return

            common = 0
            while (common < len(child.tokens)
                   and position + common < len(sequence)
                   and child.tokens[common] == sequence[position + common]):
                common += 1

            position += common
            if position == len(sequence):
                # All prefixes are cached; no terminal marker or split is needed.
                return
            if common == len(child.tokens):
                current = child
                continue

            # First-token lookup guarantees common >= 1. Preserve old prefix
            # IDs and descendants while turning the common path into a branch.
            branch = RadixNode(child.tokens[:common], child.prefix_ids[:common])
            child.tokens = child.tokens[common:]
            child.prefix_ids = child.prefix_ids[common:]
            branch.children[child.tokens[0]] = child
            current.children[token] = branch

            suffix = sequence[position:]
            prefix_ids = list(range(self.next_node_id, self.next_node_id + len(suffix)))
            branch.children[suffix[0]] = RadixNode(suffix, prefix_ids)
            self.next_node_id += len(suffix)
            return

    def search_prefix(self, request: list[int]) -> dict:
        """Match tokens along compressed edges, including partial-edge matches."""
        current = self.root
        matched_length = 0
        cache_node = 0
        while matched_length < len(request):
            child = current.children.get(request[matched_length])
            if child is None:
                break
            common = 0
            while (common < len(child.tokens)
                   and matched_length < len(request)
                   and child.tokens[common] == request[matched_length]):
                cache_node = child.prefix_ids[common]
                matched_length += 1
                common += 1
            if common < len(child.tokens):
                break
            current = child

        return {
            "matched_length": matched_length,
            "cache_node": cache_node,
            "uncached_suffix": request[matched_length:],
        }


def kv_prefix_trie(cached_sequences: list, requests: list) -> dict:
    """
    Returns a dict with matches: records of matched_length, cache_node, and uncached_suffix.
    """
    # trie = KVCacheTrie()
    trie = KVCacheRadixTree()

    # Step 1 : Build the Trie by inserting cached sequences in input order
    for seq in cached_sequences:
        trie.insert(seq)

    # Step 2 : Evaluate each request against the constructed KV-cache Trie
    matches = [trie.search_prefix(req) for req in requests]

    return {"matches": matches}
