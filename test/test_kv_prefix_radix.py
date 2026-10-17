"""Regression checks for compressed paths and compatibility with the token trie."""
import random
import runpy
import unittest
from pathlib import Path

# Run the command from the root of the repo : 
# python3 -m unittest discover -s test
implementation = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / 'study-plans/language-modeling-from-scratch'
        / 'cs336-l18-kv-prefix-trie/cs336-l18-kv-prefix-trie.py'
    )
)
KVCacheTrie = implementation['KVCacheTrie']
KVCacheRadixTree = implementation['KVCacheRadixTree']


class RadixTreeTests(unittest.TestCase):
    def test_compressed_edge_and_partial_matches(self):
        tree = KVCacheRadixTree()
        tree.insert([10, 20, 30, 40])
        edge = tree.root.children[10]
        self.assertEqual(edge.tokens, [10, 20, 30, 40])
        self.assertEqual(edge.children, {})
        for request, length, suffix in [([], 0, []), ([99], 0, [99]),
                                        ([10, 20], 2, []),
                                        ([10, 20, 99], 2, [99])]:
            self.assertEqual(tree.search_prefix(request), {
                'matched_length': length, 'cache_node': length,
                'uncached_suffix': suffix,
            })

    def test_splits_preserve_ids_and_descendants(self):
        cached = [[10, 20, 30, 40], [10, 20, 50], [10, 60],
                  [10, 20, 30, 70], [10, 20, 30, 40, 80]]
        plain, radix = KVCacheTrie(), KVCacheRadixTree()
        for seq in cached:
            plain.insert(seq)
            radix.insert(seq)
            for request in cached + [[10, 20], [10, 20, 30, 99]]:
                self.assertEqual(radix.search_prefix(request), plain.search_prefix(request))
        self.assertEqual(radix.root.children[10].tokens, [10])
        self.assertEqual(radix.next_node_id, plain.next_node_id)

    def test_duplicates_shorter_prefixes_and_input_ownership(self):
        tree = KVCacheRadixTree()
        sequence = [1, 2, 3]
        tree.insert(sequence)
        sequence[1] = 99
        for seq in [[], [1], [1, 2], [1, 2, 3]]:
            tree.insert(seq)
        self.assertEqual(tree.next_node_id, 4)
        self.assertEqual(tree.root.children[1].tokens, [1, 2, 3])
        self.assertEqual(tree.search_prefix([1, 2, 3])['matched_length'], 3)

    def test_randomized_equivalence_and_compression(self):
        rng = random.Random(336)
        for _ in range(100):
            plain, radix = KVCacheTrie(), KVCacheRadixTree()
            for _ in range(25):
                seq = [rng.randrange(5) for _ in range(rng.randrange(12))]
                plain.insert(seq)
                radix.insert(seq)
                requests = [seq[:end] for end in range(len(seq) + 1)]
                requests += [[rng.randrange(5) for _ in range(rng.randrange(15))]
                             for _ in range(10)]
                for request in requests:
                    self.assertEqual(radix.search_prefix(request), plain.search_prefix(request))
                self.assertEqual(radix.next_node_id, plain.next_node_id)
                pending = list(radix.root.children.values())
                while pending:
                    node = pending.pop()
                    self.assertTrue(node.tokens)
                    self.assertNotEqual(len(node.children), 1)
                    for token, child in node.children.items():
                        self.assertEqual(token, child.tokens[0])
                    pending.extend(node.children.values())

    def test_wrapper(self):
        self.assertEqual(implementation['kv_prefix_trie'](
            [[10, 20, 30, 40], [10, 20, 50]], [[10, 20, 50, 99]]
        ), {'matches': [{'matched_length': 3, 'cache_node': 5, 'uncached_suffix': [99]}]})


if __name__ == '__main__':
    unittest.main()
