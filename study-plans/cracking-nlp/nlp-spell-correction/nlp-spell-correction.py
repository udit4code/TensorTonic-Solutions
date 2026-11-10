class BKTreeNode:
    def __init__(self, word: str):
        self.word = word
        self.children = {}  # Maps distance -> BKTreeNode

    def insert(self, word: str, dist_func):
        d = dist_func(self.word, word)
        if d in self.children:
            self.children[d].insert(word, dist_func)
        else:
            self.children[d] = BKTreeNode(word)

    def search(self, query: str, dist_func, frequencies: dict, best=None):
        """
        Recursively searches the BK-tree for the best candidate matching:
        1. Smallest Levenshtein distance
        2. Higher frequency
        3. Lexicographical order
        """
        d = dist_func(self.word, query)
        
        # Current candidate's score tuple: (distance, -frequency, word)
        current_score = (d, -frequencies.get(self.word, 0), self.word)
        
        if best is None or current_score < best:
            best = current_score

        # Determine the search radius based on the best distance found so far.
        # Since we want the minimum distance, our max allowable distance threshold 
        # is bounded by the best distance found so far.
        best_dist = best[0]
        
        # Triangle inequality range: [d - best_dist, d + best_dist]
        for dist in range(max(0, d - best_dist), d + best_dist + 1):
            if dist in self.children:
                best = self.children[dist].search(query, dist_func, frequencies, best)
                # Update best_dist in case a closer match was found deeper down
                best_dist = best[0]
                
        return best


def build_bk_tree(dictionary: list, dist_func) -> BKTreeNode:
    if not dictionary:
        return None
    root = BKTreeNode(dictionary[0])
    for word in dictionary[1:]:
        root.insert(word, dist_func)
    return root


def spell_correction(misspelled: list, dictionary: list, frequencies: dict) -> list:
    """
    Returns one corrected dictionary word per input word using a BK-Tree.
    """
    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    # Build the BK-Tree once for the entire dictionary
    root = build_bk_tree(dictionary, levenshtein)
    
    corrected_words = []
    for word in misspelled:
        if not root:
            corrected_words.append("")
            continue
            
        # Search the tree for the optimal match
        best_score = root.search(word, levenshtein, frequencies)
        # best_score format: (distance, -frequency, word)
        corrected_words.append(best_score[2])
        
    return corrected_words