class BKTreeNode:
    """
    A node in a BK-Tree (Burkhard-Keller Tree).
    Each node holds a single word and a dictionary of child nodes 
    keyed by their Levenshtein distance from this node's word.
    """
    def __init__(self, word: str):
        self.word = word
        # Maps integer distances to child BKTreeNodes. 
        # Example: self.children[1] points to a node whose word has a distance of 1 from self.word.
        self.children = {}  

    def insert(self, word: str, dist_func):
        """
        Recursively inserts a new word into the BK-Tree based on its distance from the current node.
        """
        # Calculate the Levenshtein distance between the current node's word and the new word
        d = dist_func(self.word, word)
        
        # If a child already exists at this exact distance, insert recursively into that child's subtree
        if d in self.children:
            self.children[d].insert(word, dist_func)
        else:
            # Otherwise, create a new child node at this distance branch
            self.children[d] = BKTreeNode(word)

    def search(self, query: str, dist_func, frequencies: dict, best=None):
        """
        Searches the BK-tree for the optimal matching word using the triangle inequality 
        to prune branches that cannot possibly contain a better match.
        """
        # 1. Calculate distance between the query word and the current node's word
        d = dist_func(self.word, query)
        
        # 2. Form our sorting criteria tuple: (distance, -frequency, word)
        # Note: -frequency is used because Python's min() looks for the smallest value. 
        # A higher frequency becomes more negative, making it "smaller" and preferred.
        current_score = (d, -frequencies.get(self.word, 0), self.word)
        
        # Update our best match found so far if this node is closer (or ties with better rules)
        if best is None or current_score < best:
            best = current_score

        # 3. Get the best distance found so far. This acts as our search radius threshold (tau).
        best_dist = best[0]
        
        # 4. TRIANGLE INEQUALITY PRUNING:
        # Instead of visiting every child, we only visit children whose distance falls 
        # within the range [d - best_dist, d + best_dist]. 
        # Any child outside this range is mathematically guaranteed to be too far away to beat our current best!
        for dist in range(max(0, d - best_dist), d + best_dist + 1):
            if dist in self.children:
                best = self.children[dist].search(query, dist_func, frequencies, best)
                # Update best_dist dynamically if a closer match was found deeper in the subtree
                best_dist = best[0]
                
        return best


def build_bk_tree(dictionary: list, dist_func) -> BKTreeNode:
    """
    Builds a BK-Tree from a list of dictionary words.
    """
    if not dictionary:
        return None
    # The first word becomes the root of the tree
    root = BKTreeNode(dictionary[0])
    # Insert all subsequent words into the tree
    for word in dictionary[1:]:
        root.insert(word, dist_func)
    return root


def spell_correction(misspelled: list, dictionary: list, frequencies: dict) -> list:
    """
    Returns one corrected dictionary word per input word using an optimized BK-Tree search.
    """
    # Standard dynamic programming Levenshtein distance function
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

    # Build the BK-Tree once upfront for the entire dictionary
    root = build_bk_tree(dictionary, levenshtein)
    
    corrected_words = []
    for word in misspelled:
        if not root:
            corrected_words.append("")
            continue
            
        # Search the BK-Tree for this specific misspelled word
        best_score = root.search(word, levenshtein, frequencies)
        
        # best_score is a tuple: (distance, -frequency, word). 
        # Index [2] extracts the actual corrected word string.
        corrected_words.append(best_score[2])
        
    return corrected_words