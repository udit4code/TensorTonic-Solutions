# <span style="font-size: 20px;">DBSCAN</span>

<span style="font-size: 14px;">DBSCAN (Ester et al., 1996) is a density-based clustering algorithm that can discover clusters of arbitrary shape and automatically identifies noise points. It is one of the most cited algorithms in data mining.</span>

---

## <span style="font-size: 16px;">Key Concepts</span>

- **<span style="font-size: 14px;">Core point</span>**: <span style="font-size: 14px;">has at least</span> `min_samples` <span style="font-size: 14px;">points within distance</span> $\varepsilon$ <span style="font-size: 14px;">(including itself)</span>
- **<span style="font-size: 14px;">Border point</span>**: <span style="font-size: 14px;">not a core point but within</span> $\varepsilon$ <span style="font-size: 14px;">of a core point</span>
- **<span style="font-size: 14px;">Noise point</span>**: <span style="font-size: 14px;">neither core nor border, labeled -1</span>
- **<span style="font-size: 14px;">Density-reachable</span>**: <span style="font-size: 14px;">point</span> $q$ <span style="font-size: 14px;">is density-reachable from</span> $p$ <span style="font-size: 14px;">if there is a chain of core points connecting them</span>

---

## <span style="font-size: 16px;">Algorithm</span>

### DBSCAN Algorithm

For each unvisited point \( p \):

1. Mark \( p \) as visited.
2. Find all points within distance \( \varepsilon \).
3. If the number of neighbors is less than `min_samples`:
   - Mark \( p \) as noise (`-1`).
4. Otherwise:
   - Create a new cluster.
   - Add all neighbors to a queue.
   - While the queue is not empty:
     - Remove a point from the queue.
     - If it is a core point (`neighbors >= min_samples`):
       - Add its neighbors to the queue.
     - Assign the point to the cluster.

---

### Choosing Parameters

- **`eps` (\( \varepsilon \))**
  - Neighborhood radius.
  - Often chosen using a k-distance plot and looking for the elbow.

- **`min_samples`**
  - Typical values:
    - \( d + 1 \)
    - \( 2d \)
  - Higher values:
    - More robust to noise.
    - Fewer clusters.
    - More points labeled as noise.

---

## <span style="font-size: 16px;">Advantages</span>

- <span style="font-size: 14px;">Does not require specifying</span> $k$ <span style="font-size: 14px;">(number of clusters)</span>
- <span style="font-size: 14px;">Can find clusters of arbitrary shape (non-convex, elongated, etc.)</span>
- <span style="font-size: 14px;">Robust to outliers (they become noise)</span>
- <span style="font-size: 14px;">Deterministic for core points (border point assignment can vary with processing order)</span>

---

## <span style="font-size: 16px;">Limitations</span>

- <span style="font-size: 14px;">Struggles with clusters of varying density (a single</span> $\varepsilon$ <span style="font-size: 14px;">cannot fit both dense and sparse clusters)</span>
- <span style="font-size: 14px;">Performance degrades in high dimensions (distances become less meaningful)</span>
- <span style="font-size: 14px;">Naive implementation is</span> $O(n^2)$<span style="font-size: 14px;">; with spatial indexing (KD-tree) it can be</span> $O(n \log n)$

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: How does DBSCAN handle varying density?**</span>
  <span style="font-size: 14px;">A: Poorly with a single epsilon. HDBSCAN extends DBSCAN to handle varying densities by building a hierarchy of clusterings and extracting the most stable clusters.</span>

- <span style="font-size: 14px;">**Q: What is OPTICS?**</span>
  <span style="font-size: 14px;">A: An extension that produces an ordering of points augmented with reachability distances, from which clusterings at multiple density levels can be extracted.</span>

- <span style="font-size: 14px;">**Q: How do you scale DBSCAN?**</span>
  <span style="font-size: 14px;">A: Use spatial indexing structures like KD-trees or ball trees for neighbor queries. For very large datasets, use approximate methods or distributed implementations.</span>

---
