# Property-Based-Testing-for-NetworkX
This project implements property-based tests in NetworkX using the Hypothesis framework.The target module is networkx.algorithms.core, which implements the k-core decomposition family of algorithms: core number, k core, k shell, k crust, and k truss. 

Algorithms Tested :
1. core_number
2. k_core
3. k_shell
4. k_crust
5. k_truss

**Testing Approach**
Hypothesis is used to automatically generate a wide range of graphs, including:
Random graphs (Erdős–Rényi), Complete graphs, Path graphs, Star graphs, Cycle graphs, Scale-free graphs (Barabási–Albert)

This ensures coverage across:
1. Sparse vs dense graphs
2. Structured vs random graphs
3. Edge cases and boundary conditions


**Running the Tests**
Install dependencies:
pip install networkx hypothesis pytest

Run tests:
pytest test_networkx_cores.py -v


**Key Properties Verified**
1. k-Core Degree Invariant
Every node in the k-core has degree ≥ k.
2. Core Nesting
3. Core Number Consistency
4. k-Truss Triangle Support
Each edge participates in at least (k - 2) triangles.
5. Monotonicity under Edge Addition
Adding edges does not decrease core numbers.
6. Idempotence
7. Isomorphism Invariance
Core numbers are invariant under node relabeling.
8. Boundary Cases
Empty graph
Single-node graph
Complete graph

**Notable Finding:**
Correct Interpretation of k-Shell and k-Crust
During testing, it was found that:
k_shell(G, k) = nodes with core_number = k
k_crust(G, k) = nodes with core_number ≤ k

This implies:
k-shell = k-crust

Initial misconception:
It is tempting to assume that k-shell and k-crust are disjoint.
However, property-based testing revealed counterexamples where:
k-shell = k-crust (e.g., when all nodes have small core numbers)

Impact:
This insight corrected an incorrect property and ensured that the tests align with the actual behavior of NetworkX.



