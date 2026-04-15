# =============================================================================
# Property-Based Testing for NetworkX
# Shivani P
# Algorithms Tested: core_number, k_core, k_shell, k_crust, k_truss
# =============================================================================

import networkx as nx
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Graph generation helpers (diverse graph families)
# ---------------------------------------------------------------------------

@st.composite
def diverse_graphs(draw, min_nodes=2, max_nodes=20):
    """
    Generates structurally diverse graphs to improve coverage.

    Strategy:
        Randomly selects from multiple graph families:
        - Erdős–Rényi random graphs
        - Complete graphs
        - Path graphs
        - Star graphs
        - Barabási–Albert graphs
        - Cycle graphs

    Why this matters:
        Using only random graphs may miss important edge cases.
        Different graph families expose different structural behaviors
        critical for validating graph algorithms.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    choice = draw(st.integers(min_value=0, max_value=5))

    if choice == 0:
        #seed = draw(st.integers(0, 10_000))
        #return nx.erdos_renyi_graph(n, seed=seed)
        return nx.erdos_renyi_graph(n, draw(st.floats(0.2, 0.8)))
    elif choice == 1:
        return nx.complete_graph(n)
    elif choice == 2:
        return nx.path_graph(n)
    elif choice == 3:
        return nx.star_graph(n - 1)
    elif choice == 4:
        return nx.barabasi_albert_graph(max(n, 3), 2)
    else:
        return nx.cycle_graph(n)


@st.composite
def graph_with_valid_k(draw):
    """
    Generates (G, k) where k corresponds to a non-empty k-core.

    Ensures:
        - k is valid
        - k-core(G, k) is non-empty
    """
    G = draw(diverse_graphs(min_nodes=5, max_nodes=20))
    assume(G.number_of_nodes() > 0)

    core_nums = nx.core_number(G)
    max_core = max(core_nums.values()) if core_nums else 0
    assume(max_core >= 1)

    k = draw(st.integers(min_value=1, max_value=max_core))
    return G, k


# ---------------------------------------------------------------------------
# Test 1 — Degree Invariant of k-Core
# ---------------------------------------------------------------------------

@given(gk=graph_with_valid_k())
@settings(max_examples=200, deadline=None)
def test_kcore_degree_invariant(gk):
    """
    Property: Every node in the k-core of a graph has degree at least k
              within the k-core subgraph itself.

    Mathematical basis:
        The k-core C_k of a graph G is defined as the maximal induced
        subgraph in which every vertex has degree ≥ k. This is a defining
        property, not a derived one.

        Formally:
            ∀ v ∈ V(C_k),  deg_{C_k}(v) ≥ k

        This condition must hold for every node in the resulting subgraph.
        If any node violates this constraint, the subgraph cannot be a valid
        k-core.

    Test strategy:
        We generate random graphs of varying structure and size, along with
        a valid value of k such that the k-core is non-empty. We compute the
        k-core subgraph C_k and verify that every node in C_k satisfies the
        degree condition within the subgraph.

    Assumptions / preconditions:
        - G is a simple undirected graph (no self-loops or parallel edges).
        - k is chosen such that k-core(G, k) is non-empty.

    Why a failure matters:
        If any node in the computed k-core has degree less than k, the
        algorithm has violated the fundamental definition of a k-core.
        This would indicate a critical correctness error in the
        implementation, invalidating all downstream uses of the result.
    """
    G, k = gk
    C_k = nx.k_core(G, k)
    for node in C_k.nodes():
        assert C_k.degree(node) >= k


# ---------------------------------------------------------------------------
# Test 2 — Nesting Property
# ---------------------------------------------------------------------------

@given(gk=graph_with_valid_k())
@settings(max_examples=200, deadline=None)
def test_kcore_nesting(gk):
    """
    Property: The (k+1)-core of a graph is always a subgraph of the k-core:
                  C_{k+1} ⊆ C_k

    Mathematical basis:
        The k-core is defined as the maximal subgraph where all nodes have
        degree ≥ k. Any node that satisfies the stronger condition
        deg ≥ (k+1) automatically satisfies deg ≥ k.

        Therefore, every node in C_{k+1} must also belong to C_k, implying
        a nested sequence of cores:
            C_1 ⊇ C_2 ⊇ C_3 ⊇ ... ⊇ C_k

        This nesting property is fundamental to core decomposition.

    Test strategy:
        For a generated graph G and valid k, we compute both C_k and C_{k+1}.
        We then verify that every node in C_{k+1} also appears in C_k by
        checking subset containment.

    Assumptions / preconditions:
        - G is a simple undirected graph.
        - k is valid (k ≥ 1).
        - C_{k+1} may be empty; in that case, the property holds trivially.

    Why a failure matters:
        If the nesting property is violated, it means a node appears in a
        deeper core without belonging to a shallower one. This contradicts
        the theoretical foundation of core decomposition and would break
        algorithms that rely on hierarchical structure.
    """
    G, k = gk
    assert set(nx.k_core(G, k + 1)) <= set(nx.k_core(G, k))


# ---------------------------------------------------------------------------
# Test 3 — core_number Consistency
# ---------------------------------------------------------------------------

@given(G=diverse_graphs())
@settings(max_examples=200, deadline=None)
def test_core_number_membership_consistency(G):
    """
    Property: A node v belongs to the k-core if and only if its core number
              is at least k:
                  v ∈ C_k  ⇔  core_number(v) ≥ k

    Mathematical basis:
        The core number of a node v is defined as the largest value k such
        that v belongs to the k-core. Therefore:
            - v must belong to C_{core_number(v)}
            - v must not belong to C_{core_number(v)+1}

        This establishes an exact correspondence between core_number and
        k-core membership.

    Test strategy:
        We compute core_number(G) for all nodes. For each node v with
        core number c, we verify:
            (1) v ∈ C_c
            (2) v ∉ C_{c+1}

        These two conditions together fully characterize the definition.

    Assumptions / preconditions:
        - G is a simple undirected graph.
        - Nodes with degree 0 have core_number = 0.

    Why a failure matters:
        If core_number and k-core membership disagree, it indicates that
        these two functions implement inconsistent notions of coreness.
        This would break applications that rely on them interchangeably.
    """
    core_nums = nx.core_number(G)

    for node, c in core_nums.items():
        if c >= 1:
            assert node in nx.k_core(G, c)
        assert node not in nx.k_core(G, c + 1)


# ---------------------------------------------------------------------------
# Test 4 — Shell and Crust Partition
# ---------------------------------------------------------------------------

@given(gk=graph_with_valid_k())
@settings(max_examples=200, deadline=None)
def test_shell_and_crust_partition(gk):
    """
    Property: The k-shell and k-crust satisfy correct containment
              relationships based on core numbers:
                  - k-shell = nodes with core_number == k
                  - k-crust = nodes with core_number ≤ k
                  - Therefore: k-shell ⊆ k-crust

    Mathematical basis:
        The core decomposition partitions nodes based on their core number:
            - Nodes with core_number < k → strictly in lower shells
            - Nodes with core_number = k → k-shell
            - Nodes with core_number ≤ k → k-crust

        Hence:
            k-shell ⊆ k-crust

    Test strategy:
        For a generated graph G and valid k:
            - Compute core numbers
            - Compute k-shell and k-crust
            - Verify:
                (1) k-shell ⊆ k-crust
                (2) k-crust = {v | core_number(v) ≤ k}
                (3) k-shell = {v | core_number(v) = k}

    Assumptions / preconditions:
        - G is a simple undirected graph
        - k is a valid core level

    Why a failure matters:
        If these relationships fail, it indicates inconsistency between
        core_number, k_shell, and k_crust implementations, breaking the
        correctness of core decomposition utilities.
    """
    G, k = gk
    core_nums = nx.core_number(G)

    kshell = set(nx.k_shell(G, k, core_number=core_nums))
    kcrust = set(nx.k_crust(G, k, core_number=core_nums))

    expected_shell = {v for v, c in core_nums.items() if c == k}
    expected_crust = {v for v, c in core_nums.items() if c <= k}

    assert kshell == expected_shell
    assert kcrust == expected_crust
    assert kshell.issubset(kcrust)
    


# ---------------------------------------------------------------------------
# Test 5 — k-Truss Triangle Support
# ---------------------------------------------------------------------------

@given(n=st.integers(6, 18), p=st.floats(0.4, 0.85), k=st.integers(3, 6))
@settings(max_examples=150, deadline=None)
def test_ktruss_triangle_support(n, p, k):
    """
    Property: Every edge (u, v) in the k-truss of a graph is supported by
              at least (k - 2) triangles within the k-truss subgraph.

    Mathematical basis:
        The k-truss is defined as the maximal subgraph in which every edge
        participates in at least (k - 2) triangles. A triangle supporting
        edge (u, v) consists of a node w such that:
            (u, w) and (v, w) are also edges in the subgraph.

        Formally:
            ∀ (u, v) ∈ E(T_k),
            |{ w : (u, w) ∈ T_k and (v, w) ∈ T_k }| ≥ k - 2

        This is the defining invariant of k-truss.

    Test strategy:
        We generate graphs and compute the k-truss for k ≥ 3. For each edge
        in the resulting subgraph, we count the number of common neighbors
        (triangle support) and verify that it meets the required threshold.

        If the k-truss is empty, the property holds vacuously.

    Assumptions / preconditions:
        - G is a simple undirected graph.
        - k ≥ 3 (minimum meaningful value for k-truss).
        - Triangle support is computed within the truss subgraph.

    Why a failure matters:
        If any edge has fewer than (k - 2) supporting triangles, the
        algorithm has violated the defining property of k-truss. This would
        make the output invalid for applications requiring dense,
        triangle-based connectivity structures.
    """
    G = nx.erdos_renyi_graph(n, p)
    assume(k >= 3)

    try:
        T = nx.k_truss(G, k)
    except nx.NetworkXError:
        return

    if T.number_of_edges() == 0:
        return

    for u, v in T.edges():
        support = len(set(T.neighbors(u)) & set(T.neighbors(v)))
        assert support >= k - 2


# ---------------------------------------------------------------------------
# Test 6 — Monotonicity under Edge Addition
# ---------------------------------------------------------------------------

@given(n=st.integers(4, 15), p=st.floats(0.2, 0.6))
@settings(max_examples=150, deadline=None)
def test_core_number_monotone_under_edge_addition(n, p):
    """
    Property: Applying the k-core operation twice yields the same result
              as applying it once:
                  k_core(k_core(G, k), k) = k_core(G, k)

    Mathematical basis:
        The k-core of a graph is defined as the maximal induced subgraph
        in which every vertex has degree at least k. Once this subgraph
        is extracted, all remaining nodes already satisfy the degree
        constraint within the subgraph itself.

        Therefore, reapplying the k-core operation cannot remove any
        additional nodes, since no node violates the constraint anymore.
        This makes the operation *idempotent*.

    Test strategy:
        We generate a graph G and a valid k such that the k-core is
        non-empty. We compute:
            C1 = k_core(G, k)
            C2 = k_core(C1, k)
        and verify that both node sets are identical.

    Assumptions / preconditions:
        - G is a simple undirected graph.
        - k is chosen such that the k-core exists (non-empty).

    Why a failure matters:
        If applying k_core twice changes the result, it means the first
        application did not fully enforce the degree constraint. This
        would indicate that the algorithm is not computing the maximal
        k-core correctly, violating its definition.
    """
    G = nx.erdos_renyi_graph(n, p)

    non_edges = [(u, v) for u in G for v in G if u < v and not G.has_edge(u, v)]
    assume(len(non_edges) > 0)

    u, v = non_edges[0]

    before = nx.core_number(G)

    G.add_edge(u, v)
    after = nx.core_number(G)

    for node in before:
        assert after[node] >= before[node]


# ---------------------------------------------------------------------------
# Test 7 — Kcore Idempotence
# ---------------------------------------------------------------------------

@given(gk=graph_with_valid_k())
@settings(max_examples=150, deadline=None)
def test_kcore_idempotence(gk):
    """
    Property: k-core operation is idempotent.

    Mathematical basis:
        Once nodes satisfy degree ≥ k, reapplying does not change result.

    Test strategy:
        Compare k_core(G, k) and k_core(k_core(G, k), k).

    Why this matters:
        Ensures algorithm reaches a fixed point.
    """
    G, k = gk
    first = nx.k_core(G, k)
    second = nx.k_core(first, k)

    assert set(first.nodes()) == set(second.nodes())


# ---------------------------------------------------------------------------
# Test 8 — Core number Isomorphism Invariance
# ---------------------------------------------------------------------------

@given(G=diverse_graphs(min_nodes=4))
@settings(max_examples=150, deadline=None)
def test_core_number_isomorphism_invariance(G):
    """
    Property: Core numbers are invariant under graph isomorphism (node relabeling).

    Mathematical basis:
        The core number of a node depends only on the structure of the graph
        (i.e., adjacency relationships), not on the specific labels assigned
        to nodes. Graph isomorphisms preserve adjacency structure, meaning
        that any relabeling of nodes should leave all structural properties
        unchanged.

        Therefore, if H is an isomorphic copy of G obtained by relabeling
        nodes, then:
            core_number_G(v) = core_number_H(f(v))
        where f is the relabeling function.

    Test strategy:
        We construct a relabeled version H of G by applying a deterministic
        node mapping. We compute core_number for both graphs and then map
        the results from H back to G’s labeling. The two dictionaries must
        match exactly.

    Assumptions / preconditions:
        - G is a simple undirected graph.
        - The relabeling function is bijective (one-to-one mapping).

    Why a failure matters:
        If core numbers change under relabeling, the implementation is
        incorrectly depending on node identifiers rather than graph
        structure. This would violate a fundamental principle of graph
        algorithms and make results unreliable across equivalent graphs.
    """
    mapping = {node: i for i, node in enumerate(reversed(list(G.nodes())))}
    H = nx.relabel_nodes(G, mapping)

    core_G = nx.core_number(G)
    core_H = nx.core_number(H)

    remapped = {node: core_H[mapping[node]] for node in G.nodes()}

    assert core_G == remapped

# ---------------------------------------------------------------------------
# Test 9 — Empty graph boundary
# ---------------------------------------------------------------------------

def test_empty_graph_boundary():
    """
    Property: The core_number of an empty graph is an empty dictionary.

    Mathematical basis:
        An empty graph has no vertices and no edges. Since core numbers
        are defined per node, and there are no nodes in the graph, the
        result must be an empty mapping.

        This is a boundary condition that ensures the algorithm handles
        degenerate inputs correctly without errors.

    Test strategy:
        We construct an empty graph and verify that core_number returns
        an empty dictionary.

    Assumptions / preconditions:
        - The graph contains zero nodes.

    Why a failure matters:
        If the algorithm fails on an empty graph or returns a non-empty
        result, it indicates improper handling of edge cases. Robust graph
        algorithms must gracefully handle such degenerate inputs.
    """
    assert nx.core_number(nx.Graph()) == {}

# ---------------------------------------------------------------------------
# Test 10 — Single node graph
# ---------------------------------------------------------------------------

def test_single_node_graph():
    """
    Property: A single-node graph has core number 0.

    Mathematical basis:
        A node belongs to the k-core if it has degree at least k. In a
        graph with a single node and no edges, the degree of the node is 0.
        Therefore:
            - The node belongs to the 0-core
            - The node does not belong to any k-core for k ≥ 1

        Hence, its core number must be exactly 0.

    Test strategy:
        We construct a graph with one node and verify that its core number
        is computed as 0.

    Assumptions / preconditions:
        - The graph contains exactly one node and no edges.

    Why a failure matters:
        If the core number is computed incorrectly for this simplest case,
        it suggests a fundamental flaw in how degree constraints are being
        interpreted or applied in the algorithm.
    """
    G = nx.Graph()
    G.add_node(1)

    assert nx.core_number(G)[1] == 0

# ---------------------------------------------------------------------------
# Test 11 — Complete graph core number
# ---------------------------------------------------------------------------

@given(G=diverse_graphs(min_nodes=3))
@settings(max_examples=100, deadline=None)
def test_complete_graph_core_number(G):
    """
    Property: In a complete graph with n nodes, every node has core number n-1.

    Mathematical basis:
        In a complete graph K_n, every node is connected to every other node,
        so each node has degree (n - 1). Since all nodes satisfy the degree
        requirement for k = n - 1, the entire graph forms the (n - 1)-core.

        Furthermore, no node can belong to an n-core because the maximum
        possible degree is (n - 1). Therefore:
            core_number(v) = n - 1  for all nodes v in K_n

    Test strategy:
        For a given size n, we construct the complete graph K_n and compute
        core numbers. We then verify that every node has core number n - 1.

    Assumptions / preconditions:
        - n ≥ 3 (to avoid trivial graphs).
        - The graph is fully connected (complete graph).

    Why a failure matters:
        If nodes in a complete graph do not achieve the maximum possible
        core number, it indicates that the algorithm is incorrectly
        computing degree-based pruning, which is central to core
        decomposition.
    """
    n = len(G.nodes())
    H = nx.complete_graph(n)

    core_nums = nx.core_number(H)

    for node in core_nums:
        assert core_nums[node] == n - 1
