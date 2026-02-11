"""
Bipartite network construction and analysis.
"""

def build_bipartite_graph(beta_dict, threshold=0.0):
    """
    Construct bipartite graph from Lasso coefficients.
    Edge (commodity_feature -> stock) exists if |beta| > threshold.
    """
    raise NotImplementedError


def compute_network_statistics(graph):
    """Compute degree distribution, centrality, etc."""
    raise NotImplementedError


def visualize_network(graph, output_path=None):
    """Visualize the bipartite commodity-stock network."""
    raise NotImplementedError
