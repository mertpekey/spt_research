import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.utils import subgraph

def build_edge_index(coords, k=5):
    """
    Build edge_index for PyTorch Geometric using k-nearest neighbors.
    
    Args:
        coords (np.ndarray): Array of shape (N, 2) with (x, y) coordinates.
        k (int): Number of nearest neighbors to connect.
    
    Returns:
        torch.LongTensor: edge_index tensor of shape (2, num_edges).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coords)
    # Note: k+1 because the point itself is included in the neighbors.
    distances, indices = nbrs.kneighbors(coords)
    
    # Build edge list: exclude self-loops (first neighbor is the node itself)
    edge_index_list = []
    num_nodes = coords.shape[0]
    for i in range(num_nodes):
        for j in indices[i][1:]:  # skip the first one (self)
            # Create a directed edge from i to j.
            edge_index_list.append([i, j])
    
    # Convert to tensor and transpose to shape (2, num_edges)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return edge_index

# Example usage:
# Suppose coords is your numpy array of shape (N, 2)
# coords = np.array([[x1, y1], [x2, y2], ...])
# edge_index = build_edge_index(coords, k=5)


def extract_subgraph(global_edge_index, batch_indices, num_nodes):
    """
    Extract the subgraph for a mini-batch given the global edge_index and batch indices.
    
    Args:
        global_edge_index (torch.LongTensor): Edge index for the full dataset.
        batch_indices (torch.Tensor): 1D tensor of node indices in the mini-batch.
        num_nodes (int): Total number of nodes in the full graph.
    
    Returns:
        sub_edge_index (torch.LongTensor): Edge index for the mini-batch with re-labeled nodes.
    """
    sub_edge_index, mapping = subgraph(
        batch_indices, global_edge_index, relabel_nodes=True, num_nodes=num_nodes
    )
    return sub_edge_index
