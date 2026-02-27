import numpy as np
import numba.types as nt
from numba.typed import Dict, List
from numba import njit, prange, get_num_threads, get_thread_id
from .config import conditional_njit, node_type, neighbors_type, list_node_type, edge_list_dict_type, edge_list_ty, edge_type, rect_type, list_dict_rect


@conditional_njit
def from_adj_matrix_to_adj_list(A):

    """
    Create two adjacency lists given the bi-adjacency matrix.
    adj_list2 is the true adjacency list a dictionary that maps each node to a list of its neighbors.
    While adj_list is a dictionary that maps each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.

    Args:
        A (np.ndarray): The bi-adjacency matrix of the bipartite graph.
    Returns:
        adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
        adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
    """

    na, nb = A.shape[0], A.shape[1]
    adj_list = Dict.empty(
        key_type=node_type,
        value_type=neighbors_type
    )
    
    adj_list2 = Dict.empty(
        key_type=node_type,
        value_type=list_node_type
    )

    for i in range(na):
        neighbors = Dict.empty(
            key_type=node_type,
            value_type=node_type
        )
        cont = 0
        vicini = List.empty_list(node_type)
        for j in range(nb):
            if A[i, j] != 0:
                neighbors[j + na] = cont
                vicini.append(j + na)
                cont += 1
        adj_list[i] = neighbors
        adj_list2[i] = vicini
    
    for j in range(nb):
        neighbors = Dict.empty(
            key_type=node_type,
            value_type=node_type
        )
        cont = 0
        vicini = List.empty_list(node_type)
        for i in range(na):
            if A[i, j] != 0:
                neighbors[i] = cont
                vicini.append(i)
                cont += 1
        adj_list[j + na] = neighbors
        adj_list2[j + na] = vicini


    return adj_list, adj_list2



@conditional_njit
def get_weight_dict(A):
    """
    Create a dictionary that maps each weight to a dictionary of edges with that weight and their corresponding indices in the edge list.
    Also create a dictionary that maps each weight to a list of edges with that weight.
    And two dictionaries that map each weight to a unique index and a list of unique weights.
        Args:
        A (np.ndarray): The bi-adjacency matrix of the bipartite graph.
    Returns:
    weight_dict (Dict[int, Dict[Tuple[int, int], int]]): The comlementary dictionary of weight_dict2.
    weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
    unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
    unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique
    """

    na, nb= A.shape[0], A.shape[1]
    weight_dict = Dict.empty(
        key_type=node_type,
        value_type= edge_list_dict_type
    )
    weight_dict2 = Dict.empty(
        key_type=node_type,
        value_type= edge_list_ty
    )
    unique_weights = Dict.empty(
        key_type=node_type,
        value_type=node_type
    )
    unique_weights2 = List.empty_list(node_type)
    for i in range(na):
        for j in range(nb):
            if A[i, j] != 0:
                weight = A[i, j]
                edge = (i, j + na)
                if weight not in unique_weights:
                    unique_weights2.append(weight)
                    unique_weights[weight] = len(unique_weights2) - 1
                if weight not in weight_dict:
                    weight_dict[weight] = Dict.empty(
                        key_type=edge_type,
                        value_type=node_type
                    )
                if weight not in weight_dict2:
                    weight_dict2[weight] = List.empty_list(edge_type)
                weight_dict2[weight].append(edge)

                weight_dict[weight][edge] = len(weight_dict2[weight]) - 1

    return weight_dict, weight_dict2, unique_weights, unique_weights2



@conditional_njit
def update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links):
    #remove old weights
    #cicle over old weights to remove them from the weight dictionaries and unique weights if necessary
    for i in range(len(old_weight)):
        weight = old_weight[i]
        edge = old_links[i]
        if weight in weight_dict:
            index = weight_dict[weight][edge]
            del weight_dict[weight][edge]
            last_edge = weight_dict2[weight][-1]
            if edge != last_edge:
                weight_dict2[weight][index] = last_edge
                weight_dict[weight][last_edge] = index
            weight_dict2[weight].pop()
            if len(weight_dict[weight]) == 0:
                del weight_dict[weight]
                del weight_dict2[weight]
                index = unique_weights[weight]
                last_weight = unique_weights2[-1]
                if weight != last_weight:
                    unique_weights2[index] = last_weight
                    unique_weights[last_weight] = index
                unique_weights2.pop()
                del unique_weights[weight]

    #add new weight and links to the weight dictionaries and unique weights if necessary
    for i in range(len(new_weight)):
        weight = new_weight[i]
        edge = new_links[i]
        if weight not in weight_dict:
            weight_dict[weight] = Dict.empty(
                key_type=edge_type,
                value_type=node_type
            )
        if weight not in weight_dict2:
            weight_dict2[weight] = List.empty_list(edge_type)
        weight_dict2[weight].append(edge)
        
        weight_dict[weight][edge] = len(weight_dict2[weight]) - 1
        if weight not in unique_weights:
            unique_weights2.append(weight)
            unique_weights[weight] = len(unique_weights2) - 1
    return weight_dict, weight_dict2, unique_weights, unique_weights2


@conditional_njit
def update_adj_list(adj_list, adj_list2, old_links, new_links):

    """
    Update the adjacency lists when an edge is added or removed.
     1. Remove old links from the adjacency lists.
     2. Add new links to the adjacency lists.
    """
    #remove the old links from the adjacency lists using swap and pop technique.
    for i in range(len(old_links)):
        edge = old_links[i]
        u = edge[0]
        v = edge[1]
        

        index_v = adj_list[u][v]  
        index_u = adj_list[v][u] 
        
        vicini_u = adj_list2[u]
        last_neighbor_u = vicini_u[-1] 
        if index_v < len(vicini_u) - 1: 
            vicini_u[index_v] = last_neighbor_u 
            adj_list[u][last_neighbor_u] = index_v
        vicini_u.pop()  
        adj_list2[u] = vicini_u
        

        vicini_v = adj_list2[v]
        last_neighbor_v = vicini_v[-1]  
        if index_u < len(vicini_v) - 1:  
            vicini_v[index_u] = last_neighbor_v  
            adj_list[v][last_neighbor_v] = index_u
        vicini_v.pop()  
        adj_list2[v] = vicini_v
        
        
        del adj_list[u][v]
        del adj_list[v][u]

    #add new links to the adjacency lists
    for i in range(len(new_links)):
        edge = new_links[i]
        u = edge[0]
        v = edge[1]
        vicini_u = adj_list2[u]
        vicini_v = adj_list2[v]
        vicini_u.append(v)
        vicini_v.append(u)
        adj_list2[u] = vicini_u
        adj_list2[v] = vicini_v
        adj_list[u][v] = len(vicini_u)-1
        adj_list[v][u] = len(vicini_v)-1

    return adj_list, adj_list2





