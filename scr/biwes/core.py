import numpy as np
import numba.types as nt
from numba.typed import Dict, List
from numba import njit, prange, get_num_threads, get_thread_id
from .config import conditional_njit, node_type, neighbors_type, list_node_type, edge_list_dict_type, edge_list_ty, edge_type, rect_type, list_dict_rect
from .moves import weight_shuffling, bridge_edge_swap, simple_edge_swap
from .data_structures import from_adj_matrix_to_adj_list, get_weight_dict


@conditional_njit
def pre_compute(A):
    """
    Pre-compute data structures for the BiWES algorithm.
        args:
            A (np.ndarray): The bi-adjacency matrix of the bipartite graph.
        returns:
            A2 (np.ndarray): A copy of the input bi-adjacency matrix.
            weight_dict (Dict[int, Dict[Tuple[int, int], int]]): The comlementary dictionary of weight_dict2.
            weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
            adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
            adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
            unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
            unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique
    """
    A2 = A.copy()
    adj_list, adj_list2 = from_adj_matrix_to_adj_list(A2)
    weight_dict, weight_dict2, unique_weights, unique_weights2 = get_weight_dict(A2)
    return A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2

@conditional_njit
def BiWES(A, n_iterations):
    """
    Perform the BiWES algoritihm to randomize a bipartite graph, preserving the degree sequence and the weight sequence.
    This algorithm precompute all the necessary data structures. So if you need to perform multiple randomizations on the same graph, it is more efficient to call pre_compute once and then call the moves directly.
        args:
            A (np.ndarray): The bi-adjacency matrix of the bipartite graph.
            n_iterations (int): The number of randomization moves to perform.
        returns:
            A2 (np.ndarray): The randomized bi-adjacency matrix after performing the BiWES algorithm.
    """
    #pre compute all the necessary data structures for the BiWES algorithm
    A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute(A)
    p = 1/3
    prob = np.array([p, p, p])
    #each move is selected with equal probability.
    prob = np.cumsum(prob)
    for it in range(n_iterations):
        r = np.random.rand()
        if r < prob[0]:
            A2, weight_dict, weight_dict2, unique_weights, unique_weights2 = weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2)
        elif prob[0] <= r < prob[1]:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
        else:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = bridge_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
    return A2



@conditional_njit
def BiWES2(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations):
    """
    Perform the BiWES algoritihm to randomize a bipartite graph, preserving the degree sequence and the weight sequence.
    In this case the necessary data structures are pre-computed and passed as arguments. This is useful if you need to perform multiple randomizations on the same graph, as it avoids the overhead of pre-computing the data structures multiple times.
        args:
            A2 (np.ndarray): The bi-adjacency matrix of the bipartite graph
            weight_dict (Dict[int, Dict[Tuple[int, int], int]]): The comlementary dictionary of weight_dict2.
            weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
            adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
            adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
            unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
            unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique_weights dictionary.
            n_iterations (int): The number of randomization moves to perform.
        returns:
            A2 (np.ndarray): The randomized bi-adjacency matrix after performing the BiWES algorithm.
    """
    
    p = 1/3
    prob = np.array([p, p, p])
    prob = np.cumsum(prob)

    for it in range(n_iterations):
        r = np.random.rand()
        if r < prob[0]:
            A2, weight_dict, weight_dict2, unique_weights, unique_weights2 = weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2)
        elif prob[0] <= r < prob[1]:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
        else:
            A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2 = bridge_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2)
    return A2


@njit(parallel=True)
def get_p_value_matrix(A, n_iterations, n_run):

    """
    Perform the BiWES algorithm multiple times to compute the probability of obersing a weight greater than or equal to the one in the original matrix A for each edge.
        args:
            A (np.ndarray): The bi-adjacency matrix of the bipartite graph.
            n_iterations (int): The number of randomization moves to perform in each simulation.
            n_run (int): The number of simulations to run.
            number_of_threads (int, optional): The number of threads to use for parallel execution. If None, it will use all available threads.
        returns:
            p_values (np.ndarray): A matrix of the same shape as A, where each entry (i, j) contains the probability of observing a weight greater than or equal to A[i, j] in the randomized graphs generated by the BiWES algorithm.

    """
    
    na, nb = A.shape
    A_obs = A.copy()
    n_threads = get_num_threads()
    
    # Divide the total number of simulations among the available threads, ensuring that any remainder is distributed among the first few threads.
    sims_per_thread = n_run // n_threads
    remainder = n_run % n_threads
    
    thread_local_counts = np.zeros((n_threads, na, nb), dtype=np.int32)

    for tid in prange(n_threads):
        n_sims = sims_per_thread + (1 if tid < remainder else 0)
        
        #each thread works on its own copy of the adjacency matrix and pre-computed data structures to avoid race conditions
        A_current = A.copy()
        A2, weight_dict, weight_dict2, adj_list, adj_list2, unique_weights, unique_weights2 = pre_compute(A)
        
       
        for sim in range(n_sims):
            
            A_current = BiWES2(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2, n_iterations)
            
            #count how many times each edge has a weight greater than or equal to the one in the original matrix A
            for i in range(na):
                for j in range(nb):
                    if A_current[i, j] >= A_obs[i, j]:
                        thread_local_counts[tid, i, j] += 1
        
    #merge the counts from all threads
    total_counts = np.sum(thread_local_counts, axis=0)

    return total_counts / float(n_run)

