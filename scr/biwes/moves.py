import numpy as np
import numba.types as nt
from numba.typed import Dict, List
from numba import njit, prange, get_num_threads, get_thread_id
from .config import conditional_njit, node_type, neighbors_type, list_node_type, edge_list_dict_type, edge_list_ty, edge_type, rect_type, list_dict_rect
from .data_structures import update_adj_list, update_weight_dict



@conditional_njit
def rectangle_random_search(A2, adj_list, adj_list2):

    """
    Randomly search for a rectangle in the bipartite graph, using a path sampling algorithm.
        args:
            A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
            adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
            adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
        returns:
            rect (tuple): a tuple of the form (u1, u2, v1, v2) representing the nodes of the rectangle found, where u1 and u2 are the
    """
    
    na = A2.shape[0]
    nb = A2.shape[1]
    #random select a node finding its neighbors
    nodo = np.random.randint(0, na+nb)
    vicini = adj_list2[nodo]
    #if the node has less than 2 neighbors it can't be part of a rectangle so the searching algorithm ends.
    if len(vicini) < 2:
        return (-1,-1,-1,-1)
    else:
        #random select 2 node amog the neighbors of "nodo" and take their neighbors.
        r1 = np.random.randint(0, len(vicini))
        r2 = np.random.randint(0, len(vicini))
        while r1 == r2:   
            r2 = np.random.randint(0, len(vicini))
        nodo2 = vicini[r2]
        nodo1 = vicini[r1]
        bol = np.random.rand()
        #randomly select one of the two neighbors and find its neighbors and call it "nodo3"
        if bol < 0.5:
            vicini_lista  = adj_list2[nodo1]
            vicini_comparison = adj_list[nodo2]
        else:
            vicini_lista  = adj_list2[nodo2]
            vicini_comparison = adj_list[nodo1]
        r3 = np.random.randint(0, len(vicini_lista))
        nodo3 = vicini_lista[r3]
        #if nodo3 is a neighbor of the remaining neighbor, the rectangle is found.
        if nodo3 in vicini_comparison and nodo3 != nodo:
            if nodo < na:
                u1 = nodo
                u2 = nodo3
                v1 = nodo1
                v2 = nodo2
            else:
                u1 = nodo1
                u2 = nodo2
                v1 = nodo
                v2 = nodo3
            #adjusting the order of nodes to node to avoid bugs.
            if u1 < u2 and v1 < v2:
                rect = (u1, u2, v1, v2)
            elif u1 < u2 and v1 > v2:
                rect = (u1, u2, v2, v1)
            elif u1 > u2 and v1 < v2:
                rect = (u2, u1, v1, v2)
            else:
                rect = (u2, u1, v2, v1)
            rect = (u1, u2, v1, v2)
            return rect
        else:
            return (-1,-1,-1,-1)
        


@conditional_njit
def weight_shuffling(A2, weight_dict, weight_dict2, unique_weights, unique_weights2, adj_list, adj_list2):
    """
    Changes the weights of 4 edge od a bipartite graph presreving the topology.
    If a rectagle is found and the link's weights allow the move a delta is selected and the move is always performed.
    Otherwise the move is not performed and BiWES will return the same graph.
    Args:
        A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): A dictionary mapping each weight to a dictionary of edges with that weight and their corresponding indices in the edge list.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
        unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
        unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique_weights dictionary.
        adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
        adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
        

    Returns:
        np.ndarray A2: The restructured adjacency matrix of the bipartite graph
        Dict[int, Dict[Tuple[int, int], int]]: The updated weight dictionary
        Dict[int, List[Tuple[int, int]]]: The updated weight dictionary 2
        Dict[int, int]: The updated unique weights dictionary
        List[int]: The updated unique weights list
    
    Note: There is no need to update the adjacency lists because the topology of the graph is not changed, only the weights of some edges are changed.
    """


    na = A2.shape[0]

    #using 
    rectangle = rectangle_random_search(A2, adj_list, adj_list2)

    if rectangle[0] != -1:
        u1 = rectangle[0]
        u2 = rectangle[1]
        v1 = rectangle[2]
        v2 = rectangle[3]
        w11 = A2[u1, v1-na]
        w12 = A2[u1, v2-na]
        w21 = A2[u2, v1-na]
        w22 = A2[u2, v2-na]
        old_weight = np.array([w11, w22, w12, w21], dtype=np.int64)
        vet_link = [(u1, v1), (u2, v2), (u1, v2), (u2, v1)]
        vet1 = np.array([w11, w22])
        vet2 = np.array([w12, w21])
    
        

        #mi trovo tutti i possibili delta e ne scelgo uno a caso. Questo equivale a pescare dal campione dei rettangoli con tutti i possibili pesi
        #print(rectangle)
        min_aa = np.min(vet1)-1
        min_ab = (-1)*(np.min(vet2)-1)
        
        delta = np.random.randint(min_ab, min_aa+1)
        
        A2[u1, v1-na] = w11 - delta
        A2[u2, v2-na] = w22 - delta
        A2[u1, v2-na] = w12 + delta
        A2[u2, v1-na] = w21 + delta

        new_weight = np.array([A2[u1, v1-na], A2[u2, v2-na], A2[u1, v2-na], A2[u2, v1-na]], dtype=np.int64)
        weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, vet_link, vet_link)
        

    return A2, weight_dict, weight_dict2, unique_weights, unique_weights2


@conditional_njit
def simple_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):

    """
    Changes the topology of the graph by swapping two edges of the same weight.
    If two edges of the same weight are found and the swap is possible, the move is performed. Otherwise the move is not performed and BiWES will return the same graph.
    Args:
        A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
        adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
        adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): A dictionary mapping each weight to a dictionary of edges with that weight and their corresponding indices in the edge list.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
        unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
        unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique_weights dictionary.
    Returns:
        all the data structures updated after the move. If the move is not performed, the same data structures are returned.
    """

    #random select a weight among the unique weights in current graph
    na = A2.shape[0]
    unique_weights2_len = len(unique_weights2)
    r = np.random.randint(0, unique_weights2_len)
    weight = unique_weights2[r]
    edges = weight_dict2[weight]
    k = len(edges)
    if k < 2:
        return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2
    #randomply select two edges with the same weight
    i = np.random.randint(0, k)
    j = np.random.randint(0, k)
    if i != j:
        edge1 = edges[i]
        edge2 = edges[j]
        u1 = edge1[0]
        u2 = edge2[0]
        v1 = edge1[1]
        v2 = edge2[1]
        #check if the swap is possible avoiding to create multiple link
        if u1 != u2 and v1 != v2 and A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0:
            #perform the swap and update the data structures
            A2[u1, v1-na] = 0
            A2[u2, v2-na] = 0
            A2[u1, v2-na] = weight
            A2[u2, v1-na] = weight
            old_weight = np.array([weight, weight], dtype=np.int64)
            old_links = [(u1, v1), (u2, v2)]
            new_weight = np.array([weight, weight], dtype=np.int64)
            new_links = [(u1, v2), (u2, v1)]

            adj_list, adj_list2 = update_adj_list(adj_list, adj_list2, old_links, new_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weight, new_weight, old_links, new_links)

    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2

@conditional_njit
def research_five_nodes_structure(A2, adj_list2):
    """
    Research through a path sampling algorithm a random chain of 5 node in a graph
        args:
            A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
            adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
        returns:
            structure (List[int]): a list of 5 nodes representing the chain found. If the chain is not found, the list will contain less than 5 nodes.
    """

    na = A2.shape[0]
    nb = A2.shape[1]
    n = na+ nb
    structure = List.empty_list(node_type)
    old_node = -1
    for i in range(5):
        if i == 0:
            node = np.random.randint(0, n)
            vicini = adj_list2[node]
        else:
            k = len(vicini)
            if k == 0:
                break
            if k < 2 and i != 1:
                break
            elif k == 2:
                if i != 1:
                    if vicini[0] in structure and vicini[1] in structure:
                        break
                    else:
                        if vicini[0] not in structure:
                            node = vicini[0]
                        else:
                            node = vicini[1]
                    if i == 1:
                        r = np.random.randint(0, 2)
                        node = vicini[r]
            else:
                while True:
                    r = np.random.randint(0, k)
                    node2 = vicini[r]
                    if node2 != old_node:
                        node = node2
                        break
                
        if node in structure:
            break
        else:
            vicini = adj_list2[node]
            structure.append(node)
            old_node = node
    #adjusting the order of nodes to node to avoid bugs.
    if len(structure) == 5:
        if structure[0] < na:
            temp = structure[1]
            structure[1] = structure[4]
            structure[4] = temp
            temp = structure[2]
            structure[2] = structure[4]
            structure[4] = temp
        else:
            temp = structure[0]
            structure[0] = structure[1]
            structure[1] = temp
            temp = structure[3]
            structure[3] = structure[4]
            structure[4] = temp
            temp = structure[1]
            structure[1] = structure[4]
            structure[4] = temp
            temp = structure[2]
            structure[2] = structure[4]
            structure[4] = temp

    return structure


@conditional_njit
def check_five_nodes_structure(A2, structure):
    """
    Check if the structure the five nodes structure found is suitable for bridge edge swap
            args:
                A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
                structure (List[int]): a list of 5 nodes representing the chain found. If the chain is not found, the list will contain less than 5 nodes.
            returns:
                check (bool): True if the structure is suitable for bridge edge swap, False otherwise.
    """

    na = A2.shape[0]
    check = False
    if structure[4] < na:
        u1 = structure[0]
        u2 = structure[1]
        v1 = structure[2]
        v2 = structure[3]
        u3 = structure[4]
        if A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0 and A2[u1, v1-na] != A2[u2, v2-na]:
            arr = np.array([A2[u1, v1-na], A2[u2, v2-na]])
            peso_max = np.max(arr)
            if A2[u1, v1-na] == peso_max:
                if A2[u3, v2-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
            if A2[u2, v2-na] == peso_max:
                if A2[u3, v1-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
    if structure[4] >= na:
        u1 = structure[0]
        u2 = structure[1]
        v1 = structure[2]
        v2 = structure[3]
        v3 = structure[4]
        if A2[u1, v2-na] == 0 and A2[u2, v1-na] == 0 and A2[u1, v1-na] != A2[u2, v2-na]:
            arr = np.array([A2[u1, v1-na], A2[u2, v2-na]])
            peso_max = np.max(arr)
            if A2[u1, v1-na] == peso_max:
                if A2[u2, v3-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
            if A2[u2, v2-na] == peso_max:
                if A2[u1, v3-na] > np.abs(A2[u2, v2-na] - A2[u1, v1-na]):
                    check = True
    return check

@conditional_njit
def bridge_edge_swap(A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2):

    """
    Changes the topology and the edge weight distribution of the graph, swapping two edge with differet weights.
    If a suitable structure is found and the swap is possible, the move is performed. Otherwise the move is not performed and BiWES will return the same graph.
    Args:
        A2 (np.ndarray): the bi-adjacency matrix of the bipartite graph.
        adj_list (Dict[int, Dict[int, int]]): A dictionary mapping each node to a dictionary of its neighbors and their corresponding indices in the neighbor list.
        adj_list2 (Dict[int, List[int]]): A dictionary mapping each node to a list of its neighbors.
        weight_dict (Dict[int, Dict[Tuple[int, int], int]]): A dictionary mapping each weight to a dictionary of edges with that weight and their corresponding indices in the edge list.
        weight_dict2 (Dict[int, List[Tuple[int, int]]]): A dictionary mapping each weight to a list of edges with that weight.
        unique_weights (Dict[int, int]): A dictionary mapping each weight to a unique index.
        unique_weights2 (List[int]): A list of unique weights. The index of each weight in the list corresponds to its unique index in the unique_weights dictionary.
    Returns:
        all the data structures updated after the move. If the move is not performed, the original data structures are returned.
    """

    na = A2.shape[0]
    
    structure = research_five_nodes_structure(A2, adj_list2)
    
    if len(structure) == 5:

        if check_five_nodes_structure(A2, structure):
            

            na = A2.shape[0]
            u1 = structure[0]
            u2 = structure[1]
            v1 = structure[2]
            v2 = structure[3]
            w11 = A2[u1, v1-na]
            w22 = A2[u2, v2-na]
            delta = np.abs(w11 - w22)
            A2[u1, v1-na] = 0
            A2[u2, v2 - na] = 0

            if structure[4] < na:
                u3 = structure[4]
                old_weights = np.array([w11, w22, A2[u3, v1 - na], A2[u3, v2 - na]], dtype=np.int64)
                A2[u1, v2 - na] = w11
                A2[u2, v1 - na] = w22
                if w11 > w22:
                    A2[u3, v1 - na] = A2[u3, v1 - na] + delta
                    A2[u3, v2 - na] = A2[u3, v2 - na] - delta
                else:
                    A2[u3, v1 - na] = A2[u3, v1 - na] - delta
                    A2[u3, v2 - na] = A2[u3, v2 - na] + delta
                old_links = [(u1, v1), (u2, v2), (u3, v1), (u3, v2)]
                new_links = [(u1, v2), (u2, v1), (u3, v1), (u3, v2)]
                new_weights = np.array([w11, w22, A2[u3, v1 - na], A2[u3, v2 - na]], dtype=np.int64)
                
                    
            if structure[4] >= na:
                v3 = structure[4]
                old_weights = np.array([w11, w22, A2[u1, v3 - na], A2[u2, v3 - na]], dtype=np.int64)
                A2[u1, v2 - na] = w22
                A2[u2, v1 - na] = w11
                if w11 > w22:
                    A2[u1, v3 - na] = A2[u1, v3 - na] + delta
                    A2[u2, v3 - na] = A2[u2, v3 - na] - delta
                else:
                    A2[u1, v3 - na] = A2[u1, v3 - na] - delta
                    A2[u2, v3 - na] = A2[u2, v3 - na] + delta
                old_links = [(u1, v1), (u2, v2), (u1, v3), (u2, v3)]
                new_links = [(u1, v2), (u2, v1), (u1, v3), (u2, v3)]
                new_weights = np.array([w22, w11, A2[u1, v3 - na], A2[u2, v3 - na]], dtype=np.int64)
                

            #update data structures
            broken_links = [(u1, v1), (u2, v2)]
            created_links = [(u1, v2), (u2, v1)]
            adj_list, adj_list2 = update_adj_list(adj_list, adj_list2, broken_links, created_links)
            weight_dict, weight_dict2, unique_weights, unique_weights2 = update_weight_dict(weight_dict, weight_dict2, unique_weights, unique_weights2, old_weights, new_weights, old_links, new_links)
    return A2, adj_list, adj_list2, weight_dict, weight_dict2, unique_weights, unique_weights2


