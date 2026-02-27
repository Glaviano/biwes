from numba import njit
import numba.types as nt


USE_NJIT = True

#GLOBAL CONFIGURATION FOR NUMBA
def conditional_njit(func):
    if USE_NJIT:
        return njit(cache=True)(func)
    else:
        return func

# ==========================================
# DEFINING NUMBA TYPES THAT WILL BE USED THROUGHOUT THE CODE
# ==========================================
node_type = nt.int64
neighbors_type = nt.DictType(node_type, node_type)
edge_type = nt.UniTuple(node_type, 2)
edge_list_dict_type = nt.DictType(edge_type, node_type)
list_node_type = nt.ListType(node_type)
edge_list_ty = nt.ListType(edge_type)
rect_type = nt.UniTuple(node_type, 4)
list_dict_rect = nt.DictType(rect_type, node_type)