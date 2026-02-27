from ortools.sat.python import cp_model
import numpy as np
class MatriceSolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Merge all the solutions found by the solver in a list of matrices.
    Each matrix is a solution that satisfies the constraints of Strength and Degree.
    """
    def __init__(self, vars_matrix, rows, cols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__vars_matrix = vars_matrix
        self.__rows = rows
        self.__cols = cols
        self.__solutions = []

    def on_solution_callback(self):
        current_solution = []
        for r in range(self.__rows):
            row_vals = []
            for c in range(self.__cols):
                val = self.Value(self.__vars_matrix[r][c])
                row_vals.append(val)
            current_solution.append(row_vals)
        self.__solutions.append(current_solution)

    def get_solutions(self):
        return self.__solutions

def find_configurations(
    strength_row,
    strength_colunms,
    degree_row,
    degree_columns
):
    """
    Fixing the degree and the strenght of each node find all the possible bi-adjacency matrices that satisfy these constraints.
    The number of possible configuration increase very rapdly with the size of the graph. I suggest to use this function with small graphs (less than 16 nodes).

        args:
            strength_row (List[int]): A list of integers representing the strength of each row node.
            strength_colunms (List[int]): A list of integers representing the strength of each column node.
            degree_row (List[int]): A list of integers representing the degree of each row node.
            degree_columns (List[int]): A list of integers representing the degree of each column node.
        returns:
            List[List[List[int]]]: A list of bi-adjacency matrices that satisfy the given strength and degree constraints.

    """
    n = len(strength_row)
    m = len(strength_colunms)

    # --- 1. Controlli di consistenza di base ---
    if sum(strength_row) != sum(strength_colunms):
        print("Errore: La somma totale delle Strength righe != colonna.")
        return []
    if sum(degree_row) != sum(degree_columns):
        print("Errore: La somma totale dei Degree righe != colonna.")
        return []

    model = cp_model.CpModel()

    matrix_vars = []
    is_nonzero_vars = []

    #Defining for each entry.
    #The value of an entry can't be greater then the strenght of that node
    max_global_val = max(max(strength_row), max(strength_colunms))

    for r in range(n):
        row_x = []
        row_b = []
        for c in range(m):
            
            limit_val = min(strength_row[r], strength_colunms[c])
            x_val = model.NewIntVar(0, limit_val, f'x_{r}_{c}')
            
        
            b_val = model.NewBoolVar(f'b_{r}_{c}')

 
            model.Add(x_val > 0).OnlyEnforceIf(b_val)
            model.Add(x_val == 0).OnlyEnforceIf(b_val.Not())

            row_x.append(x_val)
            row_b.append(b_val)
        matrix_vars.append(row_x)
        is_nonzero_vars.append(row_b)


    for r in range(n):
   
        model.Add(sum(matrix_vars[r]) == strength_row[r])
   
        model.Add(sum(is_nonzero_vars[r]) == degree_row[r])

    # --- 5. Vincoli sulle Colonne ---
    # Trasponiamo le matrici di variabili per iterare sulle colonne
    col_vars = [list(i) for i in zip(*matrix_vars)]
    col_bools = [list(i) for i in zip(*is_nonzero_vars)]

    for c in range(m):
        # Vincolo Strength colonna
        model.Add(sum(col_vars[c]) == strength_colunms[c])
        # Vincolo Degree colonna
        model.Add(sum(col_bools[c]) == degree_columns[c])

    # --- 6. Risoluzione ---
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    
    solution_collector = MatriceSolutionCollector(matrix_vars, n, m)
    status = solver.Solve(model, solution_collector)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return solution_collector.get_solutions()
    else:
        return []