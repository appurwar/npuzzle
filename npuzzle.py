"""
Using BFS, DFS, Greedy Best Search and A* Algorithms to solve the n-puzzle problem
@author: Apoorv Purwar (ap3644@columbia.edu)
"""

import time

def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped.
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]

    new_state = []
    for row in range(len(state)):
        new_row = []
        for column in range(len(state[row])):
            if row == i1 and column == j1:
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else:
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)

def getIndexOfZero(state):

    for (i,row) in enumerate(state):
        for (j,item) in enumerate(row):
            if item == 0:
                return i, j

def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions.
    The result should be a list containing (Action, state) tuples.
    For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))),
                 ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))]
    """

    child_states = {}

    #Call to get the index of 0th element
    row, column = getIndexOfZero(state)

    #print("Index of Zero - %s,%s" %(row,column))

    #Case 1 - Swap towards left
    if column != len(state)-1:
        left = {'Left':swap_cells(state, row, column, row, column+1)}
        child_states.update(left)
    #Case 2 - Swap towards Right
    if column != 0:
        right = {'Right':swap_cells(state, row, column, row, column-1)}
        child_states.update(right)

    #Case 3 - Swap Up
    if row != len(state[0])-1:
        up = {'Up':swap_cells(state, row, column, row+1, column)}
        child_states.update(up)
    #Case 4 - Swap Down
    if row != 0:
        down = {'Down':swap_cells(state, row, column, row-1, column)}
        child_states.update(down)

    return child_states

def form_goal_state(state):
    goal_state = []
    val = 0

    for x in range(len(state)):
        row = []
        for y in range(len(state[0])):
            row.append(val)
            val+=1
        goal_state.append(row)

    return goal_state

def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise.
    """

    #Initialization of goal_state variable with the values for the final state to be achieved
    goal_state = form_goal_state(state)

    #goal_state = [[val+=1 for y in range(len(state))] for x in range(len(state))]
   # print('Goal State - ' + str(goal_state))
    #print('Received State - ' + str(state))
    if(state == tuple(tuple(x) for x in goal_state)):
        return True
    else:
        return False

def find_path(goal_node, parents, actions):
    sol_sequence = []
    cur_node = goal_node
    while cur_node in parents.keys():
         sol_sequence.append(actions.get(cur_node))
         cur_node = parents.get(cur_node)
         #sol_sequence.reverse()

    return sol_sequence


def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0

    frontier = [state]
    max_frontier += 1
    explored = set()
    seen = set()
    seen.add(state)

    while len(frontier) != 0:
        current_node = frontier.pop(0)
        explored.add(current_node)
        states_expanded += 1

        if(goal_test(current_node) is True):
            solution = find_path(current_node, parents, actions)
            return solution, states_expanded, max_frontier

        successors = get_successors(current_node)
        for key, value in successors.items():
            if value not in explored and value not in seen:
                frontier.append(value)
                seen.add(value)
                parents[value] = current_node
                actions[value] = key

        if len(frontier) > max_frontier:
             max_frontier = len(frontier)

    return None, states_expanded, max_frontier # No solution found


def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """

    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0

    frontier = [state]
    max_frontier += 1
    explored = set()
    seen = set()
    seen.add(state)

    while len(frontier) != 0:
        current_node = frontier.pop()
        explored.add(current_node)
        states_expanded += 1

        if(goal_test(current_node) is True):
            solution = find_path(current_node, parents, actions)
            return solution, states_expanded, max_frontier

        successors = get_successors(current_node)
        for key, value in successors.items():
            if value not in explored and value not in seen:
                frontier.append(value)
                seen.add(value)
                parents[value] = current_node
                actions[value] = key

        if len(frontier) > max_frontier:
            max_frontier = len(frontier)

    return None, states_expanded, max_frontier # No solution found


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """
    goal_state = list(form_goal_state(state))
    state_list = list(list(x) for x in state)
    misplaced_tiles = 0
    for state_item, goal_item in zip(state_list,goal_state):
         for state_row_item, goal_row_item in zip(state_item,goal_item):
             if state_row_item != 0:
                 if state_row_item != goal_row_item:
                     misplaced_tiles += 1

    return misplaced_tiles # replace this

def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the manhattan distance between the current
    position and the goal position. THen sum all distances.
    """

    return 0 # replace this


def best_first(state, heuristic = misplaced_heuristic):
    """
    Breadth first search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """

    # You might want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0

    states_expanded = 0
    max_frontier = 0
    frontier = []

    heappush(frontier, (heuristic(state),state))
    states_expanded += 1
    max_frontier += 1
    explored = set()

    while len(frontier) != 0:
        current_node = heappop(frontier)
        explored.add(current_node[1])
        states_expanded += 1

        if(goal_test(current_node[1]) is True):
            solution = find_path(current_node[1], parents, actions)
            return solution, states_expanded, max_frontier

        successors = get_successors(current_node[1])
        for key, value in successors.items():
            f = heuristic(value)
            if value not in explored and (f,value) not in frontier:
                f = heuristic(value)
                costs[value] = f
                heappush(frontier, (heuristic(value),value))
                parents[value] = current_node[1]
                actions[value] = key

        if len(frontier) > max_frontier:
             max_frontier = len(frontier)

    return None, 0, 0


def astar(state, heuristic = misplaced_heuristic):
    """
    A-star search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    # You might want to use these functions to maintain a priority queue

    from heapq import heappush
    from heapq import heappop
    from heapq import heapify

    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0

    states_expanded = 0
    max_frontier = 0

    frontier = []

    heappush(frontier, (heuristic(state),state))
    states_expanded += 1
    max_frontier += 1
    explored = set()

    while len(frontier) != 0:
        current_node = heappop(frontier)
        explored.add(current_node[1])
        states_expanded += 1

        if(goal_test(current_node[1]) is True):
            solution = find_path(current_node[1], parents, actions)
            return solution, states_expanded, max_frontier

        successors = get_successors(current_node[1])
        for key, value in successors.items():
            f = heuristic(value)
            cost = costs[current_node[1]] + 1
            if value not in costs:
                costs[value] = cost
                heappush(frontier, (cost+heuristic(value),value))
                parents[value] = current_node[1]
                actions[value] = key
                #If we comment line 325-333 we get the values of Expanded states and frontier, which were shared on Piazza by Professor, but I feell the following is the right way to do it.
            elif cost < costs[value]:
                f_val = costs[value] + f
                node = frontier.index((f_val, value))
                frontier.remove(frontier[node])
                heapify(frontier)
                costs[value] = cost
                heappush(frontier, (cost+heuristic(value),value))
                parents[value] = current_node[1]
                actions[value] = key

        if len(frontier) > max_frontier:
             max_frontier = len(frontier)

    return None, states_expanded, max_frontier # No solution found


def print_result(solution, states_expanded, max_frontier):
    """
    Helper function to format test output.
    """
    if solution is None:
        print("No solution found.")
    else:
        print("Solution has {} actions.".format(len(solution)))
    print("Total states exppanded: {}.".format(states_expanded))
    print("Max frontier size: {}.".format(max_frontier))


if __name__ == "__main__":

    #Easy test case
    test_state = ((1, 4, 2),
                  (0, 5, 8),
                  (3, 6, 7))

    #More difficult test case
    #test_state = ((7, 2, 4),
     #            (5, 0, 6),
      #          (8, 3, 1))

    print(state_to_string(test_state))
    print()

    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_frontier = bfs(test_state) #
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    if solution is not None:
        solution.reverse()
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    #Test statement for goal_test function
    print("Is goal State - %s" % goal_test(test_state))

    #Call to successor state function
    print("Possible moves - %s" % get_successors(test_state))


    print()
    print("====DFS====")
    start = time.time()
    solution, states_expanded, max_frontier = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    #print(misplaced_heuristic(test_state))

    print()
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====A* (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    #print()
    #print("====A* (Total Manhattan Distance Heuristic)====")
    #start = time.time()
    #solution, states_expanded, max_frontier = astar(test_state, manhattan_heuristic)
    #end = time.time()
    #print_result(solution, states_expanded, max_frontier)
    #print("Total time: {0:.3f}s".format(end-start))
