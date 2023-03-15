from fringes import Queue, Stack, PriorityQueue

def bfs(problem):
    """
    Breadth-first search algorithm.
    """
    node = problem.initial_state()
    frontier = Queue()
    frontier.enqueue(node)
    explored = set()
    
    while not frontier.is_empty():
        node = frontier.dequeue()
        
        if problem.goal_test(node):
            return node.path()
        
        explored.add(node.state())
        
        for child in node.expand(problem):
            if child.state() not in explored and child not in frontier.items:
                frontier.enqueue(child)
                
    return []

def dfs(problem):
    """
    Depth-first search algorithm.
    """
    node = problem.initial_state()
    frontier = Stack()
    frontier.push(node)
    explored = set()
    
    while not frontier.is_empty():
        node = frontier.pop()
        
        if problem.goal_test(node):
            return node.path()
        
        explored.add(node.state())
        
        for child in node.expand(problem):
            if child.state() not in explored and child not in frontier.items:
                frontier.push(child)
                
    return []

def ucs(problem):
    """
    Uniform-cost search algorithm.
    """
    node = problem.initial_state()
    frontier = PriorityQueue()
    frontier.push(node, 0)
    explored = set()
    
    while not frontier.is_empty():
        node = frontier.pop()
        
        if problem.goal_test(node):
            return node.path()
        
        explored.add(node.state())
        
        for child in node.expand(problem):
            if child.state() not in explored and child not in frontier.items:
                frontier.push(child, child.path_cost())
            elif child in frontier.items:
                current_priority = frontier.items[frontier.items.index(child)][0]
                if child.path_cost() < current_priority:
                    frontier.items.remove(frontier.items[frontier.items.index(child)])
                    frontier.push(child, child.path_cost())
                
    return []

'''
In this implementation, we define three search algorithms:

bfs: Implements the Breadth-First Search algorithm. It uses a queue data structure to keep track of the nodes to be explored, and expands the shallowest unexplored node at each iteration.

dfs: Implements the Depth-First Search algorithm. It uses a stack data structure to keep track of the nodes to be explored, and expands the deepest unexplored node at each iteration.

ucs: Implements the Uniform-Cost Search algorithm. It uses a priority queue data structure to keep track of the nodes to be explored, and expands the node with the lowest path cost at each iteration.

All three algorithms take in an object of SingleFoodSearchProblem as the input, and return a list of actions for pacman to travel from the initial location to the food. 
The actions are represented as strings 'N' (go up), 'S' (go down), 'W' (go to the left), 'E' (go to the right), and 'Stop' (stop). 
An example returned value of bfs() is a list of actions that lead pacman from the initial location to the food, such as ['N', 'N', 'S', 'W', 'E', 'E', 'Stop'].
'''

# YC1-6
# To modify the search functions in searchAgents.py to work for both SingleFoodSearchProblem and MultiFoodSearchProblem, 
# we can check the type of problem object passed as input and call the appropriate functions based on the type.
# Here's an example implementation for bfs function:

'''
def bfs(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    if isinstance(problem, SingleFoodSearchProblem):
        start_node = Node(problem.get_start_state(), None, None, 0)
        frontier = Queue()
        frontier.push(start_node)
        explored = set()

        while not frontier.is_empty():
            node = frontier.pop()

            if problem.is_goal(node.state):
                return node.solution()

            explored.add(node.state)

            for successor, action, step_cost in problem.get_successors(node.state):
                if successor not in explored:
                    child_node = Node(successor, node, action, node.path_cost + step_cost)
                    frontier.push(child_node)

    elif isinstance(problem, MultiFoodSearchProblem):
        start_node = Node(problem.get_start_state(), None, None, 0)
        frontier = Queue()
        frontier.push(start_node)
        explored = set()

        while not frontier.is_empty():
            node = frontier.pop()

            if problem.is_goal(node.state):
                return node.solution()

            explored.add(node.state)

            for successor, action, step_cost in problem.get_successors(node.state):
                if successor not in explored:
                    child_node = Node(successor, node, action, node.path_cost + step_cost)
                    frontier.push(child_node)

    else:
        raise TypeError("Unsupported problem type")

'''
# Similarly, we can modify the dfs and ucs functions to work for both problem types.


# YC2-1

'''
Sure, here are two example heuristic functions that could be used for estimating the cost from the current state to the goal state in SingleFoodSearchProblem:

Heuristic 1: Manhattan distance
The Manhattan distance heuristic calculates the sum of the absolute differences in the x and y coordinates between the current state and the goal state.
It is admissible because it never overestimates the actual cost to reach the goal state, as moving diagonally would be the shortest possible path and the Manhattan distance always overestimates this distance. 
It is also consistent because the Manhattan distance between any two points is always less than or equal to the sum of Manhattan distances from a third point, due to the triangle inequality.
'''
def manhattan_distance(state):
    current_pos = state[0]
    goal_pos = state[1]
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

'''
Heuristic 2: Euclidean distance
The Euclidean distance heuristic calculates the straight-line distance between the current state and the goal state.
It is admissible because it never overestimates the actual cost to reach the goal state, as moving diagonally would be the shortest possible path and the Euclidean distance always underestimates this distance.
However, it is not consistent because it violates the triangle inequality, 
where the Euclidean distance between any two points is sometimes greater than the sum of Euclidean distances from a third point.

'''

def euclidean_distance(state):
    current_pos = state[0]
    goal_pos = state[1]
    return math.sqrt((current_pos[0] - goal_pos[0]) ** 2 + (current_pos[1] - goal_pos[1]) ** 2)

# Note that these are just example heuristics and there may be other heuristics that could be used for this problem.

'''
    # impliment heuristic 
    def a_star_search(problem, heuristic):

        # Calculate the heuristic value for the initial state
        h = heuristic(problem.get_start_state())

        # Add the initial node to the frontier with the total cost
        # equal to the heuristic value (no path cost yet)
        frontier.put((h, (problem.get_start_state(), [])))
'''



# YC2-2
'''
Here is an example of a heuristic function for estimating the cost from the current state to the goal state in MultiFoodSearchProblem. 
This heuristic function calculates the Manhattan distance between the current state and the closest remaining food dot:
'''
def food_heuristic(state):
    current_pos, food = state
    if not food:
        return 0

    # Calculate the Manhattan distance to the closest food dot
    distances = [abs(current_pos[0] - f[0]) + abs(current_pos[1] - f[1]) for f in food]
    min_distance = min(distances)

    # Return the estimated cost as the sum of the Manhattan distance to the closest food dot
    # and the number of remaining food dots minus 1 (because we don't need to count the current dot)
    return min_distance + len(food) - 1

'''
This heuristic function returns a non-negative value and is admissible, meaning that it never overestimates the actual cost to reach the goal state. 
To see why, consider that the actual cost to reach the goal state is at least the Manhattan distance to the closest remaining food dot, 
since we have to visit that dot at some point. Therefore, this heuristic function is admissible because it always returns a value less than or equal to the actual cost.

Note that this heuristic function does not take into account the fact that some food dots may be blocked by obstacles,
which means that it may overestimate the actual cost in some cases. 
However, it can still be effective in practice and can help guide the search towards the goal state more efficiently.
'''


# YC2-3:

# Here is an implementation of the astar function in searchAgents.py that takes a SingleFoodSearchProblem object and a heuristic function as input,
# and returns a list of actions for Pacman to reach the food location:

from fringes import PriorityQueue

def astar(problem, fn_heuristic):
    # Initialize the start node with the initial state
    start_node = Node(problem.get_initial_state())

    # Initialize the fringe with the start node and its estimated cost
    fringe = PriorityQueue()
    fringe.push(start_node, fn_heuristic(start_node.get_state()))

    # Initialize the set of visited states to the start state
    visited = set([start_node.get_state()])

    while not fringe.is_empty():
        # Pop the node with the lowest estimated cost
        current_node = fringe.pop()

        # Check if the current state is the goal state
        if problem.is_goal_state(current_node.get_state()):
            # If so, return the sequence of actions from the start node to the current node
            return current_node.get_actions()

        # Generate the successor nodes and add them to the fringe
        for successor_node in current_node.expand(problem):
            # Check if the successor state has not been visited before
            if successor_node.get_state() not in visited:
                # Add the successor node to the fringe with its estimated cost
                fringe.push(successor_node, current_node.get_cost() + successor_node.get_cost() + fn_heuristic(successor_node.get_state()))

                # Add the successor state to the set of visited states
                visited.add(successor_node.get_state())

    # If the fringe is empty and no solution was found, return an empty list of actions
    return []

'''
The astar function works by maintaining a priority queue fringe of nodes to explore, where the priority of a node is the sum of its actual cost, the estimated cost to reach the goal state, and the cost of all the nodes on the path from the start node to the current node.
At each iteration, the function pops the node with the lowest priority from the fringe, generates its successor nodes, and adds them to the fringe if they have not been visited before.
If a successor node is the goal state, the function returns the sequence of actions from the start node to that node. Otherwise, the function continues until the fringe is empty or a solution is found.

Note that the astar function uses the fn_heuristic parameter to estimate the cost to reach the goal state from a given state. 
The heuristic function should take a state as input and return an estimate of the cost to reach the goal state from that state. One example of a heuristic function for SingleFoodSearchProblem is the maze_distance function from earlier,
which calculates the Manhattan distance between the current state and the food location.
'''

# YC2-4:
# Here is the modified astar function in searchAgents.py that works for both SingleFoodSearchProblem and MultiFoodSearchProblem:

def astar(problem, fn_heuristic):
    """
    A* search algorithm for SingleFoodSearchProblem and MultiFoodSearchProblem
    """
    start_state = problem.get_initial_state()
    visited = set()
    frontier = PriorityQueue()
    frontier.put((0, start_state, []))  # (priority, state, actions)

    while not frontier.empty():
        priority, current_state, actions = frontier.get()

        if problem.is_goal(current_state):
            return actions

        if current_state in visited:
            continue

        visited.add(current_state)

        for next_state, action, cost in problem.get_successors(current_state):
            if next_state in visited:
                continue

            heuristic = fn_heuristic(next_state)
            new_actions = actions + [action]
            new_cost = problem.get_cost_of_actions(new_actions) + heuristic
            frontier.put((new_cost, next_state, new_actions))

    return []  # Failed to find a path

'''
The only difference between this implementation and the one in YC2-3 is that this implementation takes a problem parameter instead of a SingleFoodSearchProblem parameter,
and the problem object has to implement the same methods as SingleFoodSearchProblem and MultiFoodSearchProblem. This allows the function to work with both types of problems.
'''
