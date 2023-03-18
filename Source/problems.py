import os 
import time

class Node:
    def __init__(self, state, cost, last_path):
        self.state = state 
        self.cost = cost
        self.path = [p for p in last_path] 
        self.path.append(self.state)

class Problem:
    pass 
# YC1-1
class SingleFoodSearchProblem:
    def __init__(self, maze_file):
        self.maze = self.read_maze(maze_file)
        self.goal_state = self.get_goal_state()
    
    def expand(self, node):
        cur_i, cur_j = node.state
        for i in [cur_i - 1, cur_i, cur_i + 1]:
            for j in [cur_j - 1, cur_j, cur_j + 1]:
                if i < 0 or j < 0:
                    continue 
                if (i, j) == (cur_i, cur_j):
                    continue
                
                if self.maze[i][j] in [' ', '.']:
                
                    yield Node((i, j), node.cost + 1, node.path)
        
        
    def initial_state(self):
        """
        Returns the start state of the problem, which is the location of Pacman.
        """
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == 'P':
                    return Node((i, j), 0, [])

    def get_goal_state(self):
        """
        Returns the goal state of the problem, which is the location of the food.
        """
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.maze[i][j] == '.':
                    return (i, j)

    def goal_test(self, state):
        """
        Returns True if the given state is the goal state of the problem.
        """
        return state == self.goal_state

    def get_successors(self, state):
        """
        Returns a list of successor tuples (successor, action, cost).
        """
        successors = []
        x, y = state

        if x > 0 and self.maze[x - 1][y] != '%':
            successors.append(((x - 1, y), 'North', 1))
        if x < len(self.maze) - 1 and self.maze[x + 1][y] != '%':
            successors.append(((x + 1, y), 'South', 1))
        if y > 0 and self.maze[x][y - 1] != '%':
            successors.append(((x, y - 1), 'West', 1))
        if y < len(self.maze[0]) - 1 and self.maze[x][y + 1] != '%':
            successors.append(((x, y + 1), 'East', 1))

        return successors

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a path that goes from state1 to state2 via action,
        which is always 1 in this problem.
        """
        return c + 1

    def read_maze(self, maze_file):
        """
        Reads the maze from the given text file and returns a 2D list of the maze.
        """
        with open(maze_file, 'r') as f:
            maze = [line.strip() for line in f.readlines()]

        return maze

    def print_maze(self):
        """
        Prints the maze on the screen.
        """
        for line in self.maze:
            print(line)
            
    # YC1-4:
    '''
        The animate() method takes in a list of actions, which represent the sequence of moves that pacman should make to reach the food.
        It starts by setting the current position of pacman to the initial position, and then enters a loop that iterates over the actions. 
        In each iteration, the method clears the screen, prints the maze (including the current position of pacman and the location of the food), waits for the user to press Enter, 
        and then moves pacman to the next position according to the next action in the list. 
        The time.sleep(0.5) call slows down the animation so that it is visible to the user.
    '''
    def animate(self, actions):
        current = self.initial_state().state
        init_pos = current
        
        while actions:
            os.system('cls' if os.name == 'nt' else 'clear')
            for row in range(len(self.maze)):
                for col in range(len(self.maze[row])):
                    if (row, col) == current:
                        print('P', end='')
                    elif (row, col) == init_pos:
                        print(end = ' ')                        
                    else:
                        print(self.maze[row][col], end = '')
                print()
            current = actions.pop(0)
            time.sleep(0.5)
            input('Press Enter to continue')      

    

'''
The SingleFoodSearchProblem class takes a maze file as input and initializes the maze, start_state, and goal_state attributes.
# It also implements the get_successors, is_goal_state, and path_cost methods, which define the search problem.
# The read_maze method reads the maze from the text file and returns a 2D list of the maze. The print_maze method prints the maze on the screen.
To use this class, the students can create an instance of the class and then use any uninformed search algorithm to solve the problem.
Here's an example of how to use the class to solve the problem using BFS:

'''
# from queue import Queue

# def bfs(problem):
#     frontier = Queue()
   
'''
In this implementation, we define the SingleFoodSearchProblem class, which represents the problem of finding a path from the initial state to the food in a maze. The class has several methods:

    __init__(self, maze_file): Initializes the problem by reading the maze layout from the given file.

    read_maze(self, maze_file): Reads the maze layout from the given file.

    print_maze(self): Prints the maze on the screen.

    successor(self, state): Returns the successor states of the given state.

    is_valid(self, row, col): Checks if the given row and column are valid coordinates in the maze.

    set_start_state(self, start_state): Sets the initial state.

    set_goal_state(self, goal_state): Sets the goal state.

    is_goal_state(self, state): Checks if the given state is the goal state.

    path_cost(self, c, state1, action, state2): Returns the cost of the path from state1 to state2 via action, assuming a cost of c to get to state1
'''
# YC1-5:
class MultiFoodSearchProblem:
    
    def __init__(self, maze_file):
        self.maze = self.read_maze(maze_file)
        self.food = self.find_food()
        self.start = self.find_start()
        self.actions = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}
        
    def read_maze(self, maze_file):
        with open(maze_file) as f:
            maze = [list(line.strip()) for line in f]
        return maze
    
    def find_food(self):
        food = []
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == '.':
                    food.append((i, j))
        return food
    
    def find_start(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == 'P':
                    return (i, j)
        return None
    
    def initial_state(self):
        return Node(self.start, self.food)
    
    def goal_test(self, node):
        return len(node.state[1]) == 0
    
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def actions(self, state):
        return ['N', 'S', 'W', 'E', 'Stop']
    
    def result(self, state, action):
        if action == 'Stop':
            return state
        current, food = state
        row, col = current
        row += self.actions[action][0]
        col += self.actions[action][1]
        new_pos = (row, col)
        if self.maze[row][col] == '%':
            return state
        new_food = list(food)
        if new_pos in food:
            new_food.remove(new_pos)
        return Node(new_pos, new_food)
    def animate(self, actions):
        current = self.start
        food_left = len(self.food)
        
        while actions:
            os.system('cls' if os.name == 'nt' else 'clear')
            for row in range(len(self.maze)):
                for col in range(len(self.maze[row])):
                    if (row, col) == current:
                        print('P', end=' ')
                    elif (row, col) in self.food:
                        print('.', end=' ')
                    elif self.maze[row][col] == '%':
                        print('%', end=' ')
                    else:
                        print(' ', end=' ')
                print()
            action = actions.pop(0)
            current = (current[0] + self.actions[action][0], current[1] + self.actions[action][1])
            if current in self.food:
                food_left -= 1
                self.food.remove(current)
            time.sleep(0.5)
            input('Press Enter to continue')


'''
The MultiFoodSearchProblem class is similar to the SingleFoodSearchProblem class, but with a few modifications to support the collection of multiple food items. 
The find_food() method now returns a list of food items instead of a single position, and the initial_state() method now initializes the node with a tuple of the starting position and the list of food items. 
The goal_test() method now checks if there are no more food items left to collect.
The result() method now returns a new node with an updated state that includes the new position and the updated list of food items
'''



# YC3-1:

'''
To calculate the value of each cell in the heuristic function h(state), we need to check how many other queens can attack the current cell's position. 
In this case, we are assuming that there is exactly one queen in each column.

One way to calculate the heuristic value for a given state is to loop over all the columns of the board and compute the number of queen couples that are able to attack each other.
We can start by initializing a count variable to zero and then for each column, we can compute the number of attacking queen couples.
If a pair of queens in the same column, we skip it as we know that there is exactly one queen in each column.

Here is the pseudocode for the h(state) heuristic function:

function h(state):
    count = 0
    for column in range(8):
        row = state[column]
        for i in range(column + 1, 8):
            other_row = state[i]
            if row == other_row:
                count += 1
            elif abs(row - other_row) == abs(column - i):
                count += 1
    return count

In this pseudocode, state represents the current state of the board,
which is a list of length 8 containing the row index of the queen in each column. The variable count is initialized to zero and will be incremented for each pair of queens that are able to attack each other.

To compute the heuristic value for a specific cell,
we can temporarily set the queen in that column to the given row and then compute the heuristic value for the resulting state using the h function defined above.
'''

class EightQueenProblem(Problem):

    def __init__(self, initial_state):
        super().__init__(initial_state)
        self.goal_state = None

    def actions(self, state):
        # In the 8-queens problem, each column already has a queen, so there are no actions to take
        return []

    def result(self, state, action):
        # In the 8-queens problem, each column already has a queen, so the result of any action is the same state
        return state

    def goal_test(self, state):
        # The goal state in the 8-queens problem is any state where there are no mutual attacks between queens
        for i in range(8):
            for j in range(i+1, 8):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    return False
        return True

    def h(self, state):
        # The h() function calculates the number of queen couples that are able to attack mutually
        h_value = 0
        for i in range(8):
            for j in range(i+1, 8):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    h_value += 1
        return h_value

    def read_board_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            self.initial_state = [int(x) for x in lines]

    def print_board(self, state):
        for i in range(8):
            line = ''
            for j in range(8):
                if state[i] == j:
                    line += 'Q '
                else:
                    line += '_ '
            print(line)


# YC3-2 : Here's one possible implementation of the hill_climbing_search method for the EightQueenProblem class, following the specification given

# class EightQueenProblem(Problem):
#     def __init__(self, initial_state=None):
#         self.initial_state = initial_state or self.read_board("initial_board.txt")
#         self.num_rows = len(self.initial_state)
#         self.num_cols = len(self.initial_state[0])
        
#     def read_board(self, filename):
#         with open(filename) as f:
#             board = [list(line.strip()) for line in f]
#         return board
    
#     def print_board(self, state):
#         for row in state:
#             print(" ".join(row))
    
#     def h(self, state):
#         def is_attacking(q1, q2):
#             (r1, c1), (r2, c2) = q1, q2
#             return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)
        
#         num_attacking = 0
#         queens = [(r, c) for r in range(self.num_rows) for c in range(self.num_cols) if state[r][c] == "Q"]
#         for i, q1 in enumerate(queens):
#             for q2 in queens[i+1:]:
#                 if is_attacking(q1, q2):
#                     num_attacking += 1
#         return num_attacking
    
#     def hill_climbing_search(self):
#         current_state = self.initial_state
#         while True:
#             successors = []
#             for col in range(self.num_cols):
#                 for row in range(self.num_rows):
#                     if current_state[row][col] == "Q":
#                         continue
#                     successor = deepcopy(current_state)
#                     successor[row][col] = "Q"
#                     successors.append(successor)
#             if not successors:
#                 # Local maximum found
#                 return current_state
#             best_successor = min(successors, key=self.h)
#             if self.h(best_successor) >= self.h(current_state):
#                 # Local maximum found
#                 return current_state
#             current_state = best_successor
'''
The hill_climbing_search method initializes the current state to the initial state, 
and then repeatedly generates all possible successors of the current state by moving the queen in each column to the cell with the minimum heuristic value in that column. 
It then selects the best successor according to the heuristic function h, and if this successor is better than the current state, updates the current state to this successor and repeats. 
The method returns the current state when it reaches a local maximum, which is when no successor has a lower heuristic value than the current state.
'''