from Source.problems import SingleFoodSearchProblem
from Source.searchAgents import bfs

def main():
    problem = SingleFoodSearchProblem(maze_file = 'maze.txt')
    actions = bfs(problem)
    problem.animate(actions)
    
if __name__ == '__main__':
    main()