from Source.problems import SingleFoodSearchProblem
from Source.searchAgents import bfs

def main():
    problem = SingleFoodSearchProblem(maze_file = 'pacman_single01.txt')
    actions = bfs(problem)
    problem.animate(actions)
    
if __name__ == '__main__':
    main()