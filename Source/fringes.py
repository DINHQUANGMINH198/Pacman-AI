import heapq

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Queue:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.insert(0, item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class PriorityQueue:
    def __init__(self):
        self.items = []

    def push(self, item, priority):
        heapq.heappush(self.items, (priority, item))

    def pop(self):
        return heapq.heappop(self.items)[1]

    def is_empty(self):
        return len(self.items) == 0


# The Stack class implements a stack data structure using a list. The push method adds an item to the top of the stack, the pop method removes and returns the top item from the stack, and the is_empty method returns True if the stack is empty.

# The Queue class implements a queue data structure using a list. The push method adds an item to the end of the queue, the pop method removes and returns the first item from the queue, and the is_empty method returns True if the queue is empty.

# The PriorityQueue class implements a priority queue data structure using a heap. The push method adds an item to the priority queue with the given priority, the pop method removes and returns the item with the highest priority from the priority queue, and the is_empty method returns True if the priority queue is empty. 
# Note that in Python's heapq module, the lowest value is given the highest priority.