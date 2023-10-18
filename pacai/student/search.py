"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # Initialize fringe and visited set
    fringe = Stack()
    visited = set()  

    start_state = problem.startingState()  
    fringe.push((start_state, []))
    visited.add(start_state)  # Make sure start state is visited

    while not fringe.isEmpty():
        state, actions = fringe.pop()

        
        if problem.isGoal(state):
            return actions

        # Get successors and add them to the fringe if they haven't been visited yet
        for successor, action, _ in problem.successorStates(state):
            if successor not in visited:
                visited.add(successor)
                fringe.push((successor, actions + [action]))

    # Error handling
    raise Exception("No solution found")


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # Initialize fringe as a Queue and visited set
    fringe = Queue()
    visited = set()

    
    start_state = problem.startingState()
    fringe.push((start_state, []))
    visited.add(start_state)  

    
    while not fringe.isEmpty():
        state, actions = fringe.pop()

        
        if problem.isGoal(state):
            return actions

        # For each successor of the state
        for successor, action, _ in problem.successorStates(state):
            if successor not in visited:
                visited.add(successor)
                # Enqueue the successor state with the appropriate actions list
                fringe.push((successor, actions + [action]))

    # Error handling
    raise Exception("No solution found")

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    #Initialize fringe as a priority queue and visited set
    fringe = PriorityQueue()
    visited = set()

    start_state = problem.startingState()
    fringe.push((start_state, []), 0)  # The priority here is 0
    visited.add(start_state)  # Mark the start state as visited

    
    while not fringe.isEmpty():
        state, actions = fringe.pop()

        
        if problem.isGoal(state):
            return actions

        # For each successor of the state
        for successor, action, cost in problem.successorStates(state):
            if successor not in visited:
                visited.add(successor)
                new_actions = actions + [action]
                # Calculate the new cost for this path
                new_cost = problem.actionsCost(new_actions)
                # Enqueue the successor state with the appropriate actions list and cost
                fringe.push((successor, new_actions), new_cost)

    # Error handling
    raise Exception("No solution found")

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
        # Initialize the fringe as a priority queue and the visited set
    fringe = PriorityQueue()
    visited = set()

    # Get the starting state and enqueue it with a cost of 0
    start_state = problem.startingState()
    fringe.push((start_state, [], 0), 0)  # The priority here is 0
    visited.add(start_state)  # Mark the start state as visited

    # Continue until there are no more states to explore (fringe is empty)
    while not fringe.isEmpty():
        state, actions, current_cost = fringe.pop()

        # Check if this state is the goal
        if problem.isGoal(state):
            return actions

        # For each successor of the state
        for successor, action, step_cost in problem.successorStates(state):
            if successor not in visited:
                visited.add(successor)
                new_actions = actions + [action]
                new_cost = current_cost + step_cost
                # Calculate the new priority for this path
                # Priority is the current cost plus the estimated cost to the goal
                priority = new_cost + heuristic(successor, problem)
                # Enqueue the successor state with the appropriate actions list and priority
                fringe.push((successor, new_actions, new_cost), priority)

    # If there's no solution, raise an exception
    raise Exception("No path to the goal state exists.")
