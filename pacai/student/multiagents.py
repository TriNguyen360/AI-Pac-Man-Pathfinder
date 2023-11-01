import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent

from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Extract relevant information from the game state.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        score = successorGameState.getScore()

        # Consider distance to the nearest food.
        foodDistances = [distance.manhattan(newPosition, food) for food in oldFood.asList()]
        if foodDistances:
            closestFoodDistance = min(foodDistances)
            if closestFoodDistance == 0:
                score += 1000
            else:
                score += 1.0 / closestFoodDistance

        # Consider distance to the ghosts.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = distance.manhattan(newPosition, ghostState.getPosition())
            if ghostDistance <= 1 and scaredTime == 0:
                score -= 1000
            elif ghostDistance <= 1 and scaredTime > 0:
                score += 500

        # Consider the number of remaining food pellets.
        remainingFood = len(oldFood.asList())
        score -= remainingFood * 10

        return score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        # Get all possible actions for Pacman.
        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        # Compute Minimax value.
        values = [self.minimax(gameState.generateSuccessor(0, action), 1, 0) for action in actions]

        # Return the action with the maximum Minimax value.
        return actions[values.index(max(values))]

    def minimax(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)

        # maximizing agent
        if agentIndex == 0:
            nextAgent = 1
            newDepth = currentDepth
            values = [self.minimax(gameState.generateSuccessor(agentIndex, action),
            newDepth, nextAgent)
            for action in gameState.getLegalActions(agentIndex)]
            return max(values)

        # minimizing agent
        else:
            nextAgent = agentIndex + 1
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgent = 0
                newDepth = currentDepth + 1
            else:
                newDepth = currentDepth
            values = [self.minimax(gameState.generateSuccessor(agentIndex, action),
            newDepth, nextAgent)
            for action in gameState.getLegalActions(agentIndex)]
            return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        alpha = float("-inf")
        beta = float("inf")

        actions = gameState.getLegalActions(0)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        bestAction = None
        value = float("-inf")

        # For each action, compute its Minimax value with alpha-beta pruning.
        for action in actions:
            newValue = self.alphabeta(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if newValue > value:
                value = newValue
                bestAction = action
            alpha = max(alpha, value)
        
        return bestAction

    def alphabeta(self, gameState, currentDepth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)

        # maximizing agent
        if agentIndex == 0:
            value = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                value = max(value, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                currentDepth, 1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        # minimizing agent
        else:
            value = float("inf")
            nextAgent = agentIndex + 1
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgent = 0
                newDepth = currentDepth + 1
            else:
                newDepth = currentDepth

            for action in gameState.getLegalActions(agentIndex):
                value = min(value, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                newDepth, nextAgent, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.getTreeDepth():
                return self.getEvaluationFunction()(state)
            
            numAgents = state.getNumAgents()
            
            # maximizing agent
            if agentIndex == 0:
                maxVal = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    maxVal = max(maxVal, expectimax(successor, depth, 1))
                return maxVal
            
            # expectation agent
            else:
                expectedVal = 0
                actions = state.getLegalActions(agentIndex)
                prob = 1.0 / len(actions)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    if agentIndex == numAgents - 1:
                        expectedVal += prob * expectimax(successor, depth + 1, 0)
                    else:
                        expectedVal += prob * expectimax(successor, depth, agentIndex + 1)
                return expectedVal
        
        # expectimax search
        bestAction = Directions.STOP
        maxVal = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1)
            if value > maxVal:
                maxVal = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
