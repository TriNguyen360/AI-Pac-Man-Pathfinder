from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.util import reflection

class AlphaBetaAgent(CaptureAgent):
    def chooseAction(self, gameState):
        """
        Choose an action using alpha-beta pruning.
        """
        return self.alphaBetaSearch(gameState, depth=3)

    def alphaBetaSearch(self, gameState, depth):
        def minValue(gameState, alpha, beta, depth, agentIndex):
            if gameState.isOver() or depth == 0:
                return self.evaluationFunction(gameState)
            
            value = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = min(value, maxValue(successor, alpha, beta, depth, agentIndex + 1))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        def maxValue(gameState, alpha, beta, depth, agentIndex):
            if gameState.isOver() or depth == 0:
                return self.evaluationFunction(gameState)

            value = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, minValue(successor, alpha, beta, depth - 1, agentIndex + 1))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value

        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        bestValue = float("-inf")

        for action in gameState.getLegalActions(self.index):
            value = minValue(gameState.generateSuccessor(self.index, action), alpha, beta, depth, self.index + 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

    def evaluationFunction(self, gameState):
        """
        Base evaluation function, which should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by a subclass")
    

class OffensiveAlphaBetaAgent(AlphaBetaAgent):
    def evaluationFunction(self, gameState):
        return self.offensiveEvaluationFunction(gameState)

    def offensiveEvaluationFunction(self, gameState):
        """
        Custom evaluation for offensive strategy.
        """
        foodList = self.getFood(gameState).asList()
        capsules = self.getCapsules(gameState)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Score based on remaining food
        score = -len(foodList)

        # Distance to the closest food
        if len(foodList) > 0:
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score -= 2 * minFoodDistance

        # Consider power capsules if there are any
        if len(capsules) > 0:
            minCapsuleDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
            score -= 2 * minCapsuleDistance

        # Avoid ghosts if they are close
        ghostDistances = [self.getMazeDistance(myPos, gameState.getAgentState(ghost).getPosition()) for ghost in self.getOpponents(gameState) if not gameState.getAgentState(ghost).isPacman()]
        for d in ghostDistances:
            if d < 5:  # if a ghost is too close
                score -= 5 * (5 - d)

        return score

class DefensiveAlphaBetaAgent(AlphaBetaAgent):
    def evaluationFunction(self, gameState):
        return self.defensiveEvaluationFunction(gameState)

    def defensiveEvaluationFunction(self, gameState):
        """
        Custom evaluation for defensive strategy.
        """
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        invaders = [gameState.getAgentState(i) for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman() and gameState.getAgentState(i).getPosition() != None]
        numInvaders = len(invaders)

        # Score based on number of invaders
        score = -100 * numInvaders

        # Distance to the closest invader
        if numInvaders > 0:
            dists = [self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]
            score -= 2 * min(dists)

        # Protect remaining food
        foodList = self.getFoodYouAreDefending(gameState).asList()
        if len(foodList) > 0:
            minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            score -= minFoodDistance

        return score

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.myTeam.OffensiveAlphaBetaAgent',
        second = 'pacai.student.myTeam.DefensiveAlphaBetaAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
