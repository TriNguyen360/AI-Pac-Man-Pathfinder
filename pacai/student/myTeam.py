from pacai.agents.capture.capture import CaptureAgent
import random

class OffensiveAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor (game state) after an action.
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        bestScore = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.evaluateOffensiveSuccessor(successor)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

    def evaluateOffensiveSuccessor(self, successor):
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        foodDistance = min([self.getMazeDistance(myPos, food) for food in foodList], default=0)

        # Consider nearby food density
        nearbyFoodCount = sum(1 for food in foodList if self.getMazeDistance(myPos, food) < 5)

        # Consider ghost proximity
        ghostDistances = [self.getMazeDistance(myPos, successor.getAgentState(ghost).getPosition())
                          for ghost in self.getOpponents(successor)
                          if not successor.getAgentState(ghost).isPacman()]
        ghostDistance = min(ghostDistances, default=float('inf'))

        score = -2 * foodDistance + 2 * nearbyFoodCount
        if ghostDistance < 5:  # Avoid close ghosts
            score -= 20 / ghostDistance
        return score
class DefensiveAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor (game state) after an action.
        """
        successor = gameState.generateSuccessor(self.index, action)
        return successor
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        bestAction = None
        bestScore = float('-inf')

        for action in actions:
            successor = self.getSuccessor(gameState, action)
            score = self.evaluateDefensiveSuccessor(successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

    def evaluateDefensiveSuccessor(self, successor):
        """
        Evaluate the successor state for defensive strategy.
        """
        myPos = successor.getAgentState(self.index).getPosition()
        invaders = [successor.getAgentState(i) for i in self.getOpponents(successor)
                    if successor.getAgentState(i).isPacman() and successor.getAgentState(i).getPosition() is not None]
        foodList = self.getFoodYouAreDefending(successor).asList()
        
        # Calculate score
        score = 0
        if invaders:
            invaderDistances = [self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]
            score -= min(invaderDistances)  # Chase invaders
        else:
            foodDistance = min([self.getMazeDistance(myPos, food) for food in foodList], default=0)
            score -= foodDistance  # Patrol food
        return score


def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.student.myTeam.OffensiveAgent',
               second='pacai.student.myTeam.DefensiveAgent'):
    """
    This function returns a list of two agents that will form the capture team.
    """
    firstAgent = OffensiveAgent(firstIndex)
    secondAgent = DefensiveAgent(secondIndex)
    return [firstAgent, secondAgent]
