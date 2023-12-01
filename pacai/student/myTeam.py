from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
import random


def createTeam(firstIndex, secondIndex, isRed, first='', second=''):
    """
    This function should return a
     list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    OffenseAgent = OffensiveReflexAgent
    DefenseAgent = DefensiveReflexAgent

    return [
        OffenseAgent(firstIndex),
        DefenseAgent(secondIndex),
    ]


class CustomAgent(CaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.gridP = None
        self.chokeP = []
        self.deadends = []
        self.maxx = None
        self.maxy = None
        self.mapping = None
        self.chokepointMapping = {}
        self.mySide = []
        self.enemySide = []
        self.food = None
        self.visited = set()

    def isDeadEnd(self, point):
        adjacentActions = self.getAdjacentActions(point)
        return len(adjacentActions) == 1

    def getAdjacentActions(self, point):
        x, y = point
        adjacentPoints = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        return [p for p in adjacentPoints if p in self.gridP]

    def findDeadEnd(self, gridPoints):
        for point in gridPoints:
            if self.isDeadEnd(point):
                self.deadends.append(point)

    def registerInitialState(self, gameState):
    
        super().registerInitialState(gameState)

        self.generateGridPoints(gameState)
        self.findDeadEnd(self.gridP)  
        self.updateChokePoints(self.deadends, gameState)
        self.food = gameState.getFood().asList()

    def chooseAction(self, gameState):
        return 'Stop'

    def updateChokePoints(self, deadends, gameState):
        self.resetVariables()
        self.findChokePoint(deadends, gameState)
        
        for position, steps, food in self.chokeP:
            if food == 0:
                continue

            item = (position, 2 * steps, food)
            self.chokepointMapping[position] = (2 * steps, food)
            self.categorizeChokePoint(item, gameState)

        self.highlightChokePoints(gameState)

    def categorizeChokePoint(self, chokePoint, gameState):
        position, _, _ = chokePoint
        x, _ = position

        if x < (self.maxx / 2) and not gameState.isOnRedTeam(self.index):
            self.enemySide.append(chokePoint)
        else:
            self.mySide.append(chokePoint)

    def highlightChokePoints(self, gameState):
        for position, _, _ in self.chokeP:
            self.visited.add(position)
        gameState.setHighlightLocations(self.visited)

    def resetVariables(self):
    
        self.visited.clear()
        self.chokeP = []
        self.enemySide = []
        self.mySide = []
        self.chokepointMapping = {}

    def _possibleActions(self, position, visitedList):
        x, y = position
        adjacentPoints = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        return [p for p in adjacentPoints if p in self.gridP and p not in visitedList]

    def _dfs(self, curr, prev, knownGrids, visited, num_food, gameState, visitedList):
        actions = self._possibleActions(curr, visitedList)
        if prev is None and len(actions) != 1:
            self._processNode(curr, visited, num_food, gameState)
            return

        for c in list(self.chokeP):
            if c[0] == curr:
                self.chokeP.remove(c)

        visitedList.append(curr)
        visited.append(curr)
        num_food += gameState.hasFood(*curr)

        if len(actions) != 1:
            self._processNode(curr, visited, num_food, gameState)
        else:
            self._dfs(actions[0], curr, knownGrids, visited, num_food, gameState, visitedList)

    def _processNode(self, curr, visited, num_food, gameState):
        item = (curr, len(visited), num_food)
        self.chokeP.append(item)

    def findChokePoint(self, deadends, gameState):
        visitedList = []

        for d in deadends:
            initial_food = gameState.hasFood(*d)
            self._dfs(d, None, self.gridP, [], initial_food, gameState, visitedList)

    def getPositionFromAction(self, position, action):    
            x, y = position
            if action == 'North':
                return (x, y + 1)
            elif action == 'South':
                return (x, y - 1)
            elif action == 'West':
                return (x - 1, y)
            elif action == 'East':
                return (x + 1, y)
            elif action == 'Stop':
                return position

    def generateGridPoints(self, gameState):
        walls = gameState.getWalls().asList()
        maxx, maxy = max(walls)
    
        self.gridP = [
            (x, y) for x in range(maxx) for y in range(maxy)
            if not gameState.hasWall(x, y)
        ]
        self.maxx = maxx
        self.maxy = maxy

    def findItemInList(self, item, List):
        for elem in List:
            p, b, n = elem
            if p == item:
                return True
        return False

    def distGhostToPacman(self, gameState):
        my_pos = gameState.getAgentPosition(self.index)
        ghost_pos = [
            gameState.getAgentState(i).getPosition()
            for i in self.getOpponents(gameState)]
        dist_between = [self.getMazeDistance(my_pos, gp) for gp in ghost_pos]
        sorted(dist_between)
        return dist_between[0]

class OffensiveReflexAgent(ReflexCaptureAgent, CustomAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        currentFood = gameState.getFood().asList()
        if self.food != currentFood:
            self.food = currentFood
            self.updateChokePoints(self.deadends, gameState)

        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, action) for action in actions]
        maxValue = max(values)
        bestActions = [action for action, value in zip(actions, values) if value == maxValue]

        if 'Stop' in bestActions and len(bestActions) > 1:
            bestActions.remove('Stop')

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        pos_from_action = self.getPositionFromAction(myPos, action)
        if pos_from_action in self.enemySide:
            s, f = self.chokepointMapping.get(pos_from_action, (0, 0))
            features['chokepoint'] = (-s + f)
        else:
            features['chokepoint'] = -3

        features['successorScore'] = self.getScore(successor)

        foodList = self.getFood(successor).asList()
        if foodList:
            distances = [self.getMazeDistance(myPos, food) for food in foodList]
            features['distanceToFood'] = min(distances)

        ghostDistances = []
        for ghostIndex in self.getOpponents(gameState):
            ghostState = successor.getAgentState(ghostIndex)
            if ghostState.getPosition() is not None and ghostState.getScaredTimer() <= 1:
                distance = self.getMazeDistance(myPos, ghostState.getPosition())
                ghostDistances.append(distance)
        features['distanceToGhost'] = 1 / min(ghostDistances) if ghostDistances else 0

        return features

    def getWeights(self, gameState, action):
    
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'chokepoint': 0.2,
            'distanceToGhost': -2
        }


class DefensiveReflexAgent(ReflexCaptureAgent, CustomAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        midx, midy = self.getMidPoint(gameState)
        distanceToMidPoint = self.getMazeDistance(myPos, (midx, midy))
        features['defendMid'] = -distanceToMidPoint

        if myState.isPacman():
            features['onDefense'] = 0
        else:
            features['onDefense'] = 1

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = []
        for enemy in enemies:
            if enemy.isPacman() and enemy.getPosition() is not None:
                invaders.append(enemy)
        features['numInvaders'] = len(invaders)

        if invaders:
            distancesToInvaders = []
            for invader in invaders:
                distance = self.getMazeDistance(myPos, invader.getPosition())
                distancesToInvaders.append(distance)
            features['invaderDistance'] = min(distancesToInvaders)
            features['defendMid'] = 1

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        currentDirection = gameState.getAgentState(self.index).getDirection()
        reverseAction = Directions.REVERSE[currentDirection]
        if action == reverseAction:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def getMidPoint(self, gameState):
        walls = gameState.getWalls()
        gameBoardHeight = walls.getHeight()
        gameBoardWidth = walls.getWidth()
        centerLine = gameBoardWidth // 2
        sideOffset = -2 if self.red else 2

        defensiveOpenings = []
        for y in range(1, gameBoardHeight - 1):
            if not walls[int(centerLine)][y] and not walls[int(centerLine + sideOffset)][y]:
                defensiveOpenings.append((centerLine + sideOffset, y))

        midOpeningDistance = float('inf')
        midOpening = None
        for opening in defensiveOpenings:
            _, y = opening
            distanceToMid = abs(gameBoardHeight / 2 - y)
            if distanceToMid < midOpeningDistance:
                midOpeningDistance = distanceToMid
                midOpening = opening

        return midOpening

    def getWeights(self, gameState, action):
    
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -0.5,
            'defendMid': 10,
            'defendMidTop': 0,
            'defendMidBot': 0,
            'defendMidBot': 0,
            'defendMid': 10
        }

