# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]


  def findManhattanDistance(self, p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    dist = abs(x1 - x2) + abs(y1 - y2)
    return dist

  #scores how the agent is doing in respect to food
  def getFoodScore(self, newPos, oldPos, oldFood):
    from decimal import Decimal 
    numOldFood = len(oldFood)
    numNewFood = len(oldFood)
    foodPoints = 0 
      
    #if we ate a food, add a point, 
    if newPos in oldFood:
        #print "can capture food!"
        numNewFood = numNewFood - 1
        foodPoints += 1
      
    #find food distances
    else:
        foodDistances = [0]* len(oldFood)
        i = 0
        for curFood in oldFood:
            oldD = self.findManhattanDistance(curFood, oldPos)
            newD = self.findManhattanDistance(curFood, newPos) 
            foodDistances[i] = newD
            i = i+1
        #print "fd:  ",foodDistances
        #print "min: ", Decimal(1/Decimal(min(foodDistances)))
        foodPoints = Decimal(1/Decimal(min(foodDistances)))
    #print "food points: ", foodPoints
    return Decimal(foodPoints)

  def evaluationFunction(self, currentGameState, action):
    from decimal import Decimal
    """
    Design a better evaluation function here.

    The evaluation function takes in the current GameState and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state:
    oldFood = remaining food 
    newPos  = Pacman position after moving.
    newScaredTimes = holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos         = successorGameState.getPacmanPosition()
    oldPos         = currentGameState.getPacmanPosition()
    oldFood        = currentGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    "print highest number for best performance."
    "Base performance measure on getting closer to food and farther from ghost"
    ""
    #print "food: ", oldFood
    #print "num food:       ", len(oldFood)
    #print "current score:  ", currentGameState.getScore()
    #print "successor score ", successorGameState.getScore()
    #print "ghost states: ", newGhostStates[0].getPosition()  
    
    # get food points
    foodPoints = self.getFoodScore(newPos, oldPos, oldFood)
    score = Decimal(foodPoints)
    
    #get staying away from ghost points:
    for curGhost in newGhostStates:
        gp =  curGhost.getPosition()
        gd =  curGhost.getDirection()
        if gp[0]!= newPos[0] and gp[1]!=newPos[1] :
            score +=2
        elif gp[0]== newPos[0] and gp[1]==newPos[1]:
            score = abs(score) - abs(score)#score - 2*(len(newGhostStates) + 1) 
    #print "score: ", score
    return Decimal(score)

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    numAgents =  gameState.getNumAgents()
    ai =  self.index #index of current agent
    legalActions =  gameState.getLegalActions(ai)

    print "legal actions: ", legalActions
    
    #print "game state: ", gameState
    #print "finished: ", gameState.getFood().asList()
    numAgents = gameState.getNumAgents()
    direction = self.Minimax_Decision(gameState, legalActions, numAgents, ai)
    print "The direction we are going is: ", direction
    return direction
    #return legalActions[2]
    #util.raiseNotDefined()
    
  
  #returns the best action to take
  def Minimax_Decision(self, currentGameState, legalActions, numGhosts, ai):
      curUtility = -1 
      bestUtility = -1
      bestAction = legalActions[0]
      
      for action in legalActions:
          successorGameState = currentGameState.generatePacmanSuccessor(action)
          #does the argmax part of the code:
          curUtility = self.MinValue(successorGameState, numGhosts, ai, 0)
          if curUtility >= bestUtility:
              bestAction = action
              bestUtility = curUtility
              
      return bestAction 
  
  #returns a utility value
  def MaxValue(self, state, numGhosts, ai, depth):
      #depth +=1 
      #print "depth ", depth
      

      if self.terminalTest(state, depth):
          valToReturn = self.evaluationFunction(state)
          #print "terminating with value", valToReturn
          return valToReturn 
      
      #we want the legal actions for pacman when we maximize
      pacmanLegalActions =  state.getLegalActions(ai)
      v = -1;
      vSet = False
      
      for a in pacmanLegalActions:
          if vSet: 
              v = max (v, self.MinValue(state.generatePacmanSuccessor(a), numGhosts, ai, depth+1))
          #this will only get done on the first iteration of the loop
          if not vSet:
              v = self.MinValue(state.generatePacmanSuccessor(a), numGhosts, ai, depth+1)
              vSet = True
          
      return v
  
  def MinValue(self, state, numGhosts, ai, depth):
      
      if self.terminalTest(state, depth):
          return self.evaluationFunction(state)
      import sys
      v = sys.maxint
      
      possibleActionSets = self.allActionsForAllGhosts(state, numGhosts)
      for curSetOfActions in possibleActionSets:
          
          for actionIndex in range(len(curSetOfActions)):
              #get the state after all ghosts have moved
              newState = state.generateSuccessor(actionIndex+1, curSetOfActions[actionIndex]) 
          v = min (v, self.MaxValue(newState, numGhosts, ai, depth))
                  
                  
      return v          

  
  #get 
  def allActionsForAllGhosts(self, state, numGhosts):
    allGhostsActions = []
    for ghost in range(1, numGhosts):
    #get all actions for all ghosts  
        allGhostsActions.append(state.getLegalActions(ghost))
        
    #get all combinations actions of ghost1, ghost2 etc.
    from itertools import product   
    possibleActionSets = product (*allGhostsActions)
    return possibleActionSets
  
  # returns true if the game is over
  #         false if not
  def terminalTest (self, state, depth):

      if depth >= self.depth:
          return True
      #if no food is left the game is over
      foodList = state.getFood().asList()
      if len(foodList) == 0:
          return True
      
      #if pacman and ghost have the same position, the game is over
      pacmanPos = state.getPacmanPosition()
      ghostStates = state.getGhostStates()
      
      #check if pacman eaten by ghost
      for ghostState in ghostStates:
          ghostPos = ghostState.getPosition()
          if ghostPos == pacmanPos:
              return True

      #if there is still food left and pacman is not 
      #eaten by a ghost - return false (game is not over)
      return False

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
