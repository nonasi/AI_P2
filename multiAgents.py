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
    legalActions.remove(Directions.STOP)

    numAgents = gameState.getNumAgents()
    direction = self.Minimax_Decision(gameState, legalActions, numAgents, ai)

    return direction

  #returns the best action to take
  def Minimax_Decision(self, currentGameState, legalActions, numGhosts, ai):
      curUtility = -1 
      bestUtility = -1
      bestAction = legalActions[0]
      nSet = True; 
       
      for action in legalActions:
          successorGameState = currentGameState.generatePacmanSuccessor(action)
          #does the argmax part of the code:
          curUtility = self.MinValue(successorGameState, numGhosts, ai, 0)

          if curUtility >= bestUtility or nSet:
              bestAction = action
              bestUtility = curUtility
              nSet = False
             
      return bestAction 
  
  #returns a utility value
  def MaxValue(self, state, numGhosts, ai, depth):

      if self.terminalTest(state, depth):
          valToReturn = self.evaluationFunction(state)
          #print "terminating with value", valToReturn
          return valToReturn 
      
      #we want the legal actions for pacman when we maximize
      pacmanLegalActions =  state.getLegalActions(ai)
      pacmanLegalActions.remove(Directions.STOP)
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
      
      import copy
      for curSetOfActions in possibleActionSets:
          
          newState = copy.deepcopy(state)
          for actionIndex in range(len(curSetOfActions)):
              #get the state after all ghosts have moved
              #print "goes through for once", newState
              newState = newState.generateSuccessor(actionIndex+1, curSetOfActions[actionIndex])
              if self.terminalTest(newState, depth):
                  break 
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
  """ Your minimax agent with alpha-beta pruning (question 3)"""
  import sys
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    #print "getAction gets called"
    "*** YOUR CODE HERE ***"
    numAgents =  gameState.getNumAgents()
    ai =  self.index #index of current agent
    legalActions =  gameState.getLegalActions(ai)
    legalActions.remove(Directions.STOP)

    numAgents = gameState.getNumAgents()
    direction = self.a_b_Decision(gameState, legalActions, numAgents, ai)
    return direction
    #util.raiseNotDefined()
    
    
    ##############################################################################################
    
      #returns the best action to take acording to a-b pruning
   #returns the best action to take
  def a_b_Decision(self, currentGameState, legalActions, numGhosts, ai):
      import sys
      curUtility = -1 
      bestUtility = -1
      bestAction = legalActions[0]
      nSet = True
       
      for action in legalActions:
          successorGameState = currentGameState.generatePacmanSuccessor(action)
          #does the argmax part of the code:
          curUtility = self.MinValue(successorGameState, numGhosts, ai, 0, -sys.maxint-1, sys.maxint)

          if curUtility >= bestUtility or nSet:
              bestAction = action
              bestUtility = curUtility
              nSet = False
             
      return bestAction 
  
  #returns a utility value
  def MaxValue(self, state, numGhosts, ai, depth,a,b):

      if self.terminalTest(state, depth):
          valToReturn = self.evaluationFunction(state)
          return valToReturn 
      
      #we want the legal actions for pacman when we maximize
      pacmanLegalActions =  state.getLegalActions(ai)
      pacmanLegalActions.remove(Directions.STOP)
      
      import sys
      v = -sys.maxint -1
      vSet = False
      
      for action in pacmanLegalActions: 
          v = max (v, self.MinValue(state.generatePacmanSuccessor(action), numGhosts, ai, depth+1,a,b))    
          if v >= b:
              return v
          a = max(a, v)
          
      return v
  
  def MinValue(self, state, numGhosts, ai, depth, a, b):
      
      if self.terminalTest(state, depth):
          return self.evaluationFunction(state)
      import sys
      
      v = sys.maxint
      possibleActionSets = self.allActionsForAllGhosts(state, numGhosts)
      
      import copy
      for curSetOfActions in possibleActionSets:
          
          newState = copy.deepcopy(state)
          for actionIndex in range(len(curSetOfActions)):
              #get the state after all ghosts have moved
              newState = newState.generateSuccessor(actionIndex+1, curSetOfActions[actionIndex])
              if self.terminalTest(newState, depth):
                  break 
          v = min (v, self.MaxValue(newState, numGhosts, ai, depth, a, b))
          #v = min (v, self.MaxV(newState, numGhosts, ai, depth, a,b))
          if v<=a: 
              return v
          b = min(v, b)
             
      return v          

      
      #get all combinations of actions that 
      #the ghosts can do no their move
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
  # false if not
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
    
    
    ##############################################################################################


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
    ai =  self.index #index of current agent
    numAgents = gameState.getNumAgents()
    direction = self.expectiMax_Decision(gameState, numAgents, ai)
    return direction

#***************************************************************************************

  #turn = true if pacman's turn
  #turn = false otherwise
   
  def expectiMax_Decision(self, state, numGhosts, ai):
      result = self.ExpectiMaxPlayer(state, numGhosts, ai, 0, True)
      #print "this is the result: ", result
      return result[1]
  
  # turn = true if it's pacman's turn; false otherwise
  # return a tuple; at position 0 is v, at position 1 is the action we 
  # must take at this state
  def ExpectiMaxPlayer(self, state, numGhosts, ai, depth, turn ):
      
      if self.terminalTest(state, depth):
          return (self.evaluationFunction(state), -1)
      
      if turn: #pacman's turn
          pacmanActions = state.getLegalActions(ai)
          pacmanActions.remove(Directions.STOP)
          maxV = -1
          firstPass = True
          bestAction = pacmanActions[0]
          
          for a in pacmanActions: 
              v = self.ExpectiMaxPlayer(state.generatePacmanSuccessor(a), numGhosts, ai, depth +1, not turn)[0]
              if maxV <= v or firstPass: 
                  maxV = v
                  firstPass = False
                  bestAction = a   
                  
          return (maxV, bestAction)
      
      if not turn: #ghost turn
          actionSetSum = 0 #accumulator 
          v = 0
          possibleActionSets = self.allActionsForAllGhosts(state, numGhosts)
          import copy
          numActionSets = 0
          
          for curSetOfActions in possibleActionSets:
              numActionSets+=1
              newState = copy.deepcopy(state)
              
              for actionIndex in range(len(curSetOfActions)):#get the state after all ghosts have moved
                  newState = newState.generateSuccessor(actionIndex+1, curSetOfActions[actionIndex])
                  if self.terminalTest(newState, depth):
                      break
              v = self.ExpectiMaxPlayer(newState, numGhosts, ai, depth, not turn)[0] 
              actionSetSum += v/ numActionSets 
              
          return (actionSetSum, -1)     
      
      #get all combinations of actions that 
      #the ghosts can do no their move
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
  # false if not
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
#***************************************************************************************
def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  # the farther away the nearest ghost is, the better
  # the closer the nearest food dot is, the better
  #

  import sys
  
  pacPos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  capsulesList=currentGameState.getCapsules()
  
  allghosts=currentGameState.getGhostStates()
  #ghostscaredTimes = [ghos.scaredTimer for ghos in allghosts]
  
  allghosts=currentGameState.getGhostStates()
  #ghostpos=currentGameState.getGhostPositions()
  noneDisabled=True
  disabledGhostIndexes=[]
  activeGhostIndexes=[]
  
  #find the disabled and the normal ghost
  ind=0
  for ghost in allghosts:
    if ghost.scaredTimer>0:
      noneDisabled=False
      disabledGhostIndexes.append(ind) #indexes of disabled ghosts
    else:
      activeGhostIndexes.append(ind)
    ind=ind+1
        
  #find shortest distance to an active ghost
  shortestDistanceToActiveGhost=sys.maxint
  for index in activeGhostIndexes:
    #distance=abs(x - ghostpos[index][0]) + abs(y - ghostpos[index][1])
    distance=astarMazeDistBetweenTwoPoints(currentGameState,pacPos,allghosts[index].getPosition())
    if distance<shortestDistanceToActiveGhost:
      #nearestGhostPos=ghost
      shortestDistanceToActiveGhost=distance
      
    #find shortest distance to a disabled ghost    
  shortestDistanceToDisabledGhost=sys.maxint
  numTimesThroughLoop = 1
  for index in disabledGhostIndexes:
    #print "numTimesThroughLoop", numTimesThroughLoop
    numTimesThroughLoop+=1
    #print "ghost's position: ",allghosts[index].getPosition()
    ghostPos = allghosts[index].getPosition()
    xDecimal = abs(ghostPos[0] - int(ghostPos[0]))
    yDecimal = abs(ghostPos[1] - int(ghostPos[1]))
    
    if xDecimal > 0 or yDecimal >0:
        a =ghostPos[0] + xDecimal
        b = ghostPos[1] + yDecimal
        ghostPos = (a, b)
        distance = astarMazeDistBetweenTwoPoints(currentGameState,pacPos, ghostPos)/2   
    else:
        distance = astarMazeDistBetweenTwoPoints(currentGameState,pacPos, allghosts[index].getPosition())/2
    if distance < shortestDistanceToDisabledGhost:
        shortestDistanceToDisabledGhost = distance

   
  shortestDistanceToFood = sys.maxint
  foodList = food.asList()
  foodAndCapsulesList = foodList+capsulesList
  
  for foodPiece in foodAndCapsulesList:
    distance = astarMazeDistBetweenTwoPoints(currentGameState,pacPos,foodPiece)
    if distance < shortestDistanceToFood:
      shortestDistanceToFood = distance
      
  if len(foodAndCapsulesList)==0:
    return sys.maxint
      
                    
  if noneDisabled:
    #print "ghosts not scared"
    #return 2*shortestDistanceToActiveGhost-shortestDistanceToFood
    return scoreEvaluationFunction(currentGameState) - shortestDistanceToFood
    
  if not noneDisabled:
    #print "ghosts scared!!"
    #return shortestDistanceToActiveGhost+0.5*shortestDistanceToFood-1.5*shortestDistanceToDisabledGhost
    return 2*scoreEvaluationFunction(currentGameState)-shortestDistanceToDisabledGhost
    

def astarMazeDistBetweenTwoPoints(currentGameState, pos1, pos2):
  from game import Directions
  from util import PriorityQueue

  pos1dub=(float(pos1[0]),float(pos1[1]))
  pos2dub=(float(pos2[0]),float(pos2[1]))
  
  x=pos1dub
  explored = set([x])
  frontier=PriorityQueue()
  path=[]
  pathcost=0
  frontier.push((x,path,pathcost),pathcost)
  
  while True:

    activeNode=frontier.pop()
    #activeNode[0] is the location of the node, activeNode[1] is the path from active node back to the start, activeNode[2] is the pathcost of path
    x=activeNode[0]
    #print "pathcost of node just popped: ",activeNode[2]

    
    if x[0]==pos2dub[0] and x[1]==pos2dub[1]:  #goal test
      #print "RETURNING: ",x,pos2dub
      return activeNode[2]

    succlist=getSuccessors(x,currentGameState) #is a list of triples
    explored.add(x)

    for i in succlist:      #i is a triple, i[0] is the location, i[1] is direction, i[2] is the cost
      path=activeNode[1]+[i[1]]
      pathcost=activeNode[2]+i[2]
      estpathcost=pathcost+heuristicManDis(i[0],pos2)
      izfloat=(float(i[0][0]),float(i[0][1]))
      if izfloat not in explored:
        frontier.push((izfloat,path,pathcost),estpathcost)

def getSuccessors(state,currentGameState):
  from game import Actions
  "Returns successor states, the actions they require, and a cost of 1."
  walls = currentGameState.getWalls()
  successors = []
  for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
    x,y = state
    dx, dy = Actions.directionToVector(direction)
    nextx, nexty = int(x + dx), int(y + dy)
    if not walls[nextx][nexty]:
      successors.append( ( (nextx, nexty), direction, 1) )
  return successors

def heuristicManDis(virtualpos,pos2):
  return abs(virtualpos[0] - pos2[0]) + abs(virtualpos[1] - pos2[1])

  
        
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
