# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        cloest_unscared_ghost = float("inf")
        for ghosts in newGhostStates:
            danger_x, danger_y = ghosts.getPosition()
            if ghosts.scaredTimer == 0:
                if newPos[0] == danger_x and newPos[1] == danger_y:
                    return -float("inf")
                cloest_unscared_ghost = min(
                    cloest_unscared_ghost, manhattanDistance((danger_x, danger_y), newPos))

        closest_food_dis = float("inf")
        if not newFood:
            closest_food_dis = 0
        for food in newFood.asList():
            closest_food_dis = min(
                closest_food_dis, manhattanDistance(food, newPos))

        # adding weight to the evasion
        return successorGameState.getScore() - 2/(cloest_unscared_ghost + 1) + 4/closest_food_dis

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def search_move(gameState, agent_index, depth):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState), Directions.STOP
            if agent_index == 0:
                res_act = max_value_player(gameState, agent_index, depth)
            elif agent_index > 0:
                res_act = min_value_player(gameState, agent_index, depth)
            return res_act

        def max_value_player(gameState, agent_index, depth):
            all_pos_actions = gameState.getLegalActions(agent_index)

            if agent_index == gameState.getNumAgents() - 1:
                next_agent, depth = 0, depth - 1
            else:
                next_agent = agent_index + 1

            score, curr_action = -float("inf"), None

            for action in all_pos_actions:
                all_success = gameState.generateSuccessor(agent_index, action)
                new_score = search_move(all_success, next_agent, depth)[0]
                if new_score > score:
                    score, curr_action = new_score, action

            return score, curr_action

        def min_value_player(gameState, agent_index, depth):
            all_pos_actions = gameState.getLegalActions(agent_index)

            if agent_index == gameState.getNumAgents() - 1:
                next_agent = 0
                depth -= 1

            else:
                next_agent = agent_index + 1

            score, curr_action = float("inf"), None

            for action in all_pos_actions:
                all_success = gameState.generateSuccessor(agent_index, action)
                new_score = search_move(all_success, next_agent, depth)[0]
                if new_score < score:
                    score, curr_action = new_score, action
            return score, curr_action

        return search_move(gameState, 0, self.depth)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta_search(gameState, agent_index, beta, alpha, depth):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return self.evaluationFunction(gameState), Directions.STOP
            if agent_index == 0:
                res_act = alpha_agent(
                    gameState, agent_index, beta, alpha,  depth)
            elif agent_index > 0:
                res_act = beta_agent(
                    gameState, agent_index, beta, alpha, depth)
            return res_act

        def alpha_agent(gameState, agent_index, beta, alpha,  depth):
            all_pos_actions = gameState.getLegalActions(agent_index)

            if agent_index == gameState.getNumAgents() - 1:
                next_agent, depth = 0, depth - 1
            else:
                next_agent = agent_index + 1

            score, curr_action = -float("inf"), None

            for action in all_pos_actions:
                all_success = gameState.generateSuccessor(agent_index, action)
                new_score = alpha_beta_search(
                    all_success, next_agent, beta, alpha, depth)[0]
                if new_score > score:
                    score, curr_action = new_score, action
                if new_score >= beta:
                    return new_score, action
                alpha = max(alpha, new_score)

            return score, curr_action

        def beta_agent(gameState, agent_index, beta, alpha, depth):
            all_pos_actions = gameState.getLegalActions(agent_index)

            if agent_index == gameState.getNumAgents() - 1:
                next_agent = 0
                depth -= 1

            else:
                next_agent = agent_index + 1

            score, curr_action = float("inf"), None

            for action in all_pos_actions:
                all_success = gameState.generateSuccessor(agent_index, action)
                new_score = alpha_beta_search(
                    all_success, next_agent, beta, alpha, depth)[0]
                if new_score < score:
                    score, curr_action = new_score, action
                if new_score < alpha:
                    return new_score, action
                beta = min(beta, new_score)

            return score, curr_action

        return alpha_beta_search(gameState, 0,  float("inf"), -float("inf"), self.depth)[1]



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
        best = Directions.STOP
        highest_score = -float("inf")
        for moves in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, moves)
            curr_score = max(highest_score, self.expectimax(state, 1, self.depth))
            if highest_score < curr_score:
                highest_score = curr_score
                best = moves
        return best
    def maxv(self, gameState, current_agent, depth, depth_left, agent_num):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -float("inf")
        for moves in gameState.getLegalActions(current_agent):
            succ_state = gameState.generateSuccessor(current_agent, moves)
            curr_score = self.expectimax(succ_state, agent_num, depth_left)
            if v < curr_score:
                v = curr_score
        return v
    def expectimax(self, gameState, current_agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        agent_num = current_agent + 1
        depth_left = depth
        v = 0
        if gameState.getNumAgents() <= agent_num:
            depth_left -= 1
            agent_num = 0
        if current_agent == 0:
            v = self.maxv(gameState, current_agent, depth, depth_left, agent_num)
        else:
            total = 0
            for moves in gameState.getLegalActions(current_agent):
                succ_state = gameState.generateSuccessor(current_agent, moves)
                total += self.expectimax(succ_state, agent_num, depth_left)
            v = total/len(gameState.getLegalActions(current_agent))
        return v
  
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
