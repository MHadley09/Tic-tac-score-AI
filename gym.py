import numpy as np
import collections
import math, random
import time
from copy import deepcopy

### Define parameters here
BOARDSIZE = 5
BOARDHEIGHT = 5
WINLENGTH = 3
        
class gym:
    def __init__(self, **kwargs):
        '''
        Data structure to represent state of the environment
        self.state : state of board
        '''
        # default settings
        self.boardsize = kwargs.get('boardsize', BOARDSIZE)
        self.boardheight = kwargs.get('boardheight', BOARDHEIGHT)
        self.mapping = {0: ' ', 1: 'O', 2: 'X'}
        self.display = kwargs.get('display', 'tensor')
        self.state = kwargs.get('state', np.zeros((self.boardheight, self.boardsize), dtype = int))
        self.initialstate = deepcopy(self.state)
        self.winlength = kwargs.get('winlength', WINLENGTH)
        self.turn = kwargs.get('turn', 1)
        self.reward = kwargs.get('reward', 0)
        self.done = kwargs.get('done', False)
        self.moves = [i*self.boardsize+j for i in range(0, self.boardheight) for j in range(0, self.boardsize) if self.state[i, j] == 0]
        
    def step(self, action):
        '''
        Takes action at self.state and returns the next step
        '''
        # check to see if action can be taken

        if(action not in self.moves):
            print('Invalid move')
            return
            
        # remove current action
        self.moves.remove(action)

        # play the move at the row and col
        h, w = action//self.boardsize, action%self.boardsize
        self.state[h, w] = self.turn
    
        # check if anyone won based on just that move
        winner = 0

        if len(self.moves) == 1:
            self.done = True
            winner = self.checkwin()

        if winner != 0:
            self.done = True
            #player 1 is rewarded and game always ends on player 2s turn
            self.reward = 1 if winner > 0  else -1

        # if no available moves left, then it is a draw

        # update turn
        self.turn = 1 if self.turn == 2 else 2
        
    def sample(self):
        return(np.random.choice(self.moves))

    def doubleStep(self, action, minimax_depth = 2, p = 0.5):
        # Initialize state
        self.step(action)

        # if not ended yet, chance% probability to play a game with a minimax agent
        # otherwise, random action
        
        if not self.done:
            if np.random.rand() < p:
                action = minimax_agent(self, depth = minimax_depth)
                self.step(action)
            else:
                self.step(self.sample())

    def checkwin(self):
        p1score = 0
        p2score = 0

        for h in range(0,self.boardsize):
            for w in range(0, self.boardsize):
                board = self.state.reshape(self.boardsize, self.boardheight)
                matchSymbol = self.state[h, w]
                if matchSymbol == 1:
                    scores = self.scoresFromSpot(board, matchSymbol, h, w)
                    p1score += scores
                elif matchSymbol == 2:
                    scores = self.scoresFromSpot(board, matchSymbol, h, w)
                    p2score += scores
        if p1score == p2score:
            return 0
        return 1 if p1score > p2score else -1

    def scoresFromSpot(self, board, matchSymbol, h, w):
        scores = 0
        
        #vertical check
        if h < self.boardheight-2:
            scores += 1 if board[h+1, w] == matchSymbol and board[h+2,w] == matchSymbol else 0
        #horizontal check
        if w < self.boardsize-2:
            scores += 1 if board[h, w+1] == matchSymbol and board[h,w+2] == matchSymbol else 0
        #diagonal down right
        if w < self.boardsize-2 and h < self.boardsize-2:
            scores += 1 if board[h+1, w+1] == matchSymbol and board[h+2,w+2] == matchSymbol else 0
        if w > 2 and h < self.boardsize-2:
            scores += 1 if board[h+1, w-1] == matchSymbol and board[h+2,w-2] == matchSymbol else 0
        
        return scores

    def reset(self):
        self.state = deepcopy(self.initialstate)
        self.moves = [i*self.boardsize+j for i in range(0, self.boardheight) for j in range(0, self.boardsize) if self.state[i, j] == 0]
        self.turn = 1
        self.reward = 0
        self.done = False

    def printboard(self):
        rowheading = "ABCDEFG"
        rowindex = 0
        
        for row in self.state:
            print(rowheading[rowindex], end = '')
            rowindex += 1
            
            print('|', end = '')
            for index in row:
                print(self.mapping[index], end = '|')
            print()
            
        # bottom border
        print(' ', end = '')
        print('__'*(self.boardsize+1))

        # bottom index
        print(' ', end = '')
        print('|', end = '')
        for num in range(self.boardsize):
            print(num+1, end = '|')
        print()
        
    def get_tensor_state(self):
        ''' This returns the state in tensor form for deep learning methods '''
        tensor = np.zeros((3, self.boardheight, self.boardsize))
        for i in range(self.boardheight):
            for j in range(self.boardsize):
                # first 2D grid is for turn
                tensor [0, i, j] = 1 if self.turn == 1 else -1
                # next 2D grid is for O
                if self.state[i, j] == 1:
                    tensor[1, i, j] = 1
                # next 2D grid is for X
                elif self.state[i, j] == 2:
                    tensor[2, i, j] = 1
                # next 2D grid is for B
        return tensor
    
    def get_alphazero_state(self):
        ''' This returns the state for the alphazero method to use '''
        newstate = self.state.flatten()
        for i in range(len(newstate)):
            if newstate[i]==1:
                newstate[i] = 1
            elif newstate[i]==2:
                newstate[i] = -1

        return newstate
        
    def get_state(self):
        return self.state, self.turn, self.reward, self.done
        
    def get_env(self):
        return gym(state = deepcopy(self.state), turn = self.turn, reward = self.reward, done = self.done)
    
        
#### Monte Carlo Agent here (default agent)

### Define parameters here
random_rollout_iter = 1
exploration_param = 1

def randomPolicy(myenv):
    '''
    Random rollouts in the environment
    '''
    
    # if terminal state, return reward
    if myenv.done == True:
        return myenv.reward

    # else play until terminal state, return its final reward
    reward = 0

    # do random rollouts for random_rollout_iter number of times
    for _ in range(random_rollout_iter):
        # create a copy of the environment
        env = myenv.get_env()
        
        while not env.done:
            env.step(env.sample())

        reward += env.reward
    
    return reward/random_rollout_iter
    
def heuristicPolicy(env):
    '''
    Heuristic value approximation using the expert heuristic instead of rollouts
    '''
    
    # if terminal state, return reward
    if env.done == True:
        return env.reward
        
    # else do heuristic evaluation (no need random rollouts)
    else:
        value = evaluate(env.state)
#        print(value)
#        print((np.log10(abs(value)+1e-6)))
#        print((4+np.log10(abs(value)+1e-6))/4 * np.sign(value))
        return (6+np.log10(abs(value)+1e-6))/6 * np.sign(value)

class Node:
    def __init__(self, parent, env):
        '''
        Data structure for a node of the MCTS tree
        self.env : gym environment represented by the node
        self.parent : Parent of the node in the MCTS tree
        self.numVisits : Number of times the node has been visited
        self.totalReward : Sum of all rewards backpropagated to the node
        self.allChildrenAdded : Denotes whether all actions from the node have been explored
        self.children : Set of children of the node in the MCTS tree
        '''

        # state parameters
        self.env = env

        # parameters for the rest
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0

        # if the state is a terminal state, by default all children are added
        self.allChildrenAdded = self.env.done
        self.children = {}
        # children of the form "action: Node()"
        
class MonteCarloTreeSearch:
    def __init__(self, num_iter, explorationParam, playoutPolicy=randomPolicy, random_seed=None):
        '''
        self.num_iter : Number of iterations to play out
        self.explorationParam : exploration constant used in computing value of node
        self.playoutPolicy : Policy followed by agent to simulate rollout from leaf node
        self.root : root node of MCTS tree
        '''
        self.num_iter = num_iter
        self.explorationParam = explorationParam
#        self.playoutPolicy = playoutPolicy
        self.playoutPolicy = heuristicPolicy
        self.root = None

    def buildTreeAndReturnBestAction(self, env):
        '''
        Function to build MCTS tree and return best action at initialState
        '''
        self.root = Node(parent=None, env = env)

#        timeout_start = time.time()
#        numiter = 0
#
#        while time.time() < timeout_start + self.timeout:
#            self.addNodeAndBackpropagate()
#            numiter += 1

#        print('Numiter:', numiter)

        for i in range(self.num_iter):
            self.addNodeAndBackpropagate()

        # return action with highest value
   
        values = np.full(self.root.env.state.shape[0]*self.root.env.state.shape[1], -2, dtype = float)
        numvisits = np.full(self.root.env.state.shape[0]*self.root.env.state.shape[1], -2, dtype = int)

        for action, cur_node in self.root.children.items():
            values[action] = (cur_node.totalReward/cur_node.numVisits)
            numvisits[action] = cur_node.numVisits

#         print('Values')
#         for i in range(BOARDHEIGHT):
#             print(np.round(values[i*BOARDSIZE: (i+1)*BOARDSIZE], 3))
#         print ('Num visits')
#         for i in range(BOARDHEIGHT):
#             print(np.round(numvisits[i*BOARDSIZE: (i+1)*BOARDSIZE], 0))

#        return np.argmax(values)
        return np.argmax(numvisits)
        
    def addNodeAndBackpropagate(self):
        '''
        Function to run a single MCTS iteration
        '''
        node = self.addNode()
        reward = self.playoutPolicy(node.env)
        self.backpropagate(node, reward)

    def addNode(self):
        '''
        Function to add a node to the MCTS tree
        '''
        cur_node = self.root
        while not cur_node.env.done:
        # this is to check if the current node is a leaf node
            if cur_node.allChildrenAdded:
                cur_node = self.chooseBestActionNode(cur_node, self.explorationParam)
            else:
                actions = cur_node.env.moves
                for action in actions:
                    if action not in cur_node.children:
                        new_env = cur_node.env.get_env()
                        new_env.step(action)
                        newNode = Node(parent=cur_node, env = new_env)
                        cur_node.children[action] = newNode
                        if len(actions) == len(cur_node.children):
                            cur_node.allChildrenAdded = True
                        return newNode
        return cur_node
                
    def backpropagate(self, node, reward):
        '''
        This function implements the backpropation step of MCTS.
        '''
        # adds rewards to all nodes along the path (upwards to parent)
        while (node is not None):

            if node.env.turn == 1:
                node.totalReward -= reward
            else:
                node.totalReward += reward

            # add counter to node
            node.numVisits += 1
            # go up one level
            node = node.parent
        

    def chooseBestActionNode(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        
        for action, child in enumerate(node.children.values()):
            
            '''
            Populate the list bestNodes with all children having maximum value
                       
            Value of all nodes should be computed as mentioned in question 3(b).
            All the nodes that have the largest value should be included in the list bestNodes.
            We will then choose one of the nodes in this list at random as the best action node.
            '''
            
            value = child.totalReward/child.numVisits + explorationValue * math.sqrt(math.log(node.numVisits)/child.numVisits)

            # if this value is higher, set this to be new bestValue, bestNode will be current child node
            if value > bestValue:
                bestValue = value
                bestNodes = [child]
            
            # if the value ties with current best value, then current child node will be appended to the list of best nodes
            elif value == bestValue:
                bestNodes.append(child)
            
        return np.random.choice(bestNodes)

    ### MCTS agent here ###
        
def mcts_agent(env = None, num_iter = 2500):
    '''
    Input: environment

    Output: MCTS action
    '''
    
    # hard-set this
    num_iter = 2500
    
    # create a local copy
    env = env.get_env()
    
    # Do the Monte Carlo Policy prediction for this step, without changing the environment
    mcts = MonteCarloTreeSearch(num_iter = num_iter, explorationParam=exploration_param, random_seed=42)
    action = mcts.buildTreeAndReturnBestAction(env)
    
    return action
    
    ### Random agent here ###
    
def random_agent(env = None):
    '''
    Input: environment
    Output: random action
    '''
    
    return env.sample()
    
    ### Minimax agent here ###
    
def minimax_agent(env = None, depth = 2):
    '''
    Input: environment
    Output: Minimax action
    '''

    env = env.get_env()
    
    if env.moves == []:
        print("No moves to choose from in environment")
        return None
        
    maxreward = -10000 if env.turn == 1 else 10000
    bestaction = []
    
    for action in env.moves:
        new_env = env.get_env()
        new_env.step(action)
        reward = minimax(new_env, alpha = -10000, beta = 10000, depth = depth-1)
        if reward == maxreward:
            bestaction.append(action)
        if (env.turn == 1 and reward > maxreward) or (env.turn == 2 and reward < maxreward):
            maxreward = reward
            bestaction = [action]

    # this is a fail-safe in case minimax does not return any action
    if bestaction == []:
        print("No best action, choosing the first action instead")
        return env.moves[0]
    
    return np.random.choice(bestaction)
    # return bestaction[0]
        
def minimax(env, alpha, beta, depth = 2):
    if (env.done):
        return env.reward
    
    if(depth == 0):
        return evaluate(env.state)

    # alpha beta pruning
    if alpha >= beta:
        if env.turn == 1:
            return alpha
        else:
            return beta
        
    maxreward = -10000 if env.turn == 1 else 10000
    
    for action in env.moves:
        new_env = env.get_env()
        new_env.step(action)
#        print('Player {}: reward {}'.format(state.turn, reward))
        reward = minimax(new_env, alpha = alpha, beta = beta, depth = depth-1)
        if (env.turn == 1 and reward >= maxreward) or (env.turn == 2 and reward <= maxreward):
            maxreward = reward
            
        # update alpha and beta
        if env.turn == 1 and maxreward > alpha:
            alpha = maxreward
        if env.turn == 2 and maxreward < beta:
            beta = maxreward
            
    return maxreward
    
### Helper function to evaluate the state of a board position heuristically
def evaluate(state):
    ''' This returns the value of the current board position'''
    p1score = 0
    p2score = 0
    for h in range(0,state.shape[0]):
        for w in range(0, state.shape[1]):
            matchSymbol = state[h, w]
            if matchSymbol == 1:
                scores = rewardFromSpot(state, 1, h, w)
                p1score += scores
            elif matchSymbol == 2:
                scores = rewardFromSpot(state, 2, h, w)
                p2score += scores
    return p1score-p2score

def rewardFromSpot(state, matchSymbol, h, w):
    scores = 0
    
    #vertical check
    if h < state.shape[0]-2:
        scores += 1 if state[h+1, w] == matchSymbol and state[h+2,w] == matchSymbol else 0
    #horizontal check
    if w < state.shape[1]-2:
        scores += 1 if state[h, w+1] == matchSymbol and state[h,w+2] == matchSymbol else 0
    #diagonal down right
    if w < state.shape[1]-2 and h < state.shape[0]-2:
        scores += 1 if state[h+1, w+1] == matchSymbol and state[h+2,w+2] == matchSymbol else 0
    if w > 2 and h < state.shape[0]-2:
        scores += 1 if state[h+1, w-1] == matchSymbol and state[h+2,w-2] == matchSymbol else 0

    return scores