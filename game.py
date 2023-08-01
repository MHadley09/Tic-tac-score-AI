import numpy as np
import logging

class Game:
    
    def __init__(self, board = []):        
        self.currentPlayer = 1
        self.grid_shape = (5,5)
        self.input_shape = (2,5,5)
        self.b = np.array([0 for i in range(25)], dtype=np.int)
        self.gameState = GameState(self.b, 1)
        self.actionSpace = np.array([0 for i in range(25)], dtype=np.int)
        self.pieces = {'1':'O', '0': '-', '-1':'X'}
        self.name = 'tic-tac-score'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(self.b, 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state,actionValues)]

                                   
        currentBoard = state.board.reshape(self.grid_shape)
        currentAV = actionValues.reshape(self.grid_shape)

        #learn faster by accounting for symmetry

        identities.append((GameState(np.flip(currentBoard,0).flatten(), state.playerTurn), np.flip(currentAV,0).flatten()))
        identities.append((GameState(np.flip(currentBoard,1).flatten(), state.playerTurn), np.flip(currentAV,1).flatten()))
        identities.append((GameState(np.flip(currentBoard,(0,1)).flatten(), state.playerTurn), np.flip(currentAV,(0,1)).flatten()))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.grid_shape = (5,5)
        self.winlength = 3
        self.pieces = {'1':'O', '0': ' ', '-1':'X'}
        self.playerTurn = playerTurn

        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        # only those empty cells can be filled
        return np.where(self.board == 0)[0]

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board==self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-self.playerTurn] = 1
        
        position = np.concatenate((currentplayer_position,other_position))

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board==1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-1] = 1
        
        position = np.concatenate((player1_position,other_position))

        id = ''.join(map(str,position))

        return id

    def _checkForEndGame(self):
        return 1 if len(self.allowedActions) == 1 else 0

    def _getValue(self):
        # This is the value of the state for the current player
        # if there is only 1 move left calculate who wins
        if len(self.allowedActions) == 1:
            scores = _getScore(self)
            p1winning = 1 if score[0]-scores[1] > 0 else -1
            return p1winning*self.playerTurn
        return 0

    def _getScore(self):
        ''' This returns the value of the current board position'''
        p1score = 0
        p2score = 0
        for h in range(0,self.grid_shape[0]):
            for w in range(0, self.grid_shape[1]):
                calcBoard = self.board.reshape(self.grid_shape[0], self.grid_shape[1])
                matchSymbol = calcBoard[h, w]
                if matchSymbol == 1:
                    scores = rewardFromSpot(calcBoard, 1, h, w)
                    p1score += scores
                elif matchSymbol == 2:
                    scores = rewardFromSpot(calcBoard, 2, h, w)
                    p2score += scores
        return (p1score, p2score)

    def rewardFromSpot(calcBoard, matchSymbol, h, w):
        scores = 0
        
        #vertical check
        if h < self.grid_shape[0]-2:
            scores += 1 if calcBoard[h+1, w] == matchSymbol and calcBoard[h+2,w] == matchSymbol else 0
        #horizontal check
        if w < self.grid_shape[1]-2:
            scores += 1 if calcBoard[h, w+1] == matchSymbol and calcBoard[h,w+2] == matchSymbol else 0
        #diagonal down right
        if w < self.grid_shape[1]-2 and h < state.shape[0]-2:
            scores += 1 if calcBoard[h+1, w+1] == matchSymbol and calcBoard[h+2,w+2] == matchSymbol else 0
        if w > 2 and h < self.grid_shape[0]-2:
            scores += 1 if calcBoard[h+1, w-1] == matchSymbol and calcBoard[h+2,w-2] == matchSymbol else 0

        return scores

    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action]=self.playerTurn
        
        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done) 


    def printgame(self):
        l='ABCDEFGHIJKLMNOPQRSTUVWYZ'
        for r in range(self.grid_shape[0]):
            print(l[r],*[self.pieces[str(x)] for x in self.board[self.grid_shape[1]*r : 
                (self.grid_shape[1]*r + self.grid_shape[1])]],sep='|')
        print(' ',*['__' for i in range(1,8)],sep='')
        print(' ',*[str(i) for i in range(1,8)],sep='|')
        
    def render(self, logger):
        for r in range(self.grid_shape[0]):
            logger.info([self.pieces[str(x)] for x in self.board[self.grid_shape[1]*r : 
                (self.grid_shape[1]*r + self.grid_shape[1])]])
        logger.info(['_' for i in range(1,8)])
        logger.info([str(i) for i in range(1,8)])