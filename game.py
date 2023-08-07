import numpy as np
import logging

class Game:
    
    def __init__(self, board = []):        
        self.currentPlayer = 1
        self.grid_shape = (5,5)
        self.input_shape = (2, 5, 5)
        self.b = np.array([0 for i in range(25)], dtype=np.int32)
        self.gameState = GameState(self.b, 1)
        self.actionSpace = np.array([0 for i in range(25)], dtype=np.int32)
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

                                   
        currentBoard = state.board
        currentAV = actionValues
        
        #exploit symmetry for faster learning
        currentBoard = np.array([
			currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1], currentBoard[0],
			currentBoard[9], currentBoard[8], currentBoard[7], currentBoard[6], currentBoard[5],
			currentBoard[14], currentBoard[13], currentBoard[12], currentBoard[11], currentBoard[10],
			currentBoard[19], currentBoard[18], currentBoard[17], currentBoard[16], currentBoard[15],
			currentBoard[24], currentBoard[23], currentBoard[22], currentBoard[21], currentBoard[20]
			])

        currentAV =  np.array([
			currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0],
			currentAV[9], currentAV[8], currentAV[7], currentAV[6], currentAV[5],
			currentAV[14], currentAV[13], currentAV[12], currentAV[11], currentAV[10],
			currentAV[19], currentAV[18], currentAV[17], currentAV[16], currentAV[15],
			currentAV[24], currentAV[23], currentAV[22], currentAV[21], currentAV[20]
			])


        identities.append((GameState(currentBoard, state.playerTurn), currentAV))

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
        allowed = np.where(self.board == 0)[0]
        return allowed

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int32)
        currentplayer_position[self.board==self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int32)
        other_position[self.board==-self.playerTurn] = 1
        
        position = np.concatenate((currentplayer_position,other_position))

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int32)
        player1_position[self.board==1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int32)
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
            scores = self._getBoardScore()
            p1winning = 1 if scores[0]-scores[1] > 0 else -1
            return (p1winning*self.playerTurn, p1winning*self.playerTurn, 1)
        return (0,0, 0)

    def _getBoardScore(self):
        ''' This returns the value of the current board position'''
        p1score = 0
        p2score = 0
        for h in range(0,self.grid_shape[0]):
            for w in range(0, self.grid_shape[1]):
                calcBoard = self.board.reshape(self.grid_shape[0], self.grid_shape[1])
                matchSymbol = calcBoard[h, w]
                if matchSymbol == 1:
                    scores = self.rewardFromSpot(calcBoard, 1, h, w)
                    p1score += scores
                elif matchSymbol == 2:
                    scores = self.rewardFromSpot(calcBoard, 2, h, w)
                    p2score += scores
        return (p1score, p2score)

    def _getScore(self):
        temp = self._getValue()
        return (temp[1], temp[2])
        
    def rewardFromSpot(self, calcBoard, matchSymbol, h, w):
        scores = 0
        
        #vertical check
        if h < self.grid_shape[0]-2:
            scores += 1 if calcBoard[h+1, w] == matchSymbol and calcBoard[h+2,w] == matchSymbol else 0
        #horizontal check
        if w < self.grid_shape[1]-2:
            scores += 1 if calcBoard[h, w+1] == matchSymbol and calcBoard[h,w+2] == matchSymbol else 0
        #diagonal down right
        if w < self.grid_shape[1]-2 and h < self.grid_shape[0]-2:
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