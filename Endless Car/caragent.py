import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from car import CARGAME
from plot import Plot

class CARAI(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, 100)
        self.linear2 = nn.Linear(100, 200)
        self.linear4 = nn.Linear(200, outputSize)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.log_softmax(self.linear4(x), dim=-1)
        return x

    def save(self, modelName='carAIModel.pth'):
        torch.save(self.state_dict(), modelName)


class DRLTrainer:
    def __init__(self, gamma, model, lr):
        self.learningRate = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def train(self, currentState, nextState, direction, reward, gameOver):
        currentState, nextState, direction, reward = self.putToGPU(currentState, nextState, direction, reward)

        if len(currentState.shape) == 1:
            currentState, nextState, direction, reward, gameOver = self.unsqueezeData(currentState, nextState, direction, reward, gameOver)

        pred = self.model(currentState)

        target = pred.clone()
        for idx in range(len(gameOver)):
            newQ = reward[idx]
            if not gameOver[idx]:
                newQ = reward[idx] + self.gamma * torch.max(self.model(nextState))
            
            i = torch.argmax(direction[idx]).item()
            target[idx][i] = newQ
    
        self.backPropogation(target, pred)

    def putToGPU(self, currentState, nextState, direction, reward):
        gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.tensor(currentState, dtype=torch.float).to(gpu), torch.tensor(nextState, dtype=torch.float).to(gpu), torch.tensor(direction, dtype=torch.long).to(gpu), torch.tensor(reward, dtype=torch.float).to(gpu)

    def unsqueezeData(self, currentState, nextState, direction, reward, gameOver):
        return torch.unsqueeze(currentState, 0), torch.unsqueeze(nextState, 0), torch.unsqueeze(direction, 0), torch.unsqueeze(reward, 0), (gameOver, ) 

    def backPropogation(self, target, pred):
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        return

class Agent:

    def __init__(self):
        self.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.brain = CARAI(18, 3).to(self.gpu)
        self.numberOfGames = 0
        self.epsilon = 0
        self.gameEpisodes = deque(maxlen=100000)
        self.gamma = 0.9 
        self.drlTrainer = DRLTrainer(gamma=self.gamma, model=self.brain, lr=0.0001)

    def storeGameEnvironment(self, currentState, nextState, action, reward, gameOver):
        self.gameEpisodes.append((currentState, nextState, action, reward, gameOver))

    def trainAction(self, currentState, nextState, action, reward, gameOver):
        currentState, nextState, action, reward = np.array(currentState), np.array(nextState), np.array(action), np.array(reward)
        self.drlTrainer.train(currentState, nextState, action, reward, gameOver)

    def getGameState(self, game: CARGAME):
        fieldOfView = np.zeros((18), dtype= int)

        carRect = game.carRect.copy()
        leftRect, rightRect, obstacleRect, collectableRect = game.track[0]

        carRect.left -= game.blockSize
        fieldOfView[0] = 20 if carRect.colliderect(leftRect) else 4
        if obstacleRect is not None:
            fieldOfView[5] = 30 if carRect.colliderect(obstacleRect) else 6
        if collectableRect is not None:
            fieldOfView[10] = 40 if carRect.colliderect(collectableRect) else 8

        carRect.left += game.blockSize + game.blockSize #+ game.blockSize
        fieldOfView[1] = 20 if carRect.colliderect(rightRect) else 4
        if obstacleRect is not None:
            fieldOfView[6] = 30 if carRect.colliderect(obstacleRect) else 6
        if collectableRect is not None:
            fieldOfView[11] = 40 if carRect.colliderect(collectableRect) else 8

        leftRect, rightRect, obstacleRect, collectableRect = game.track[1]

        carRect.left -= game.blockSize - game.blockSize #+ game.blockSize
        carRect.bottom -= game.blockSize
        fieldOfView[2] = 20 if carRect.colliderect(leftRect) else 4
        if obstacleRect is not None:
            fieldOfView[7] = 30 if carRect.colliderect(obstacleRect) else 6
        if collectableRect is not None:
            fieldOfView[12] = 40 if carRect.colliderect(collectableRect) else 8

        carRect.left += game.blockSize
        fieldOfView[3] = 20 if carRect.colliderect(leftRect) or carRect.colliderect(rightRect) else 4
        if obstacleRect is not None:
            fieldOfView[8] = 30 if carRect.colliderect(obstacleRect) else 6
        if collectableRect is not None:
            fieldOfView[13] = 40 if carRect.colliderect(collectableRect) else 8

        carRect.left += game.blockSize
        fieldOfView[4] = 20 if carRect.colliderect(rightRect) else 4
        if obstacleRect is not None:
            fieldOfView[9] = 30 if carRect.colliderect(obstacleRect) else 6
        if collectableRect is not None:
            fieldOfView[14] = 40 if carRect.colliderect(collectableRect) else 8

        fieldOfView[15] = 20 if game.direction == [1,0,0] else 4
        fieldOfView[16] = 20 if game.direction == [0,1,0] else 4
        fieldOfView[17] = 20 if game.direction == [0,0,1] else 4

        return fieldOfView

    def trainEpisode(self):
        if len(self.gameEpisodes) > 1000:
            episode = random.sample(self.gameEpisodes, 1000)
        else:
            episode = self.gameEpisodes

        currentStates, nextStates, actions, rewards, gameOvers = zip(*episode)
        currentStates, nextStates, actions, rewards = np.array(currentStates), np.array(nextStates), np.array(actions), np.array(rewards)
        self.drlTrainer.train(currentStates, nextStates, actions, rewards, gameOvers)

    def getAction(self, state):
        finalAction = [0,0,0]
        stateTensor = torch.tensor(state, dtype=torch.float)
        stateTensor = stateTensor.to(self.gpu)
        predictedAction = self.brain(stateTensor)
        finalAction[torch.argmax(predictedAction).item()] = 1

        return finalAction


def train():

    scores = []
    scoreMean = []
    scoreTotal = 0
    record = 0
    agent = Agent()
    carGame = CARGAME()
    graph = Plot()
    random.seed(383)
    while True:

        oldState = agent.getGameState(carGame)
        finalAction = agent.getAction(oldState)
        gameOver, score, reward = carGame.performAction(finalAction)
        newState = agent.getGameState(carGame)
        agent.trainAction(oldState, newState, finalAction, reward, gameOver)
        agent.storeGameEnvironment(oldState, newState, finalAction, reward, gameOver)

        if gameOver:
            carGame.resetGame()
            agent.numberOfGames += 1
            agent.trainEpisode()

            if score > record:
                record = score
                agent.brain.save()

            print('Game', agent.numberOfGames, 'Score', score, 'Record:', record)

            scores.append(score)
            scoreTotal += score
            scoreMean.append(scoreTotal / agent.numberOfGames)

            graph.plot('Training Endless Car Agent', scores, scoreMean)



if __name__ == '__main__':
    train()