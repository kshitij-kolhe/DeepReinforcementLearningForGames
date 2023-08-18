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
    def __init__(self, model, lr, gamma):
        self.learningRate = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def train_step(self, currentState, nextState, direction, reward, gameOver):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        currentState = torch.tensor(currentState, dtype=torch.float).to(device)
        nextState = torch.tensor(nextState, dtype=torch.float).to(device)
        direction = torch.tensor(direction, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        if len(currentState.shape) == 1:
            currentState = torch.unsqueeze(currentState, 0)
            nextState = torch.unsqueeze(nextState, 0)
            direction = torch.unsqueeze(direction, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        pred = self.model(currentState)

        target = pred.clone()
        for idx in range(len(gameOver)):
            newQ = reward[idx]
            if not gameOver[idx]:
                newQ = reward[idx] + self.gamma * torch.max(self.model(nextState))
            
            i = torch.argmax(direction[idx]).item()
            target[idx][i] = newQ
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class Agent:

    def __init__(self):
        self.numberOfGames = 0
        self.epsilon = 0
        self.gamma = 0.9 
        self.memory = deque(maxlen=100000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CARAI(18, 3).to(self.device)
        self.trainer = DRLTrainer(self.model, lr=0.0001, gamma=self.gamma)


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

    def storeGameEnvironment(self, currentState, nextState, action, reward, gameOver):
        self.memory.append((currentState, nextState, action, reward, gameOver))

    def trainEpisode(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        currentStates, nextStates, actions, rewards, gameOvers = zip(*mini_sample)
        currentStates = np.array(currentStates)
        nextStates = np.array(nextStates)
        actions = np.array(actions)
        rewards = np.array(rewards)

        self.trainer.train_step(currentStates, nextStates, actions, rewards, gameOvers)

    def trainAction(self, currentState, nextState, action, reward, gameOver):
        currentState = np.array(currentState)
        nextState = np.array(nextState)
        action = np.array(action)
        reward = np.array(reward)

        self.trainer.train_step(currentState, nextState, action, reward, gameOver)

    def getAction(self, state):
        finalAction = [0,0,0]
        stateTensor = torch.tensor(state, dtype=torch.float)
        stateTensor = torch.unsqueeze(stateTensor, 0)
        stateTensor = stateTensor.to(self.device)
        predictedAction = self.model(stateTensor)
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
                agent.model.save()

            print('Game', agent.numberOfGames, 'Score', score, 'Record:', record)

            scores.append(score)
            scoreTotal += score
            scoreMean.append(scoreTotal / agent.numberOfGames)

            graph.plot('Training Endless Car Agent', scores, scoreMean)



if __name__ == '__main__':
    train()