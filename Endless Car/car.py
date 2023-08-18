import pygame
import random
from sys import exit
from collections import deque

pygame.init()

BLOCK_SIZE = 25
TRACK_WIDTH = 6
FONT_PATH = "../Fonts/playkiddo/PlayKiddo.ttf"
FONT = pygame.font.Font(FONT_PATH, size=30)

fontSurface = FONT.render("MY Game", False, 'red')

class CARGAME:

    def __init__(self, width=525, height=700) -> None:
        self.gameScreenWidth = width
        self.gameScreenHeight = height
        self.blockSize = 25

        self.gameScreen = pygame.display.set_mode((self.gameScreenWidth, self.gameScreenHeight))
        pygame.display.set_caption('CAR')
        self.gameClock = pygame.time.Clock()
        self.resetGame()


    def resetGame(self):
        self.track = deque(maxlen= 28)
        self.obstacle = deque(maxlen= 3)
        self.collectable = deque(maxlen= 5)

        self.currentGap = 0
        self.direction = [0,0,0]
        self.path = [True, True, True]
        self.obstacleTimer = 0
        self.collectableTimer = 0
        self.score = 0
        self.collect = 0
        self.gameOver = False
        self.carPosIndexX = 10
        self.carPosIndexY = self.gameScreenHeight / BLOCK_SIZE
        
        self.carRect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
        self.carRect.bottomleft = (BLOCK_SIZE * self.carPosIndexX, BLOCK_SIZE * self.carPosIndexY)

        for i in reversed(range(0, 28)):
            spawnCollectable = None
            spawnObstacle = None

            if i % 8 == 0:
                spawnObstacle = self._getObstacle()
                spawnObstacle.top = BLOCK_SIZE * i

            if i == 4 or i == 13 or i == 20:
                spawnCollectable = self._spwanCollectable()
                spawnCollectable.top = BLOCK_SIZE * i

            self._getTrack(y= i,obstacleRect = spawnObstacle, collectableRect= spawnCollectable)


    def _getTrack(self, y:int = 0, obstacleRect = None, collectableRect= None):
        randomInt = random.randint(-8, 8)

        if randomInt <= -1:
            self.currentGap = max(-5, self.currentGap - 1)
        elif randomInt >= 1:
            self.currentGap = min(5, self.currentGap + 1)

        leftRect = pygame.Rect(0, BLOCK_SIZE * y, BLOCK_SIZE * (TRACK_WIDTH + self.currentGap), BLOCK_SIZE)
        rightRect = pygame.Rect(0, BLOCK_SIZE * y, BLOCK_SIZE * (TRACK_WIDTH - self.currentGap), BLOCK_SIZE)
    
        self._appendTrack(leftRect, rightRect, obstacleRect, collectableRect)

    
    def _getObstacle(self):
        obstacleRect = pygame.Rect(0, 0, BLOCK_SIZE * 2, BLOCK_SIZE)
        obstacleRect.left = BLOCK_SIZE * (TRACK_WIDTH + self.currentGap + random.randint(3, 4))

        return obstacleRect


    def _spwanCollectable(self):
        collectableRect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
        collectableRect.left = BLOCK_SIZE * (TRACK_WIDTH + self.currentGap + random.randint(3, 4))

        return collectableRect  


    def _appendTrack(self, leftRect, rightRect, obstacleRect = None, collectableRect= None):
        leftRect.left = 0
        rightRect.right = self.gameScreenWidth

        self.track.append((leftRect, rightRect, obstacleRect, collectableRect))


    def _drawGameObjects(self):
        self.gameScreen.fill('black')

        for item in self.track:
            leftRect, rightRect, obstacleRect, collectableRect = item
            pygame.draw.rect(surface= self.gameScreen, color= 'grey', rect= leftRect)
            pygame.draw.rect(surface= self.gameScreen, color= 'grey', rect= rightRect)

            if obstacleRect is not None:
                pygame.draw.rect(surface= self.gameScreen, color= 'grey', rect= obstacleRect)

            if collectableRect is not None:
                pygame.draw.rect(surface= self.gameScreen, color= 'blue', rect= collectableRect)
            
            leftRect.top += BLOCK_SIZE
            rightRect.top += BLOCK_SIZE

            if obstacleRect is not None:
                obstacleRect.top += BLOCK_SIZE
            
            if collectableRect is not None:
                collectableRect.top += BLOCK_SIZE

        pygame.draw.rect(surface= self.gameScreen, color= 'green', rect= self.carRect)
        text = FONT.render("Score: " + str(self.score), True, 'white')
        text2 = FONT.render("Collect: " + str(self.collect), True, 'white')
        self.gameScreen.blit(text, [25, 0])
        self.gameScreen.blit(text2, [250, 0])


        return


    def isCollision(self, rect: pygame.Rect):
        if self.carRect.colliderect(rect):
            return True

        return False
    

    def _moveCar(self, direction: []):
        #move right by one place
        if direction == [0,1,0]:
            self.carPosIndexX += 1

        #move left by one place
        elif direction == [0,0,1]:
            self.carPosIndexX -= 1
        
        self.direction = direction
        self.carRect.bottomleft = (BLOCK_SIZE * self.carPosIndexX, BLOCK_SIZE * self.carPosIndexY)


    def performAction(self, direction: []):
        reward = 0
        spawnCollectableRect = None
        spawnObstacleRect = None


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self._moveCar(direction)

        leftRect, rightRect, obstacleRect, collectableRect = self.track[0]
        if self.isCollision(leftRect) or self.isCollision(rightRect):
            reward = -300
            self.gameOver = True

            return self.gameOver, self.score, reward
        
        if obstacleRect is not None:
            if self.isCollision(obstacleRect):
                reward = -300
                self.gameOver = True

                return self.gameOver, self.score, reward
            
        if collectableRect is not None:
            if self.isCollision(collectableRect):
                reward += 50
                self.collect += 1

        self._drawGameObjects()

        if self.obstacleTimer >= 180:
            spawnObstacleRect = self._getObstacle()
            self.obstacleTimer = 0
            self.collectableTimer = 0

        if self.collectableTimer == 100:
            spawnCollectableRect = self._spwanCollectable()
            self.score += 1

        clone = self.carRect.copy()

        if self.currentGap < 0 and (self.carPosIndexX >= self.carPosIndexX + 4 or self.carPosIndexX <= self.carPosIndexX + 6):
            reward += 5
        elif self.currentGap > 0 and (self.carPosIndexX >= self.carPosIndexX - 6 or self.carPosIndexX <= self.carPosIndexX - 4):
            reward += 5
        elif self.currentGap == 0 and (self.carPosIndexX >= 9 or self.carPosIndexX <= 11):
            reward += 5
        else:
            reward -= 30

        if direction == [0,0,1] and self.path[2] == True:
            reward += 30
        elif direction == [1,0,0] and self.path[0] == True:
            reward += 30
        elif direction == [0,1,0] and self.path[1] == True:
            reward += 30
        else:
            reward -= 30

        clone.left -= BLOCK_SIZE
        clone.top -= 2*BLOCK_SIZE

        _, _, obstacleRect, collectableRect = self.track[2]
        self.path = [True, True, True]
        if obstacleRect is not None and clone.colliderect(obstacleRect) == True:
            self.path[2] = False

        clone.left += BLOCK_SIZE

        if obstacleRect is not None and clone.colliderect(obstacleRect) == True:
            self.path[0] = False

        clone.left += BLOCK_SIZE

        if obstacleRect is not None and clone.colliderect(obstacleRect) == True:
            self.path[1] = False

        if collectableRect is not None and clone.colliderect(collectableRect) == True:
            self.path = [False, True, False]

        clone.left -= BLOCK_SIZE

        if collectableRect is not None and clone.colliderect(collectableRect) == True:
            self.path = [True, False, False]

        clone.left -= BLOCK_SIZE

        if collectableRect is not None and clone.colliderect(collectableRect) == True:
            self.path = [False, False, True]


        clone.left += BLOCK_SIZE
        clone.top += BLOCK_SIZE

        _, _, _, collectableRect = self.track[1]
        if collectableRect is not None and clone.colliderect(collectableRect) == True:
            self.path = [True, False, False]
        

        self._getTrack(y= 0, obstacleRect= spawnObstacleRect, collectableRect= spawnCollectableRect)
        
        pygame.display.flip()
        self.gameClock.tick(20)
        self.obstacleTimer += 20
        self.collectableTimer += 20

        return self.gameOver, self.score, reward



# car = CARGAME(525, 700)

# while True:
#     dir = [1,0,0]
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()

#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_LEFT:
#                 dir = [0,0,1]
#             elif event.key == pygame.K_RIGHT:
#                 dir = [0,1,0]

#     car.performAction(dir)