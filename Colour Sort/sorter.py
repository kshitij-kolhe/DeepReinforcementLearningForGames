import pygame
import random
from sys import exit
from collections import deque, namedtuple

pygame.init()

FPS = 30
FONT_PATH = "../Fonts/playkiddo/PlayKiddo.ttf"
FONT = pygame.font.Font(FONT_PATH, size=30)

fontSurface = FONT.render("MY Game", False, 'red')

Point = namedtuple('Point', 'x, y')

class Sorter:

    def __init__(self, width=600, height=600) -> None:
        self.gameScreenWidth = width
        self.gameScreenHeight = height


        self.gameScreen = pygame.display.set_mode((self.gameScreenWidth, self.gameScreenHeight))
        pygame.display.set_caption('Colour Sort')
        self.gameClock = pygame.time.Clock()

        self.blockSize = 20
        self.spawnButton = pygame.Rect(0, 0, self.blockSize, self.blockSize)
        self.spawnButton.topleft = (300, 300)
        self.redRegion = pygame.Rect(0, 0, 60, 60)
        self.redRegion.topleft = (0, 0)
        self.greenRegion = pygame.Rect(0, 0, 60, 60)
        self.greenRegion.topright = (600, 0)
        self.blueRegion = pygame.Rect(0, 0, 60, 60)
        self.blueRegion.bottomleft = (0, 600)
        self.yellowRegion = pygame.Rect(0, 0, 60, 60)
        self.yellowRegion.bottomright = (600, 600)

        self.resetGame()
        return

    def resetGame(self):
        self.score = 0
        self.innerSorter = 'orange'
        self.sortItemColor = 'blue'
        self.direction = [0,0,0,0]
        self.frame = 0
        self.distance = 999999999
        self.target = Point(300, 300)
        self.spawnObstacles = False
        self.obstacles = deque(maxlen= 10)
        self.points = deque(maxlen= 10)

        self.sorter = pygame.Rect(0, 0, self.blockSize, self.blockSize)
        self.sorter.topleft = (400, 400)
        self.sortItem = pygame.Rect(0, 0, self.blockSize, self.blockSize)
        self.sortItem.topleft = (-100, -100)
        return

    
    def spawnSortItem(self):
        color = ['red', 'green', 'blue', 'yellow']
        i = random.randint(0, 3)
        x = random.randint(60, 540)
        y = random.randint(60, 540)

        x = (x // self.blockSize ) * self.blockSize
        y = (y // self.blockSize) * self.blockSize

        self.sortItemColor = color[i]
        self.sortItem.topleft = (x, y)
        if x == 300 and y == 300 :
            self.spawnSortItem()

        return

    def move(self, action): 

        if action == [1,0,0,0]:
            self.sorter.y -= self.blockSize
        elif action == [0,1,0,0]:
            self.sorter.x += self.blockSize
        elif action == [0,0,1,0]:
            self.sorter.y += self.blockSize
        elif action == [0,0,0,1]:
            self.sorter.x -= self.blockSize
        
        self.direction = action
        return
    
    def isOutOfBound(self):

        if self.sorter.x > 580 or self.sorter.x < 0 or self.sorter.y > 580 or self.sorter.y < 0:
            return True

        return False
    
    
    def updateScreen(self):
        self.gameScreen.fill('black')

        pygame.draw.rect(self.gameScreen, 'grey', self.spawnButton)
        pygame.draw.rect(self.gameScreen, 'red', self.redRegion)
        pygame.draw.rect(self.gameScreen, 'green', self.greenRegion)
        pygame.draw.rect(self.gameScreen, 'blue', self.blueRegion)
        pygame.draw.rect(self.gameScreen, 'yellow', self.yellowRegion)
        pygame.draw.rect(self.gameScreen, self.sortItemColor, self.sortItem)
        pygame.draw.rect(self.gameScreen, 'orange', self.sorter)
        pygame.draw.rect(self.gameScreen, self.innerSorter, (self.sorter.x + 5, self.sorter.y + 5, 10, 10))


        text = FONT.render("Score: " + str(self.score), True, 'white')
        self.gameScreen.blit(text, [250, 0])
        pygame.display.flip()

        return

    def performAction(self, action = [0,0,0,0], obs= False):
        gameOver = False
        reward = 0
        score = self.score
        self.frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    gameOver = True
                if event.key == pygame.K_LEFT:
                    action = [0,0,0,1]
                elif event.key == pygame.K_RIGHT:
                    action = [0,1,0,0]
                elif event.key == pygame.K_UP:
                    action = [1,0,0,0]
                elif event.key == pygame.K_DOWN:
                    action = [0,0,1,0]


        self.move(action)

        if self.sorter.colliderect(self.sortItem):
            score += 1
            reward = 50
            self.innerSorter = self.sortItemColor
            self.sortItem.topleft = (-100, -100)
            self.frame = 0
            if self.innerSorter == 'red':
                self.target = Point(0,0)
            elif self.innerSorter == 'green':
                self.target = Point(600,0)
            elif self.innerSorter == 'blue':
                self.target = Point(0,600)
            elif self.innerSorter == 'yellow':
                self.target = Point(600,600)
        elif (self.innerSorter == 'blue' and self.sorter.colliderect(self.blueRegion)) or (self.innerSorter == 'red' and self.sorter.colliderect(self.redRegion)) or (self.innerSorter == 'green' and self.sorter.colliderect(self.greenRegion)) or (self.innerSorter == 'yellow' and self.sorter.colliderect(self.yellowRegion)):
            score += 2
            reward = 50
            self.innerSorter = 'orange'
            self.target = Point(self.spawnButton.x, self.spawnButton.y)
            self.frame = 0
        elif self.sorter.colliderect(self.spawnButton) and self.sortItem.topleft == (-100, -100) and self.innerSorter == 'orange':
            score += 1
            reward = 50
            self.spawnSortItem()
            self.target = Point(self.sortItem.x, self.sortItem.y)
            self.frame = 0
        elif self.isOutOfBound():
            gameOver = True
            reward = -100
        elif self.frame >= 300 and self.score == score:
            gameOver = True
            reward = -30


        distance = pygame.Vector2(self.sorter.x, self.sorter.y).distance_to(pygame.Vector2(self.target.x, self.target.y))
        if distance <= self.distance:
            reward = 3
        else:
            reward = -3


        self.score = score
        self.distance = distance
        self.updateScreen()
        self.gameClock.tick(FPS)
        return gameOver, self.score, reward




# sorter = Sorter()
# gameOver = False
# sorter.spawnObstacles = True
# while True:

#     if gameOver == True:
#         break


#     gameOver, score, reward = sorter.performAction()