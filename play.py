import numpy as np
import pygame
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygame.locals import *

WIDTH = 4
WINDOW_WIDTH = 600
WALL_WIDTH = 500 // WIDTH

global playground

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
pygame.event.set_allowed([12, KEYUP])
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DIRECTIONS = ['left', 'right', 'up', 'down']

class ANet(nn.Module):  # a(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x * 2
        return actions_value

class Play2048:
    def __init__(self, width, debug=False):
        self.__debug = debug
        self.__terminal = False
        self.__playground = None
        self.__empty_list = []
        self.__width = width
        self.__init_playground()
        self.__random_generate()
        if self.__debug:
            print('init:(%s * %s)' % (width, width))
            print(self.__playground)

    def __init_playground(self):
        self.__playground = np.zeros((self.__width, self.__width), dtype=int)

    def __generate_empty_list(self):
        self.__empty_list = list(np.argwhere(self.__playground == 0))

    def __random_generate(self, number=2, n=2):
        self.__generate_empty_list()
        if len(self.__empty_list) < n:
            self.__terminal = True
            if self.__debug:
                print('game is over, press R to restart!')
            return
        for row, col in random.sample(self.__empty_list, n):
            self.__playground[row][col] = number
        # for row, col in random.sample(self.empty_list, np.random.randint(1, 3)):
        #     self.playground[row][col] = np.random.choice([2, 4])

    def __realize_slide(self, ground):
        for i in range(len(ground)):
            tmp = []
            for a in ground[i]:
                if a != 0:
                    tmp.append(a)
            if len(tmp) == 0:
                continue
            elif len(tmp) >= 1:
                for _ in range(len(tmp) - 1):
                    tip = 0
                    while True:
                        if tip < len(tmp) - 1:
                            if tmp[tip] == tmp[tip + 1]:
                                tmp[tip] *= 2
                                del tmp[tip + 1]
                            tip += 1
                        else:
                            break
                for j in range(len(ground)):
                    if j < len(tmp):
                        ground[i][j] = tmp[j]
                    else:
                        ground[i][j] = 0

    def get_score(self):
        return sum(sum(self.__playground))

    def get_playground(self):
        return self.__playground

    def is_terminal(self):
        return self.__terminal

    def move(self, direction):
        if self.__terminal:
            if self.__debug:
                print('game is over, press R to restart!')
            return
        move_success = True
        if direction == 'left':
            ground = self.__playground
            self.__realize_slide(ground)
            self.__playground = ground
        elif direction == 'right':
            ground = np.flip(self.__playground, axis=1)
            self.__realize_slide(ground)
            self.__playground = np.flip(ground, axis=1)
        elif direction == 'up':
            ground = self.__playground.T
            self.__realize_slide(ground)
            self.__playground = ground.T
        elif direction == 'down':
            ground = np.flip(self.__playground, axis=0).T
            self.__realize_slide(ground)
            self.__playground = np.flip(ground.T, axis=0)
        else:
            move_success = False
            print('Direction error!')
        if move_success:
            if self.__debug:
                print('move %s' % direction)
            self.__random_generate()
            if self.__debug:
                print(self.__playground)

    def fake_move(self, direction):
        next_playground = None
        if direction == 'left':
            ground = self.__playground
            self.__realize_slide(ground)
            next_playground = ground
        elif direction == 'right':
            ground = np.flip(self.__playground, axis=1)
            self.__realize_slide(ground)
            next_playground = np.flip(ground, axis=1)
        elif direction == 'up':
            ground = self.__playground.T
            self.__realize_slide(ground)
            next_playground = ground.T
        elif direction == 'down':
            ground = np.flip(self.__playground, axis=0).T
            self.__realize_slide(ground)
            next_playground = np.flip(ground.T, axis=0)
        else:
            print('Direction error!')
        return next_playground

    def restart(self):
        self.__terminal = False
        self.__init_playground()
        self.__random_generate()
        if self.__debug:
            print('restart:')
            print(self.__playground)


def update_env(play):
    screen.fill(WHITE)
    ground = play.get_playground()
    x_start = y_start = (WINDOW_WIDTH - WIDTH * WALL_WIDTH) / 2
    pygame.draw.rect(screen, BLACK, [y_start - 1, x_start - 1, WIDTH * WALL_WIDTH + 2, WIDTH * WALL_WIDTH + 2], 1)
    for i in range(WIDTH):
        for j in range(WIDTH):
            x_pos = x_start + WALL_WIDTH * i
            y_pos = y_start + WALL_WIDTH * j
            if ground[i][j] != 0:
                pygame.draw.rect(screen, YELLOW, [y_pos, x_pos, WALL_WIDTH, WALL_WIDTH], 0)
                map_text = pygame.font.Font(None, int(WALL_WIDTH * 3 / 5)).render(str(ground[i][j]), True, (106, 90, 205))
                text_rect = map_text.get_rect()
                text_rect.center = (y_pos + WALL_WIDTH / 2, x_pos + WALL_WIDTH / 2)
                screen.blit(map_text, text_rect)
            pygame.draw.rect(screen, BLACK, [y_pos, x_pos, WALL_WIDTH, WALL_WIDTH], 1)
    if play.is_terminal():
        screen.blit(pygame.font.SysFont('simsunnsimsun', 100).render('按R重开', True, BLACK), (100, 100))


def human_play(play):
    is_updated = True
    while True:
        if is_updated:
            update_env(play)
            is_updated = False
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            if event.type == KEYUP:
                if event.key == K_LEFT:
                    play.move('left')
                if event.key == K_RIGHT:
                    play.move('right')
                elif event.key == K_UP:
                    play.move('up')
                elif event.key == K_DOWN:
                    play.move('down')
                elif event.key == K_r:
                    play.restart()
                is_updated = True
        pygame.display.flip()
        pygame.event.pump()


def ai_play(play):
    is_updated = True
    while True:
        if is_updated:
            update_env(play)
            is_updated = False
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        pygame.display.flip()
        pygame.event.pump()

        if not play.is_terminal():
            assess_score = []
            assess_empty = []
            for direction in DIRECTIONS:
                next_playground = play.fake_move(direction)
                assess_score.append(sum(sum(next_playground)))
                assess_empty.append(len(list(np.argwhere(next_playground == 0))))
            assess = [a + b for a, b in zip(assess_score, assess_empty)]
            play.move(DIRECTIONS[np.argmax(assess)])
            is_updated = True
        else:
            print('score:', play.get_score())
            play.restart()

def main():
    play = Play2048(WIDTH, debug=False)
    # human_play(play)
    ai_play(play)

if __name__ == '__main__':
    main()