import numpy as np
import pygame
import sys
import random
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


class Play:
    def __init__(self, width):
        self.terminal = False
        self.playground = None
        self.empty_list = []
        self.width = width
        self.init_playground()
        self.random_generate()
        print('init:(%s * %s)' % (width, width))
        print(self.playground)

    def init_playground(self):
        self.playground = np.zeros((self.width, self.width), dtype=int)

    def generate_empty_list(self):
        self.empty_list = list(np.argwhere(self.playground == 0))

    def random_generate(self, number=2, n=2):
        self.generate_empty_list()
        if len(self.empty_list) < n:
            self.terminal = True
            print('game is over, press R to restart!')
            return
        for row, col in random.sample(self.empty_list, n):
            self.playground[row][col] = number
        # for row, col in random.sample(self.empty_list, np.random.randint(1, 3)):
        #     self.playground[row][col] = np.random.choice([2, 4])

    def get_playground(self):
        return self.playground

    def is_terminal(self):
        return self.terminal

    def move(self, direction):
        if self.terminal:
            print('game is over, press R to restart!')
            return
        move_success = True
        if direction == 'left':
            ground = self.playground
            realize_slide(ground)
            self.playground = ground
        elif direction == 'right':
            ground = np.flip(self.playground, axis=1)
            realize_slide(ground)
            self.playground = np.flip(ground, axis=1)
        elif direction == 'up':
            ground = self.playground.T
            realize_slide(ground)
            self.playground = ground.T
        elif direction == 'down':
            ground = np.flip(self.playground, axis=0).T
            realize_slide(ground)
            self.playground = np.flip(ground.T, axis=0)
        else:
            move_success = False
            print('Direction error!')
        if move_success:
            print('move %s' % direction)
            self.random_generate()
            print(self.playground)

    def restart(self):
        self.terminal = False
        self.init_playground()
        self.random_generate()
        print('restart:')
        print(self.playground)

def realize_slide(ground):
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


def running(play):
    is_updated = True
    while True:
        if is_updated:
            screen.fill(WHITE)
            ground = play.get_playground()
            x_start = y_start = (WINDOW_WIDTH - WIDTH * WALL_WIDTH) / 2
            for i in range(WIDTH):
                for j in range(WIDTH):
                    x_pos = x_start + WALL_WIDTH * i
                    y_pos = y_start + WALL_WIDTH * j
                    if ground[i][j] != 0:
                        pygame.draw.rect(screen, YELLOW, [y_pos, x_pos, WALL_WIDTH, WALL_WIDTH], 0)
                        screen.blit(pygame.font.SysFont('simsunnsimsun', WALL_WIDTH // 2).render(str(ground[i][j]), True, BLACK), (
                        y_pos + WALL_WIDTH // 4, x_pos + WALL_WIDTH // 4))
                    pygame.draw.rect(screen, BLACK, [y_pos, x_pos, WALL_WIDTH, WALL_WIDTH], 1)
            is_updated = False
            if play.is_terminal():
                screen.blit(pygame.font.SysFont('simsunnsimsun', 100).render('按R重开', True, BLACK), (100, 100))
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


def main():
    play = Play(WIDTH)
    running(play)


if __name__ == '__main__':
    main()