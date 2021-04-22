import numpy as np
import pygame
import sys
import random
from pygame.locals import *

WIDTH = 4
WINDOW_WIDTH = 600

global playground

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
WHITE = pygame.Color('white')
BLACK = pygame.Color('black')


class Play:
    def __init__(self, width):
        self.terminal = False
        self.playground = None
        self.empty_list = []
        self.width = width
        self.init_playground()
        self.random_generate()
        print('init:(%s * %s)' % (width, width))
        self.show_playground()

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

    def show_playground(self):
        print(self.playground)

    def move(self, direction):
        if self.terminal:
            print('game is over, press R to restart!')
            return
        move_success = True
        if direction == 'left':
            for i in range(self.width):
                tmp = []
                for a in self.playground[i]:
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
                    for j in range(self.width):
                        if j < len(tmp):
                            self.playground[i][j] = tmp[j]
                        else:
                            self.playground[i][j] = 0

        elif direction == 'right':
            pass
        elif direction == 'up':
            pass
        elif direction == 'down':
            pass
        else:
            move_success = False
            print('Direction error!')
        if move_success:
            print('move %s' % direction)
            self.random_generate()
            self.show_playground()

    def restart(self):
        self.terminal = False
        self.init_playground()
        self.random_generate()
        print('restart:')
        self.show_playground()



def detect_keys(play):
    while True:
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


def main():
    play = Play(WIDTH)
    detect_keys(play)


if __name__ == '__main__':
    main()