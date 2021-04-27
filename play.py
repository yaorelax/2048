import numpy as np
import pygame
import sys
import random
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pygame.locals import *

WIDTH = 4
WINDOW_WIDTH = 600
WALL_WIDTH = 450 // WIDTH
MAX_EPISODE = 100

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
        self.__width = width
        self.__score = 0
        self.__init_playground()
        self.__random_generate()
        if self.__debug:
            print('init:(%s * %s)' % (width, width))
            print(self.__playground)

    def __init_playground(self):
        self.__playground = np.zeros((self.__width, self.__width), dtype=int)

    def __random_generate(self, n=1):
        empty_list = list(np.argwhere(self.__playground == 0))
        if len(empty_list) < n:
            self.__terminal = True
            if self.__debug:
                print('game is over, press R to restart!')
            return
        for row, col in random.sample(empty_list, n):
            self.__playground[row][col] = random.choice([2, 4])
            if self.__debug:
                print('generate:', self.__playground[row][col])

    def __realize_slide(self, ground):
        score = 0
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
                                score += tmp[tip]
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
        return score

    def get_score(self):
        return self.__score

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
        score = 0
        if direction == 'left':
            ground = self.__playground
            score = self.__realize_slide(ground)
        elif direction == 'right':
            ground = np.flip(self.__playground, axis=1)
            score = self.__realize_slide(ground)
        elif direction == 'up':
            ground = self.__playground.T
            score = self.__realize_slide(ground)
        elif direction == 'down':
            ground = np.flip(self.__playground, axis=0).T
            score = self.__realize_slide(ground)
        else:
            move_success = False
            print('Direction error!')
        if move_success:
            self.__score += score
            if self.__debug:
                print('move %s, get score: %s' % (direction, score))
            self.__random_generate()
            if self.__debug:
                print(self.__playground)

    def fake_move(self, direction):
        next_playground = None
        score = 0
        if direction == 'left':
            next_playground = ground = copy.deepcopy(self.__playground)
            score = self.__realize_slide(ground)
        elif direction == 'right':
            next_playground = ground = copy.deepcopy(np.flip(self.__playground, axis=1))
            score = self.__realize_slide(ground)
        elif direction == 'up':
            next_playground = ground = copy.deepcopy(self.__playground.T)
            score = self.__realize_slide(ground)
        elif direction == 'down':
            next_playground = ground = copy.deepcopy(np.flip(self.__playground, axis=0).T)
            score = self.__realize_slide(ground)
        else:
            print('Direction error!')
        return next_playground, score

    def restart(self):
        self.__terminal = False
        self.__score = 0
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

    map_text = pygame.font.SysFont('simsunnsimsun', int(y_start * 3 / 5)).render('得分：%4d' % play.get_score(), True,
                                                                                 (106, 90, 205))
    text_rect = map_text.get_rect()
    text_rect.center = (WINDOW_WIDTH / 2, y_start / 2)
    screen.blit(map_text, text_rect)
    for i in range(WIDTH):
        for j in range(WIDTH):
            x_pos = x_start + WALL_WIDTH * i
            y_pos = y_start + WALL_WIDTH * j
            if ground[i][j] != 0:
                pygame.draw.rect(screen, YELLOW, [y_pos, x_pos, WALL_WIDTH, WALL_WIDTH], 0)
                map_text = pygame.font.Font(None, int(WALL_WIDTH * 3 / 5)).render(str(ground[i][j]), True,
                                                                                  (106, 90, 205))
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

def culculate_succession(ground):
    result = 0
    if True:
        for i in range(WIDTH):
            for j in range(WIDTH - 1):
                if ground[i][j] != 0:
                    if ground[i][j] == ground[i][j + 1]:
                        result += 1
                if ground[j][i] != 0:
                    if ground[j][i] == ground[j + 1][i]:
                        result += 1
    else:
        tmps = []
        for i in range(WIDTH):
            tmp1 = []
            tmp2 = []
            for j in range(WIDTH):
                if ground[i][j] != 0:
                    tmp1.append(ground[i][j])
                if ground[j][i] != 0:
                    tmp2.append(ground[j][i])
            tmps.append(tmp1)
            tmps.append(tmp2)
        for tmp in tmps:
            if len(tmp) <= 1:
                break
            tip_l = 0
            tip_r = 1
            while True:
                if tmp[tip_l] != tmp[tip_r]:
                    tip_l += 1
                    tip_r += 1
                else:
                    result += 1
                    tip_l += 2
                    tip_r += 2
                if tip_l >= len(tmp) or tip_r >= len(tmp):
                    break
    return result



def heuristic_algorithm(play, config):
    is_updated = True
    step = 0
    episode = 0
    scores = []
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
            assess_score = []  # 单步得分
            assess_empty = []  # 空格增加数
            assess_succession = []  # 连续相同对数
            assess_corner = []  # 评估大数在角落的程度
            current_playground = play.get_playground()
            current_empty = len(list(np.argwhere(current_playground == 0)))
            current_succession = culculate_succession(current_playground)

            for direction in DIRECTIONS:
                next_playground, score_of_onestep = play.fake_move(direction)
                assess_score.append(score_of_onestep)

                assess_empty.append(len(list(np.argwhere(next_playground == 0))) - current_empty)

                next_succession = culculate_succession(next_playground)
                assess_succession.append(next_succession - current_succession)

                big_num_locs = list(np.argwhere(next_playground == np.max(next_playground)))
                big_num_corner_diss = [[abs(row - row_c) + abs(col - col_c) for row_c, col_c in
                                        [(0, 0), (0, WIDTH - 1), (WIDTH - 1, 0), (WIDTH - 1, WIDTH - 1)]] for row, col
                                       in big_num_locs]
                assess_corner.append(np.mean([max(t) for t in big_num_corner_diss]))

            assess = np.array(
                [a * config[0] + b * config[1] + c * config[2] + d * config[3] for a, b, c, d in
                 zip(assess_score, assess_empty, assess_succession, assess_corner)]
            )

            play.move(DIRECTIONS[random.choice(np.where(assess == max(assess))[0])])
            is_updated = True
            step += 1
        else:
            # print('score:%4d, step:%4d' % (play.get_score(), step))
            # print(play.get_playground())
            print('[%s]episode:%3d score:%4d' % (''.join(str(x) for x in config), episode, play.get_score()))
            scores.append(play.get_score())
            episode += 1
            step = 0
            play.restart()
            if episode >= MAX_EPISODE:
                return scores

def ai_play(play):
    # assess_score assess_empty assess_succession assess_corner
    configs = []
    # configs.append([1, 0, 0, 0])
    # configs.append([0, 1, 0, 0])
    # configs.append([0, 0, 1, 0])
    # configs.append([0, 0, 0, 1])
    # configs.append([1, 1, 0, 0])
    # configs.append([1, 1, 1, 0])
    configs.append([1, 1, 1, 1])
    name_list = [''.join(str(x) for x in config) for config in configs]
    min_list = []
    max_list = []
    mean_list = []
    var_list = []
    x = list(range(len(configs)))
    for config in configs:
        scores = heuristic_algorithm(play, config)
        min = np.min(scores)
        max = np.max(scores)
        mean = np.mean(scores)
        min_list.append(min)
        max_list.append(max)
        mean_list.append(mean)
        var_list.append(
            np.std((np.array(scores) - min) / (max - min))
        )

    plt.subplot(221)
    plt.title('min')
    plt.bar(x, min_list, tick_label=name_list)

    plt.subplot(222)
    plt.title('max')
    plt.bar(x, max_list, tick_label=name_list)

    plt.subplot(223)
    plt.title('mean')
    plt.bar(x, mean_list, tick_label=name_list)

    plt.subplot(224)
    plt.title('var')
    plt.bar(x, var_list, tick_label=name_list)

    plt.show()

def main():
    play = Play2048(WIDTH, debug=False)
    # human_play(play)
    ai_play(play)

if __name__ == '__main__':
    main()
