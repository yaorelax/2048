import numpy as np
import pygame
import sys
import random
import time
import matplotlib.pyplot as plt
from pygame.locals import *
from collections import OrderedDict

MAX_EPISODE = 10

class ENV:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    WINDOW_WIDTH = 600

    def __init__(self, play):
        pygame.init()
        pygame.event.set_allowed([12, KEYUP])
        self.play = play
        self.WALL_WIDTH = 450 // play.width
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_WIDTH))

    def update_env(self):
        self.screen.fill(self.WHITE)
        ground = self.play.get_playground()
        x_start = y_start = (self.WINDOW_WIDTH - self.play.width * self.WALL_WIDTH) / 2
        pygame.draw.rect(self.screen, self.BLACK,
                         [y_start - 1, x_start - 1, self.play.width * self.WALL_WIDTH + 2,
                          self.play.width * self.WALL_WIDTH + 2], 1)

        map_text = pygame.font.SysFont('simsunnsimsun', int(y_start * 3 / 5)).render('得分：%4d' % self.play.get_score(),
                                                                                     True,
                                                                                     (106, 90, 205))
        text_rect = map_text.get_rect()
        text_rect.center = (self.WINDOW_WIDTH / 2, y_start / 2)
        self.screen.blit(map_text, text_rect)
        for i in range(self.play.width):
            for j in range(self.play.width):
                x_pos = x_start + self.WALL_WIDTH * i
                y_pos = y_start + self.WALL_WIDTH * j
                if ground[i][j] != 0:
                    pygame.draw.rect(self.screen, self.YELLOW, [y_pos, x_pos, self.WALL_WIDTH, self.WALL_WIDTH], 0)
                    map_text = pygame.font.Font(None, int(self.WALL_WIDTH * 3 / 5)).render(str(ground[i][j]), True,
                                                                                           (106, 90, 205))
                    text_rect = map_text.get_rect()
                    text_rect.center = (y_pos + self.WALL_WIDTH / 2, x_pos + self.WALL_WIDTH / 2)
                    self.screen.blit(map_text, text_rect)
                pygame.draw.rect(self.screen, self.BLACK, [y_pos, x_pos, self.WALL_WIDTH, self.WALL_WIDTH], 1)
        if self.play.is_terminal():
            self.screen.blit(pygame.font.SysFont('simsunnsimsun', 100).render('按R重开', True, self.BLACK), (100, 100))

class Play2048:
    DIRECTIONS = ['left', 'right', 'up', 'down']

    def __init__(self, width, debug=False):
        self.__debug = debug
        self.__terminal = False
        self.__playground = None
        self.width = width
        self.__score = 0
        self.__init_playground()
        self.__random_generate(n=2)
        if self.__debug:
            print('init:(%s * %s)' % (width, width))
            print(self.__playground)

    def __init_playground(self):
        self.__playground = np.zeros((self.width, self.width), dtype=int)

    def __random_generate(self, n=1):
        empty_list = list(np.argwhere(self.__playground == 0))
        if len(empty_list) < n:
            self.__terminal = True
            if self.__debug:
                print('game is over, press R to restart!')
            return
        for row, col in random.sample(empty_list, n):
            self.__playground[row][col] = 2 if np.random.rand() < 0.9 else 4
            if self.__debug:
                print('generate:', self.__playground[row][col])
        if len(empty_list) == 1:
            is_terminal = True
            for direction in self.DIRECTIONS:
                if len(list(np.argwhere(self.fake_move(direction) == 0))) != 0:
                    is_terminal = False
                    break
            self.__terminal = is_terminal

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
                for tip in range(len(tmp) - 1):
                    if tip < len(tmp) - 1:
                        if tmp[tip] == tmp[tip + 1]:
                            score += tmp[tip]
                            tmp[tip] *= 2
                            del tmp[tip + 1]
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

    def heuristic1(self, ground):
        def score(ground):
            weight = [[pow(4, 2 * self.width - 2 - i - j) for j in range(self.width)] for i in range(self.width)]
            sco = sum(sum(np.array(weight) * np.array(ground)))
            return sco

        def penalty(ground):
            pen = 0
            for i in range(0, 4):
                for j in range(0, 4):
                    if i - 1 >= 0:
                        pen += abs(ground[i][j] - ground[i - 1][j])
                    if i + 1 < 4:
                        pen += abs(ground[i][j] - ground[i + 1][j])
                    if j - 1 >= 0:
                        pen += abs(ground[i][j] - ground[i][j - 1])
                    if j + 1 < 4:
                        pen += abs(ground[i][j] - ground[i][j + 1])
            pen2 = sum(sum(ground == 0))
            return pen - 2 * pen2

        return score(ground) - penalty(ground)

    def heuristic(self, ground):
        def culculate_succession(ground):
            result = 0
            if True:
                for i in range(self.width):
                    for j in range(self.width - 1):
                        if ground[i][j] != 0:
                            if ground[i][j] == ground[i][j + 1]:
                                result += 1
                        if ground[j][i] != 0:
                            if ground[j][i] == ground[j + 1][i]:
                                result += 1
            else:
                tmps = []
                for i in range(self.width):
                    tmp1 = []
                    tmp2 = []
                    for j in range(self.width):
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
                            result += np.log2(tmp[tip_l])
                            tip_l += 2
                            tip_r += 2
                        if tip_l >= len(tmp) or tip_r >= len(tmp):
                            break
            return result

        assess_score = self.get_score()
        assess_empty = sum(sum(ground == 0))
        assess_succession = culculate_succession(ground)
        big_num_locs = list(np.argwhere(ground == np.max(ground)))
        big_num_corner_diss = [[abs(row - row_c) + abs(col - col_c) for row_c, col_c in
                                [(0, 0), (0, self.width - 1), (self.width - 1, 0),
                                 (self.width - 1, self.width - 1)]] for row, col
                               in big_num_locs]
        assess_corner = np.mean([max(t) for t in big_num_corner_diss])
        assess_array = np.array([assess_score, assess_empty, assess_succession, assess_corner])
        weights = [1, 1, 1, 1]
        assess = sum(assess_array * weights)
        return assess

    def heuristic3(self, ground, commonRatio=0.25):
        linearWeightedVal = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile = (-1, -1)
        for y in range(self.width):
            for x in range(self.width):
                b_x = x
                b_y = y
                if invert:
                    b_x = self.width - 1 - x
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile == (-1, -1):
                    criticalTile = (b_x, b_y)
                linearWeightedVal += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal2 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile2 = (-1, -1)
        for x in range(self.width):
            for y in range(self.width):
                b_x = x
                b_y = y
                if invert:
                    b_y = self.width - 1 - y
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile2 == (-1, -1):
                    criticalTile2 = (b_x, b_y)
                linearWeightedVal2 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal3 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile3 = (-1, -1)
        for y in range(self.width):
            for x in range(self.width):
                b_x = x
                b_y = self.width - 1 - y
                if invert:
                    b_x = self.width - 1 - x
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile3 == (-1, -1):
                    criticalTile3 = (b_x, b_y)
                linearWeightedVal3 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal4 = 0
        invert = False
        weight = 1.
        malus = 0
        criticalTile4 = (-1, -1)
        for x in range(self.width):
            for y in range(self.width):
                b_x = self.width - 1 - x
                b_y = y
                if invert:
                    b_y = self.width - 1 - y
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile4 == (-1, -1):
                    criticalTile4 = (b_x, b_y)
                linearWeightedVal4 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal5 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile5 = (-1, -1)
        for y in range(self.width):
            for x in range(self.width):
                b_x = x
                b_y = y
                if invert:
                    b_x = self.width - 1 - x
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile5 == (-1, -1):
                    criticalTile5 = (b_x, b_y)
                linearWeightedVal5 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal6 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile6 = (-1, -1)
        for x in range(self.width):
            for y in range(self.width):
                b_x = x
                b_y = y
                if invert:
                    b_y = self.width - 1 - y
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile6 == (-1, -1):
                    criticalTile6 = (b_x, b_y)
                linearWeightedVal6 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal7 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile7 = (-1, -1)
        for y in range(self.width):
            for x in range(self.width):
                b_x = x
                b_y = self.width - 1 - y
                if invert:
                    b_x = self.width - 1 - x
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile7 == (-1, -1):
                    criticalTile7 = (b_x, b_y)
                linearWeightedVal7 += currVal * weight
                weight *= commonRatio
            invert = not invert

        linearWeightedVal8 = 0
        invert = True
        weight = 1.
        malus = 0
        criticalTile8 = (-1, -1)
        for x in range(self.width):
            for y in range(self.width):
                b_x = self.width - 1 - x
                b_y = y
                if invert:
                    b_y = self.width - 1 - y
                # linearW
                currVal = ground[b_x, b_y]
                if currVal == 0 and criticalTile8 == (-1, -1):
                    criticalTile8 = (b_x, b_y)
                linearWeightedVal8 += currVal * weight
                weight *= commonRatio
            invert = not invert

        maxVal = max(linearWeightedVal, linearWeightedVal2, linearWeightedVal3, linearWeightedVal4, linearWeightedVal5,
                     linearWeightedVal6, linearWeightedVal7, linearWeightedVal8)
        if linearWeightedVal2 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal2
            criticalTile = criticalTile2
        if linearWeightedVal3 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal3
            criticalTile = criticalTile3
        if linearWeightedVal4 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal4
            criticalTile = criticalTile4
        if linearWeightedVal5 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal5
            criticalTile = criticalTile5
        if linearWeightedVal6 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal6
            criticalTile = criticalTile6
        if linearWeightedVal7 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal7
            criticalTile = criticalTile7
        if linearWeightedVal8 > linearWeightedVal:
            linearWeightedVal = linearWeightedVal8
            criticalTile = criticalTile8

        return maxVal

    def move(self, direction):
        if self.__terminal:
            if self.__debug:
                print('game is over, press R to restart!')
            return
        move_success = True
        score = 0
        current_playground_backup = [[x for x in row] for row in self.__playground]
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
        if np.all(current_playground_backup == self.__playground):
            move_success = False
        if move_success:
            self.__score += score
            if self.__debug:
                print('move %s, get score: %s' % (direction, score))
            self.__random_generate()
            if self.__debug:
                print(self.__playground)

    def fake_move(self, direction, current_playground=None):
        if current_playground is None:
            current_playground = self.__playground
        next_playground = np.array([[x for x in row] for row in current_playground])
        score = 0
        if direction == 'left':
            ground = next_playground
            score = self.__realize_slide(ground)
        elif direction == 'right':
            ground = np.flip(next_playground, axis=1)
            score = self.__realize_slide(ground)
        elif direction == 'up':
            ground = next_playground.T
            score = self.__realize_slide(ground)
        elif direction == 'down':
            ground = np.flip(next_playground, axis=0).T
            score = self.__realize_slide(ground)
        else:
            print('Direction error!')
        return next_playground, score

    def restart(self):
        self.__terminal = False
        self.__score = 0
        self.__init_playground()
        self.__random_generate(n=2)
        if self.__debug:
            print('restart:')
            print(self.__playground)

def coast_time(func):
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} coast time:{time.perf_counter() - t:.8f} s')
        return result

    return fun

def human_play(env):
    play = env.play
    is_updated = True
    while True:
        if is_updated:
            env.update_env()
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

@coast_time
def heuristic_algorithm(env, weights):  # assess_score assess_empty assess_succession assess_corner
    play = env.play

    def culculate_succession(ground):
        result = 0
        if True:
            for i in range(play.width):
                for j in range(play.width - 1):
                    if ground[i][j] != 0:
                        if ground[i][j] == ground[i][j + 1]:
                            result += 1
                    if ground[j][i] != 0:
                        if ground[j][i] == ground[j + 1][i]:
                            result += 1
        else:
            tmps = []
            for i in range(play.width):
                tmp1 = []
                tmp2 = []
                for j in range(play.width):
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

    is_updated = True
    episode = 0
    scores = []
    step = 0
    while True:
        if is_updated:
            env.update_env()
            is_updated = False
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        pygame.display.flip()
        pygame.event.pump()

        if not play.is_terminal():
            assess = []
            current_playground = play.get_playground()
            current_empty = sum(sum(current_playground == 0))
            current_succession = culculate_succession(current_playground)
            for direction in play.DIRECTIONS:
                next_playground, score_of_onestep = play.fake_move(direction)
                if np.all(next_playground == play.get_playground()):
                    assess.append(-99999999)
                    continue
                assess_score = score_of_onestep
                assess_empty = sum(sum(next_playground == 0)) - current_empty
                assess_succession = culculate_succession(next_playground) - current_succession

                big_num_locs = list(np.argwhere(next_playground == np.max(next_playground)))
                big_num_corner_diss = [[abs(row - row_c) + abs(col - col_c) for row_c, col_c in
                                        [(0, 0), (0, play.width - 1), (play.width - 1, 0),
                                         (play.width - 1, play.width - 1)]] for row, col
                                       in big_num_locs]
                assess_corner = np.mean([max(t) for t in big_num_corner_diss])

                assess_array = np.array([assess_score, assess_empty, assess_succession, assess_corner])
                assess.append(sum(assess_array * weights))
            assess = np.array(assess)
            play.move(play.DIRECTIONS[random.choice(np.where(assess == max(assess))[0])])
            is_updated = True
            step += 1
        else:
            print('[%s%s]episode:%3d score:%4d step:%3d max_block:%4d'
                  % ('heuristic',
                     ''.join(str(x) for x in weights),
                     episode,
                     play.get_score(),
                     step,
                     np.max(play.get_playground())
                     )
                  )
            scores.append(play.get_score())
            episode += 1
            step = 0
            play.restart()
            if episode >= MAX_EPISODE:
                return scores

def heuristic_algorithm_with_one_step_heuristic(env, x=None):  # assess_score assess_empty assess_succession assess_corner
    play = env.play
    is_updated = True
    episode = 0
    scores = []
    step = 0
    while True:
        if is_updated:
            env.update_env()
            is_updated = False
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        pygame.display.flip()
        pygame.event.pump()

        if not play.is_terminal():
            assess = []
            for direction in play.DIRECTIONS:
                next_playground, score_of_onestep = play.fake_move(direction)
                if np.all(next_playground == play.get_playground()):
                    assess.append(-99999999)
                    continue
                assess.append(play.heuristic(next_playground))
            assess = np.array(assess)
            play.move(play.DIRECTIONS[random.choice(np.where(assess == max(assess))[0])])
            is_updated = True
            step += 1
        else:
            print('[%s]episode:%3d score:%4d step:%3d max_block:%4d'
                  % ('heuristic__one_step',
                     episode,
                     play.get_score(),
                     step,
                     np.max(play.get_playground())
                     )
                  )
            scores.append(play.get_score())
            episode += 1
            step = 0
            play.restart()
            if episode >= MAX_EPISODE:
                return scores

@coast_time
def expectimax_algorithm(env, max_depth):
    play = env.play

    class Brain:
        def __init__(self, max_memory):
            self.__memory = OrderedDict()
            self.__max_memory = max_memory

        def remember(self, something, value):
            if something not in self.__memory:
                if len(self.__memory) == self.__max_memory:
                    del self.__memory[next(iter(self.__memory))]
                self.__memory[something] = value

        def recall(self, something):
            if something in self.__memory:
                return self.__memory[something]
            else:
                return None

    def search(ground, depth, move=False):
        if depth == 0 or (move and play.is_terminal()):
            return play.heuristic(ground)
        if move:
            alpha = play.heuristic(ground)
            for direction in play.DIRECTIONS:
                child = play.fake_move(direction, ground)[0]
                if np.all(child == ground):
                    continue
                # with memory
                alpha_ = brain.recall(hash(str((child, depth - 1))))
                if alpha_ is None:
                    alpha_ = search(child, depth - 1)
                    brain.remember(hash(str((child, depth - 1))), alpha_)
                alpha = max(alpha, alpha_)
        else:
            alpha = 0
            zeros = [(i, j) for i, j in list(np.argwhere(ground == 0))]
            for i, j in zeros:
                c1 = np.array([[x for x in row] for row in ground])
                c2 = np.array([[x for x in row] for row in ground])
                c1[i][j] = 2
                c2[i][j] = 4
                alpha += (.9 * search(c1, depth - 1, True) / len(zeros) +
                          .1 * search(c2, depth - 1, True) / len(zeros))
        return alpha



    brain = Brain(10000)
    is_updated = True
    episode = 0
    step = 0
    scores = []
    while True:
        if is_updated:
            env.update_env()
            is_updated = False
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
        pygame.display.flip()
        pygame.event.pump()

        if not play.is_terminal():
            assess = []
            for direction in play.DIRECTIONS:
                next_playground = play.fake_move(direction)[0]
                if np.all(next_playground == play.get_playground()):
                    assess.append(-99999999)
                    continue
                result = search(next_playground, max_depth)
                assess.append(result)

            assess = np.array(assess)
            direction = play.DIRECTIONS[random.choice(np.where(assess == max(assess))[0])]
            play.move(direction)
            is_updated = True
            step += 1
        else:
            print('[%s%s]episode:%3d score:%4d step:%3d max_block:%4d'
                  % ('expectimax',
                     max_depth,
                     episode,
                     play.get_score(),
                     step,
                     np.max(play.get_playground())))
            scores.append(play.get_score())
            episode += 1
            step = 0
            play.restart()
            if episode >= MAX_EPISODE:
                return scores

def ai_play(env):
    configs = []
    # assess_score, assess_empty, assess_succession, assess_corner
    # configs.append((heuristic_algorithm, [1, 1, 1, 1]))
    # configs.append((heuristic_algorithm, [4, 3, 3, 1]))
    # configs.append((heuristic_algorithm, [5, 4, 3, 1]))
    # configs.append((heuristic_algorithm, [10, 4, 3, 1]))
    # max_depth

    configs.append((expectimax_algorithm, 4))
    # configs.append((heuristic_algorithm_with_one_step_heuristic, ['one']))
    name_list = [('H' + (''.join(str(x) for x in config[1]))
                  if type(config[1]) is list
                  else ('E' + str(config[1])))
                 for config in configs]
    min_list = []
    max_list = []
    mean_list = []
    cv_list = []
    x = list(range(len(configs)))
    for algorithm, config in configs:
        scores = algorithm(env, config)
        min = np.min(scores)
        max = np.max(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        min_list.append(min)
        max_list.append(max)
        mean_list.append(mean)
        cv_list.append(std / min)

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
    plt.title('cv')
    plt.bar(x, cv_list, tick_label=name_list)

    plt.show()

def main():
    play = Play2048(width=4, debug=False)
    env = ENV(play)
    # human_play(env)
    ai_play(env)

if __name__ == '__main__':
    main()
