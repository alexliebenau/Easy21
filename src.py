# usr/bin/python3

import numpy as np
import random as rd
import matplotlib.pyplot as plt
from progress.bar import ChargingBar
from os.path import exists
from time import process_time as cpuTime


class framework():
    def __init__(self):
        self.S = []  # state space, [dealerSum][playerSum]
        (d, p) = (10, 22)  # d: dealer card, from 1 - 10; p: player sum, from 0 - 21)
        for dealerCard in range(0, d):  # fill S-array with states
            tmp = []
            for playerSum in range(0, p):
                tmp.append(state(dealerCard, playerSum))
            self.S.append(tmp)  # S[dealerCard][playerSum]

        # self.A = {'hit': 0, 'stick': 1}  # initialize possible actions
        self.A = {0: 'hit', 1: 'stick'}
        self.Q = np.zeros((d, p, 2), dtype='float32')  # action-state matrix Q = [d, p, a] where a=0: hit, a=1: stick
        self.V = np.zeros((d, p), dtype='float32')  # value function
        self.N = np.zeros((d, p, 2), dtype='int')  # counts how many times a has been selected in a

    def getVopt(self):  # get optimal value function V* = max Q*(s, a)
        for d in range(0, len(self.V)):
            for p in range(0, len(self.Q[0])):
                self.V[d, p] = max(self.Q[d, p])

    def plotRes(self):  # plots value matrix
        x = np.arange(1, 10, 1)  # dealer axis
        y = np.arange(0, 21, 1)  # player axis
        x, y = np.meshgrid(x, y)
        z = self.V[x, y]  # hit?
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=plt.cm.cividis)
        plt.show()

    def printRes(self):
        for d in range(0, 10):
            for p in range(0, 22):
                print('D: ', d, 'P: ', p, ' ',
                      self.S[d][p].N, ' N(s)  N(s,a)', self.N[d ,p ,0], ' / ', self.N[d, p, 1],
                      '  Q hit ', self.Q[d, p, 0],
                      '  Q stick ', self.Q[d, p, 1],
                      '  V ', self.V[d][p])
                if self.E is not None:
                    print('  E ', self.E[d, p, 0], '  ', self.E[d, p, 1])
                print('-')
            print('----------')

    def run(self, i):  # hit it babyyyy

        if not ((exists('Q_mc.npy') or exists('Q_sarsa.npy')) and exists('V_opt.npy')):
            st = cpuTime()  # start timers
            self.getQ(i)
            print('   CPU-time: ', cpuTime() - st, 's')
            self.getVopt()

            with open('V_opt.npy', 'wb') as f:  # write P to file
                np.save(f, self.V)

        else:
            print('   ->  Learning process already completed\n       Resume with plotting...')
            # self.Q = np.load('Q_mc.npy')
            # self.V = np.load('V_opt.npy')

    @staticmethod
    def getProb(s):
        N_0 = 100
        eps = N_0 / (s.N + N_0)  # define epsilon for greedy-ness
        return [1 - eps / 2, eps / 2]

    @staticmethod
    def getBar(name, m):
        bar = ChargingBar(
            name,
            max=m,
            suffix='%(index)d  / %(max)d (%(percent)d%%) - %(elapsed)ds elapsed'
        )
        return bar


class state:
    def __init__(self, d, p, t=None):  # overloaded for termination
        self.N = 0
        self.dealerSum = d
        self.playerSum = p

        if t is None:
            self.isTerminal = False
        else:
            self.isTerminal = True
