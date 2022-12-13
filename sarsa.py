# usr/bin/python3

from game import game
from src import framework, state
from mc import mc
import numpy as np
import random as rd
from os.path import exists
import matplotlib.pyplot as plt


# state space S[d, p]
# action space A = [0, 1]
# value function V[S[d, p]]
# action function Q[S[d, p], A]
# eligibility trace E[S[d, p], A]
#         ---> dim(E) = dim(Q)


class sarsa(framework):
    # implement Sarsa(lambda) algorithm
    def __init__(self, l):
        super().__init__()
        self.E = self.Q  # eligibility traces -- same initialization as Q
        self.lmd = l

    def getQ(self, i):  # load action-state matrix. i: number of episodes to go through
        bar = self.getBar('Iterations: ', i)
        for i in range(0, i):
            self.E = np.zeros((10, 22, 2), dtype='float32')
            for d in range(0, len(self.Q)):
                for p in range(0, len(self.Q[0])):
                    a = self.epsilon(self.S[d][p])  # get action from epsilon-greedy policy
                    s_next, reward = self.step(self.S[d][p], a)  # get next state
                    self.update(self.S[d][p], s_next, a, reward)  # update Q, E
                    # print('Is terminal? ', s_next.isTerminal)
                    while not s_next.isTerminal:
                        a = self.epsilon(self.S[s_next.dealerSum][s_next.playerSum])  # get next action based on greedy
                        s_next, reward = self.step(self.S[s_next.dealerSum][s_next.playerSum], a)  # get next state
                        self.update(self.S[d][p], s_next, a, reward)  # update Q, E
            bar.next()
        bar.finish()

    def step(self, s, a):
        s.N += 1  # hello state for i have visited u plz increment counter
        self.N[s.dealerSum, s.playerSum, a] += 1  # thiz 1 too plz
        g = game(s.dealerSum, s.playerSum)  # initialize game environment from current state
        s_next, reward = g.step(s, self.A[a])  # retrieve new state (res) and reward
        return s_next, reward

    def update(self, s, s_next, a, reward):
        self.E[s.dealerSum, s.playerSum, a] += 1  # increment current eligibility trace
        for d in range(0, len(self.Q)):  # instant online update of Q and E
            for p in range(0, len(self.Q[0])):
                for a in (0, 1):
                    if self.N[d, p, a] != 0:
                        delta =self.delta(s, s_next, a, reward)
                        E = self.E[d, p, a]
                        N = self.N[d, p, a]
                        Q = self.Q[d, p , a] + delta / N * E
                        self.Q[d, p , a] += Q
                        # self.Q[d, p, a] += self.delta(s, s_next, a, reward) / self.N[d, p, a] * self.E[d, p, a]
                    self.E[d, p, a] = self.lmd * self.E[d, p, a]

    def delta(self, s, s_next, a, reward):  # get TD error
        if s_next.isTerminal:
            return reward  # reward
            print('@ d / p   ', s.dealerSum, ' / ', s.playerSum, '  q next is ', q_next, ', current q is ',
                  self.Q[s.dealerSum, s.playerSum, a])
        else:
            d_next = s_next.dealerSum
            p_next = s_next.playerSum
            q_next = self.Q[d_next, p_next, self.epsilon(self.S[d_next][p_next])]  # get next q based on eps-greedy
            print('@ d / p   ', s.dealerSum, ' / ', s.playerSum, '  q next is ', q_next,', current q is ', self.Q[s.dealerSum, s.playerSum, a])
            return q_next - self.Q[s.dealerSum, s.playerSum, a]  # TD error (omitted reward)

    def epsilon(self, s):  # epsilon-greedy policy. s: current state
        greedy = rd.choices([True, False], weights=self.getProb(s), k=1)  # weighted probability if greedy or not
        v = self.Q[s.dealerSum, s.playerSum]  # array with state values depending on action
        if greedy:
            if v[0] > v[1]:
                return 0
            else:
                return 1
        else:
            return rd.choice([0, 1])


def ms_error():  # get mean-squared error of difference to mc
    # if exists('Q_mc.npy'):
    #     print('      Found pre-calculated Q_mc! Will use this one')
    #     Q_mc = np.load('Q_mc.py')
    # else:
    i = int(input(' Enter number of iterations: '))
    print('      Run Monte-Carlo for true values...')
    inst = mc()
    inst.getQ(i)
    Q_mc = inst.Q

    print('\n  [+] Monte-Carlo Q loaded. \n')
    lmd = np.arange(0, 1.1, 0.1, dtype='float32')
    lmd_res = []

    for l in range(0, len(lmd)):
        print('\nCalculating Sarsa with Lambda ', l/10, '...\n')
        inst = sarsa(lmd[l])
        inst.getQ(i)
        Q_sarsa = inst.Q

        res = 0
        for d in range(0, 10):
            for p in range(0,22):
                for a in (0, 1):
                    res += np.square(Q_sarsa[d, p, a] - Q_mc[d, p, a])
        lmd_res.append(res)
    print('\n  [+] Sarsa Q loaded. \n')

    x = lmd
    y = lmd_res
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':

    print('\n    its-a meeeeeeee, Sarsa!!')
    i = int(input('Please enter # of iterations: '))
    l = int(input('Please enter lambda: '))
    print('\n\n')
    inst = sarsa(l/10)
    inst.run(i)
    inst.printRes()
    # inst.plotRes()