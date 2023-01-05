# usr/bin/python3

from game import game
from sarsa import sarsa
from mc import mc
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# state space S[d, p]
# action space A = [0, 1]
# value function V[S[d, p]]
# quality function Q[S[d, p], A]
# eligibility trace E[S[d, p], A]
#         ---> dim(E) = dim(Q)


class sarsa_threaded(sarsa):
    # implement Sarsa(lambda) algorithm
    def __init__(self, l):
        super().__init__(l)

    def getQ(self, i):  # load action-state matrix. i: number of episodes to go through
        bar = self.getBar('Iterations: ', i)
        
        with ThreadPoolExecutor(10) as executor:
            for iter in range(i):
                executor.submit(self.iterate)

        bar.finish()

    def iterate(self):
        for d in range(0, len(self.Q)):       
            for p in range(0, len(self.Q[0])): 
                E = np.zeros((10, 22, 2))                 
                a = self.epsilon(self.S[d][p])  # get action from epsilon-greedy policy
                s_next, reward = self.step(self.S[d][p], a)  # get next state
                E = self.update(d, self.S[d][p], s_next, a, reward, E)  # update Q, E
                while not s_next.isTerminal:
                    a = self.epsilon(self.S[s_next.dealerSum][s_next.playerSum])  # get next action based on greedy
                    s_next, reward = self.step(self.S[s_next.dealerSum][s_next.playerSum], a)  # get next state
                    E = self.update(d, self.S[d][p], s_next, a, reward, E)  # update Q, E
    
    def update(self, d, s, s_next, a, reward, E):
        E[s.dealerSum, s.playerSum, a] += 1  # increment current eligibility trace
        # for d in range(0, len(self.Q)):  # instant online update of Q and E
        for p in range(0, len(self.Q[0])):
            for a in (0, 1):
                if self.N[d, p, a] != 0:  # if state hasn't been visited its E is zero anyway
                    self.Q[d, p, a] += self.delta(s, s_next, a, reward) / self.N[d, p, a] * E[d, p, a]
                E[d, p, a] = self.lmd * E[d, p, a]
        return E

   
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
        inst = sarsa_threaded(lmd[l])
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
    inst = sarsa_threaded(l/10)
    inst.run(i)
    # inst.printRes()
    inst.plotRes()
