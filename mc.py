# usr/bin/python3

from game import game
import random as rd
import numpy as np
from src import framework


class mc(framework):
    # implement MC learning as class
    def __init__(self):  # inherit constructor from framework
       super().__init__()

    def q(self, s, a):  # recursive sum, based on bellman optimality eq
        s.N += 1  # hello state for i have visited u plz increment counter
        self.N[s.dealerSum, s.playerSum, a] += 1  # thiz 1 too plz
        g = game(s.dealerSum, s.playerSum)  # initialize game environment from current state
        (res, reward) = g.step(s, self.A[a])  # retrieve new state (res) and reward
        if res.isTerminal:  # game has ended?
            return reward
        else:
            next_s = self.S[res.dealerSum][res.playerSum]  # get next state s'
            greedy = rd.choices([True, False], weights=self.getProb(s), k=1)
            if greedy:
                return reward + max(self.q(next_s, 0), self.q(next_s, 1))
            else:
                return reward + rd.choice([self.q(next_s, 0), self.q(next_s, 1)])

    def getQ(self, i):  # load action-state matrix. i: number of episodes to go through
        bar = self.getBar('Iterations: ', i)
        for i in range(0, i):  # go through i iterations MC
            for d in range(0, len(self.Q)):
                for p in range(0, len(self.Q[0])):
                    hit = self.q(self.S[d][p], 0)
                    stick = self.q(self.S[d][p], 1)
                    if hit > stick:
                        self.Q[d, p, 0] += (hit - self.Q[d, p, 0]) / self.N[d, p, 0]
                    else:
                        self.Q[d, p, 1] += (stick - self.Q[d, p, 1]) / self.N[d, p, 1]
            bar.next()
        bar.finish()
        with open('Q_mc.npy', 'wb') as f:  # write Q to file
            np.save(f, self.Q)


if __name__ == '__main__':

    print('\n     Helloooouuu. Its MC calling \n')
    i = int(input('Pleaser enter number of iterations for learning: '))
    print('\n\n')
    inst = mc()
    inst.run(i)
    # inst.printRes()
    inst.plotRes()