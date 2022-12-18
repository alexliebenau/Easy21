# usr/bin/python3

from src import framework
from sarsa import sarsa, ms_error
import numpy as np
from joblib import Parallel, delayed
from functools import partial
import random as rd
from game import game


class sarsa_parallel(framework):
    # implement Sarsa(lambda) algorithm
    def __init__(self, l):
        super().__init__()
        self.E = self.Q  # eligibility traces -- same initialization as Q
        self.lmd = l
        np.seterr(all='raise')

    def getQ(self, i):  # load action-state matrix. i: number of episodes to go through
        bar = self.getBar('Parallel Iterations: ', i)
        for i in range(i):
            self.E = np.zeros((10, 22, 2), dtype='float32')
            # for dealer in range(0, len(self.Q)):
            #
            #     Q = self.Q[dealer]
            #     N = self.N[dealer]
            #     E = self.E[dealer]
            print(hex(id(self.Q[0])))
            num_cores = 8
            # batch = int(i / num_cores)
            # iter_ = partial(self.iterate, d=dealer, Q=Q, N=N, E=E)
            # self.Q, self.N, self.E = \
            res = \
                Parallel(
                    n_jobs=num_cores,
                    # batch_size = batch,
                    verbose=0,
                    backend='loky')\
                    (delayed(self.iterate)(self.S[d],
                                           self.Q[d],
                                           self.N[d],
                                           self.E[d])
                                for d in range(len(self.Q)))
            resQ, resN = zip(*res)
            self.Q = np.array(resQ)
            self.N = np.array(resN)
            bar.next()
        bar.finish()

    def iterate(self, s, Q_iter, N_iter, E_iter):  # run parallel iterations of Q
        # it is not necessary to iterate through d because the dealer sum will not change during game
        # for d in range(0, len(self.Q)):

        # self.Q = Q
        # self.N = N
        # self.E = E
        print(hex(id(Q_iter)))
        for p in range(0, len(Q_iter[0])):
            a = self.epsilon(s[p], Q_iter)  # get action from epsilon-greedy policy
            s_next, reward = self.step(s[p], a, N_iter)  # get next state
            Q_iter, N_iter, E_iter = self.update(s[p], s_next, a, reward, Q_iter, N_iter, E_iter)  # update Q, E
            while not s_next.isTerminal:
                a = self.epsilon(self.S[s_next.dealerSum][s_next.playerSum], Q_iter)  # get next action based on greedy
                s_next, reward = self.step(self.S[s_next.dealerSum][s_next.playerSum], a, N_iter)  # get next state
                Q_iter, N_iter, E_iter = self.update(s[p], s_next, a, reward, Q_iter, N_iter, E_iter)  # update Q, E
        return Q_iter, N_iter

    def update(self, s, s_next, a, reward, Q_iter, N_iter, E_iter):  # returns E to use it in smaller scope for parallel
        E_iter[s.playerSum, a] += 1  # increment current eligibility trace
        # for d in range(0, len(Q)):  # instant online update of Q and E
        for p in range(0, len(Q_iter[0])):
            for a in (0, 1):
                if N_iter[p, a] != 0:  # if state hasn't been visited its E is zero anyway
                    Q_iter[p, a] += self.delta(Q_iter, s, s_next, a, reward) / N_iter[p, a] * E_iter[p, a]
                E_iter[p, a] = self.lmd * E_iter[p, a]
        return Q_iter, N_iter, E_iter

    def step(self, s, a, N_iter):
        s.N += 1  # hello state for i have visited u plz increment counter
        N_iter[s.playerSum, a] += 1  # thiz 1 too plz
        g = game(s.dealerSum, s.playerSum)  # initialize game environment from current state
        s_next, reward = g.step(s, self.A[a])  # retrieve new state (res) and reward
        return s_next, reward

    def epsilon(self, s, Q_iter):  # epsilon-greedy policy. s: current state
        greedy = rd.choices([True, False], weights=self.getProb(s), k=1)  # weighted probability if greedy or not
        v = Q_iter[s.playerSum]  # array with state values depending on action
        if greedy:
            if v[0] < v[1]:
                return 1
            else:
                return 0
        else:
            return rd.choice([0, 1])

    def delta(self, Q_iter, s, s_next, a, reward):  # get TD error
        if s_next.isTerminal:
            return reward - Q_iter[s.playerSum, a]  # if terminal there is no q_next
        else:
            d_next = s_next.dealerSum
            p_next = s_next.playerSum
            q_next = Q_iter[p_next, self.epsilon(self.S[d_next][p_next], Q_iter)]  # get next q based on eps-greedy
            return q_next - Q_iter[s.playerSum, a]  # TD error (omitted reward because always 0 if not terminal)


if __name__ == '__main__':

    print('\n    its-a meeeeeeee, Sarsa in parallel!!')
    # i = int(input('Please enter # of iterations: '))
    # l = int(input('Please enter lambda: '))
    print('\n\n')
    inst = sarsa_parallel(5)  # l/10)
    inst.run(1000)  # i)
    # inst.printRes()
    inst.plotRes()
