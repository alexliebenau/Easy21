# usr/bin/python3

from sarsa import sarsa, ms_error
import numpy as np
from joblib import Parallel, delayed


class sarsa_parallel(sarsa):
    # implement Sarsa(lambda) algorithm
    def __init__(self,l):
        super().__init__(l)

    def getQ(self, i):  # load action-state matrix. i: number of episodes to go through
        bar = self.getBar('Parallel Iterations: ', i)
        for i in range(i):
            self.E = np.zeros((10, 22, 2), dtype='float32')
            for d in range(0, len(self.Q)):

                Q = self.Q[d]
                N = self.N[d]
                E = self.E[d]

                num_cores = 8
                # batch = int(i / num_cores)
                self.Q, self.N, self.E = \
                    Parallel(
                        n_jobs=num_cores,
                        # batch_size = batch,
                        verbose=0,
                        backend='loky')\
                        (delayed(self.iterate)(Q, N, E) for _ in range(i))
            bar.next()
        bar.finish()

    def iterate(self, d, p, Q, N, E):  # run parallel iterations of Q

        # for d in range(0, len(self.Q)):
        # for p in range(0, len(Q[0])):
        a = self.epsilon(self.S[d][p])  # get action from epsilon-greedy policy
        s_next, reward = self.step(self.S[d][p], a)  # get next state
        Q, N, E = self.update(self.S[d][p], s_next, a, reward, Q, N, E)  # update Q, E
        while not s_next.isTerminal:
            a = self.epsilon(self.S[s_next.dealerSum][s_next.playerSum])  # get next action based on greedy
            s_next, reward = self.step(self.S[s_next.dealerSum][s_next.playerSum], a)  # get next state
            Q, N, E = self.update(self.S[d][p], s_next, a, reward, Q, N, E)  # update Q, E
        return Q, N

    def update(self, s, s_next, a, reward, Q, N, E):  # returns E to use it in smaller scope for parallel
        E[s.dealerSum, s.playerSum, a] += 1  # increment current eligibility trace
        for d in range(0, len(Q)):  # instant online update of Q and E
            for p in range(0, len(Q[0])):
                for a in (0, 1):
                    if N[d, p, a] != 0:  # if state hasn't been visited its E is zero anyway
                        Q[d, p, a] += self.delta(s, s_next, a, reward) / N[d, p, a] * E[d, p, a]
                    E[d, p, a] = self.lmd * E[d, p, a]
        return Q, N, E


if __name__ == '__main__':

    print('\n    its-a meeeeeeee, Sarsa!!')
    i = int(input('Please enter # of iterations: '))
    l = int(input('Please enter lambda: '))
    print('\n\n')
    inst = sarsa_parallel(l/10)
    inst.run(i)
    # inst.printRes()
    inst.plotRes()