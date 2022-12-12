# usr/bin/python3

from game import game
import random as rd
from mc import state

# x = game()
# print('Dealer card is', x.dealerCard.value, '\n')
# cstate = state(x)
#
# print('# player cards: ', len(x.playerCards))
# print('Player Summe: ', x.sum(x.playerCards))
# print('--------------')
#
# (nstate, reward) = x.step(cstate, 'hit')
# print('is terminated: ', nstate.isTerminal)
# print('# player cards: ', len(x.playerCards))
# print('Player Summe: ', x.sum(x.playerCards))
# print('Reward: ', reward)
# print('--------------')
#
# (nstate, reward) = x.step(cstate, 'stick')
#
# print('is terminated: ', nstate.isTerminal)
# print('# player cards: ', len(x.playerCards))
# print('Player Summe: ', x.sum(x.playerCards))
# print('Reward: ', reward)
# print('--------------')





dealerCard = range(1, 11)
playerSum = range(12, 22)
S = []  # state space, [dealerSum][playerSum]
A = ['hit', 'stick']  # initialize possible actions

for p in range(12, 22):
    P = []
    for d in range(1, 11):
        P.append(state(d, p))
    S.append(P)

    print(S[10][21])