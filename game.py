# usr/bin/python3

import random as rd


class game:
    def __init__(self, dealer=None, player=None):  # overloaded to create game from state or from scratch
        if dealer is None:
            self.dealerCard = card('black')
        else:
            self.dealerCard = [card(dealer)]

        if player is None:
            self.playerCards = [card('black')]
        else:
            self.playerCards = [card(player)]

    def isBust(self, inCards):
        if 0 <= self.sum(inCards) <= 21:
            return False
        else:
            return True

    def step(self, s, a):

        if a == 'hit':
            self.playerCards.append(card())
            if self.isBust(self.playerCards):
                reward = -1
                return gamestate(self, reward), reward
            else:
                return gamestate(self), 0  # go on with reward 0

        elif a == 'stick':
            dealer = [card(s.dealerSum), card()]  # create dealer's deck from state and add new card
            while 0 <= self.sum(dealer) < 17:  # dealer hit or stick
                dealer.append(card())  # if hit then add another card

            if self.isBust(dealer) or (self.sum(self.playerCards) > self.sum(dealer)):
                reward = 1
                return gamestate(self, reward), reward
            elif self.sum(self.playerCards) < self.sum(dealer):
                reward = -1
                return gamestate(self, reward), reward
            else:
                reward = 0
                return gamestate(self, reward), reward

    @staticmethod
    def sum(cards):
        # Add if black, subtract if red
        res = 0
        for c in cards:
            if c.color == 'black':
                res += c.value
            else:
                res -= c.value
        return res


class gamestate(game):
    def __init__(self, inGame, reward=None):  # overloaded using reward: set isTerminal to True if reward is given
        #  properties of state: dealer's first card & player's sum of cards
        super().__init__()
        self.dealerSum = inGame.dealerCard[0].value
        self.playerSum = super().sum(inGame.playerCards)
        if reward is None:
            self.isTerminal = False
        else:
            self.isTerminal = True


class card:
    def __init__(self, c=None):  # overloaded using c: set color / value as specified in arg

        if c is None:
            self.color = rd.choice(['black', 'black', 'red'])
            self.value = rd.randint(1, 10)

        elif type(c) is str:
            self.color = c
            self.value = rd.randint(1, 10)

        elif type(c) is int:
            self.color = 'black'
            self.value = c
