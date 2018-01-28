# Easy21
This repository contains implementation of reinforcement learning algorithms to Easy21 ( a variation of BlackJack ) as described in David Silver's course on Reinforcement Learning.

The following are the rules to the game:

- The game is played with an infinite deck of cards (i.e. cards are sampled
with replacement)
- Each draw from the deck results in a value between 1 and 10 (uniformly
distributed) with a colour of red (probability 1/3) or black (probability
2/3).
- There are no aces or picture (face) cards in this game
- At the start of the game both the player and the dealer draw one black
card (fully observed)
- Each turn the player may either stick or hit
- If the player hits then she draws another card from the deck
- If the player sticks she receives no further cards
- The values of the player’s cards are added (black cards) or subtracted (red
cards)
- If the player’s sum exceeds 21, or becomes less than 1, then she “goes
bust” and loses the game (reward -1)
- If the player sticks then the dealer starts taking turns. The dealer always
sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes
bust, then the player wins; otherwise, the outcome – win (reward +1),
lose (reward -1), or draw (reward 0) – is the player with the largest sum.

I have implmented model-free control for Easy21. This assumes that the player has no information about the underlying MDP (environment dynamics) such as distribution from which cards are drawn or the dealer's strategy. All the player has access to is the state it is in and the reward it receives for performing some action.

Here, State is defined by the sum of cards in the player's hand and the initial card with the dealer. In each state, two actions can be performed, either 'Hit' or 'Stick'. Each game runs till the terminal state is reached.

The optimal value function V*(s) = max (over all a) Q*(s,a) has been plotted for all the states. Exact monte-carlo, exact SARSA and SARSA with linear function approximation algorithms have been implemented while following an epsilon-greedy strategy.
The details regarding step-size, epsilon variation and features selected for linear approximation can be found here: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf

## Exact Monte Carlo

![ExactMC](https://github.com/tusharsircar95/Easy21_ReinforcementLearning/blob/master/ExactMC.png)

## Exact Sarsa

![ExactSarsa](https://github.com/tusharsircar95/Easy21_ReinforcementLearning/blob/master/Exact%20Sarsa.png)

## Sarsa With Linear Function Approximation

![ApproxSarsa](https://github.com/tusharsircar95/Easy21_ReinforcementLearning/blob/master/ApproxSarsa.png)
