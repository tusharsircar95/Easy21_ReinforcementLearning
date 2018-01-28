import numpy as np
import random
import os
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Draws a card from the deck based on given distribution
class Deck:
	def __init__(self):
		pass
	
	@staticmethod
	def getNextCard():
		number = random.randint(1,10)
		runif = random.uniform(0,1)
		color = 0
		if runif < 0.6666667:
			color = +1
		else: color = -1
		return [number,color]
	
# State consisting of sum of player's and dealer's card
class State:
	def __init__(self,playerSum=random.randint(1,10),dealerSum=random.randint(1,10)):
		self.playerSum = playerSum
		self.dealerSum = dealerSum
	
	def __str__(self):
		return 'playerSum,dealerSum = ' + str(self.playerSum) + ',' + str(self.dealerSum)
		
	def getFeatures(self):
		return [self.playerSum,self.dealerSum]
	
	def getPlayerSum(self):
		return self.playerSum
		
	def getDealerSum(self):
		return self.dealerSum
	
	def isTerminalState(self):
		if self.playerSum < 1 or self.playerSum > 21:
			return True
		if self.dealerSum < 1 or self.dealerSum >= 17:
			return True
		return False
	
# Action can be 0 for Hit or 1 for Stick
class Action:
	# 0 : Hit
	# 1 : Stick
	def __init__(self,move=0):
		self.move = move
	def __str__(self):
		return 'Action: ' + ('Hit' if self.move == 0 else 'Stick')
		
	def getMove(self):
		return self.move
	
# Returns a feature vector corresponding to a state-action pair
def getFeatures(state,action):
	features = np.zeros(36)
	dealerIntervals = [[1,4],[4,7],[7,10]]
	playerIntervals = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]]
	actionIntervals = [0,1]
	
	[x,y],z = state.getFeatures(),action.getMove()
	index = 0
	for i in range(len(playerIntervals)):
		for j in range(len(dealerIntervals)):
			for k in range(len(actionIntervals)):
				if x >= playerIntervals[i][0] and x <= playerIntervals[i][1]:
					if y >= dealerIntervals[j][0] and y <= dealerIntervals[j][1]:
						if z == actionIntervals[k]:
							features[index] = 1
				index = index + 1
	return features
	
# Given a state and action, returns reward and next state
# This is the environment that the player interacts with	
def step(state,action):
	if action.getMove() == 0:
		[number,color] = Deck.getNextCard()
		[playerSum,dealerSum] = state.getFeatures()
		playerSum = playerSum + (number*color)
		reward = 0
		if playerSum < 1 or playerSum > 21:
			reward = -1
		return State(playerSum,dealerSum),reward
	if action.getMove() == 1:
		[playerSum,dealerSum] = state.getFeatures()
		while dealerSum < 17 and dealerSum > 0:
			[number,color] = Deck.getNextCard()
			dealerSum = dealerSum + (number*color)
		reward = 0
		if dealerSum < 1 or dealerSum > 21:
			reward = 1
		elif dealerSum > playerSum:
			reward = -1
		elif dealerSum < playerSum:
			reward = 1
		else: reward = 0
		return State(playerSum,dealerSum),reward

# Node to store meta information about each state-action pair
class QNode:
	def __init__(self):
		self.value = 0
		self.count = 0
		self.store = 0
		self.visited = False

# Node to store meta information about each state
class StateNode:
	def __init__(self):
		self.visited = False
		self.count = 0

# Exact Monte-Carlo Control algorithm		
def MonteCarloControl():

	numberOfEpisodes = 50000
	Q = [[[QNode() for x in range(2)] for y in range(11)] for z in range(22)]
	SN = [[StateNode() for x in range(11)] for y in range(22)]
	N0 = 1000
	
	for i in range(numberOfEpisodes):
		print('Episode ' + str(i+1) + '\n')
		S0 = State(random.randint(1,10),random.randint(1,10))
		totalRewardTillNow = 0
		
		#print('Initial State: ' + str(S0))
		#input("")
		
		# Update visited for each state since we are using First Visit MC
		for x in range(22):
			for y in range(11):
				SN[x][y].visited = False
				for z in range(2):
					Q[x][y][z].visited = False
					
		while not S0.isTerminalState():
			x,y = S0.getPlayerSum(),S0.getDealerSum()
			epsilon = N0/(N0+SN[x][y].count)
			probability = random.uniform(0,1)
			action = Action()
			if probability < epsilon: # select random action
				move = 0 if random.uniform(0,1) < 0.50 else 1
				action = Action(move)
				#print('Action chosen randomly: ' + str(action))
				#input("")
				z = move
			else:
				move = 0 if (Q[x][y][0].value > Q[x][y][1].value) else 1
				action = Action(move)
				#print('Action chosen greedily: ' + str(action))
				#input("")
				z = move
				
			if not Q[x][y][z].visited:
					Q[x][y][z].visited = True
					Q[x][y][z].count = Q[x][y][z].count + 1
					Q[x][y][z].store = totalRewardTillNow
			if not SN[x][y].visited:
					SN[x][y].visited = True
					SN[x][y].count = SN[x][y].count + 1
			S0,reward = step(S0,action)
			#print('Next state: ' + str(S0))
			#print('Reward received: ' + str(reward))
			#input("")
			totalRewardTillNow = totalRewardTillNow + reward
		
		# Episode ends here
		for x in range(22):
			for y in range(11):
				for z in range(2):
					if Q[x][y][z].visited == True:
						rewardFromHere = totalRewardTillNow - Q[x][y][z].store
						Q[x][y][z].value = Q[x][y][z].value + (rewardFromHere-Q[x][y][z].value)/Q[x][y][z].count
						#print(str(x) + ',' + str(y) + ',' + str(z) + ' was ' + str(Q[x][y][z].value))
						#print('Changed to: ' + str(Q[x][y][z].value))
						#input("")
		#print(S0)
		#print(totalRewardTillNow)
		
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(1, 21, 1)
	y = np.arange(1, 10, 1)
	X, Y = np.meshgrid(x, y)
	zs = np.array([(Q[x][y][0].value if Q[x][y][0].value > Q[x][y][1].value else Q[x][y][1].value) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)

	ax.plot_surface(Y, X, Z,cmap=cm.coolwarm)

	ax.set_ylabel('Player Sum')
	ax.set_xlabel('Dealer Card')
	ax.set_zlabel('Value')
	ax.set_title('V* using Exact Monte Carlo')
	fig = plt.gcf()
	fig.savefig('ExactMC.png')
	#plt.show()
	
	
# Exact Sarsa Lambda Control algorithm	
def SarsaControl():
	
	numberOfEpisodes = 40000
	Q = [[[QNode() for x in range(2)] for y in range(11)] for z in range(22)]
	SN = [[StateNode() for x in range(11)] for y in range(22)]
	lambdas =  0.2#[i/10 for i in range(11)]
	N0 = 100
	
	for i in range(numberOfEpisodes):
		print('Episode ' + str(i+1) + '\n')
		S0 = State(random.randint(1,10),random.randint(1,10))
		totalRewardTillNow = 0
		
		#print('Initial State: ' + str(S0))
		#input("")
		
		# Update visited for each state since we are using First Visit MC
		for x in range(22):
			for y in range(11):
				SN[x][y].visited = False
				for z in range(2):
					Q[x][y][z].visited = False
		
		
		SARPath = []
		while not S0.isTerminalState():
			x,y = S0.getPlayerSum(),S0.getDealerSum()
			epsilon = N0/(N0+SN[x][y].count)
			probability = random.uniform(0,1)
			action = Action()
			if probability < epsilon: # select random action
				move = 0 if random.uniform(0,1) < 0.50 else 1
				action = Action(move)
				#print('Action chosen randomly: ' + str(action))
				#input("")
				z = move
			else:
				move = 0 if (Q[x][y][0].value > Q[x][y][1].value) else 1
				action = Action(move)
				#print('Action chosen greedily: ' + str(action))
				#input("")
				z = move
				
			if not Q[x][y][z].visited:
					Q[x][y][z].visited = True
					Q[x][y][z].count = Q[x][y][z].count + 1
			if not SN[x][y].visited:
					SN[x][y].visited = True
					SN[x][y].count = SN[x][y].count + 1	
			
			S1,reward = step(S0,action)
			SARPath.append([S0,action,reward])
			S0 = S1
			#print('Next state: ' + str(S0))
			#print('Reward received: ' + str(reward))
			#input("")
			totalRewardTillNow = totalRewardTillNow + reward
		
		# Episode ends here
		SARPath.append([S0,Action(0),0])
		pathLength = len(SARPath)
		for i in range(pathLength-1):
			S0 = SARPath[i][0]
			rewardSum = 0
			totalTDReturn = 0
			for j in range(i,pathLength-1):
				rewardSum = rewardSum + SARPath[j][2]
				nextState = SARPath[j+1][0]
				[x,y] = nextState.getFeatures()
				if not nextState.isTerminalState():
					z = SARPath[j+1][1].getMove()
					totalTDReturn = totalTDReturn + (rewardSum + Q[x][y][z].value) * (lambdas**(j-i))
				else: totalTDReturn = totalTDReturn + (rewardSum) * (lambdas**(j-i))
				
			totalTDReturn = totalTDReturn * (1-lambdas)/(1-(lambdas**(pathLength-1-i)))
			[x,y] = S0.getFeatures()
			z = SARPath[i][1].getMove()
			Q[x][y][z].value = Q[x][y][z].value + (totalTDReturn - Q[x][y][z].value)/Q[x][y][z].count
			#print(str(x) + ',' + str(y) + ',' + str(z) + ' was ' + str(Q[x][y][z].value))
			#print('Changed to: ' + str(Q[x][y][z].value))
			#input("")
				
		#print(S0)
		#print(totalRewardTillNow)
		
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(1, 21, 1)
	y = np.arange(1, 10, 1)
	X, Y = np.meshgrid(x, y)
	zs = np.array([(Q[x][y][0].value if Q[x][y][0].value > Q[x][y][1].value else Q[x][y][1].value) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)

	ax.plot_surface(Y, X, Z,cmap=cm.coolwarm)

	ax.set_ylabel('Player Sum')
	ax.set_xlabel('Dealer Card')
	ax.set_zlabel('Value')
	ax.set_title('V* using Exact Sarsa')
	fig = plt.gcf()
	fig.savefig('Exact Sarsa.png')
	#plt.show()


# Sarsa Lambda Control algorithm with Function Approximation
def SarsaControlFA():
	
	numberOfEpisodes = 20000
	lambdas =  0.8#[i/10 for i in range(11)]
	N0 = 100
	alpha = 0.01
	epsilon = 0.05
	weights = np.zeros(len(getFeatures(State(),Action(0))))
	
	for i in range(numberOfEpisodes):
		print('Episode ' + str(i+1) + '\n')
		S0 = State(random.randint(1,10),random.randint(1,10))
		totalRewardTillNow = 0
		
		#print('Initial State: ' + str(S0))
		#input("")
		
		SARPath = []
		while not S0.isTerminalState():
			x,y = S0.getPlayerSum(),S0.getDealerSum()
			probability = random.uniform(0,1)
			action = Action()
			if probability < epsilon: # select random action
				move = 0 if random.uniform(0,1) < 0.50 else 1
				action = Action(move)
				#print('Action chosen randomly: ' + str(action))
				#input("")
				z = move
			else:
				move = 0 if (sum(getFeatures(State(x,y),Action(0))*weights) > sum(getFeatures(State(x,y),Action(1))*weights)) else 1
				action = Action(move)
				#print('Action chosen greedily: ' + str(action))
				#input("")
				z = move
				
			S1,reward = step(S0,action)
			SARPath.append([S0,action,reward,totalRewardTillNow])
			S0 = S1
			#print('Next state: ' + str(S0))
			#print('Reward received: ' + str(reward))
			#input("")
			totalRewardTillNow = totalRewardTillNow + reward
		
		# Episode ends here
		SARPath.append([S0,Action(0),0,totalRewardTillNow])
		pathLength = len(SARPath)
		for i in range(pathLength-1):
			S0 = SARPath[i][0]
			action = SARPath[i][1]
			rewardSum = 0
			totalTDReturn = 0
			for j in range(i,pathLength-1):
				rewardSum = rewardSum + SARPath[j][2]
				nextState = SARPath[j+1][0]
				[x,y] = nextState.getFeatures()
				if not nextState.isTerminalState():
					z = SARPath[j+1][1].getMove()
					totalTDReturn = totalTDReturn + (rewardSum + sum(getFeatures(State(x,y),Action(z))*weights)) * (lambdas**(j-i))
				else: totalTDReturn = totalTDReturn + (rewardSum) * (lambdas**(j-i))
				
			totalTDReturn = totalTDReturn * (1-lambdas)/(1-(lambdas**(pathLength-1-i)))
			
			
			[x,y] = S0.getFeatures()
			z = action.getMove()
			features = getFeatures(S0,action)
			# Update weights
			weightsCopy = weights
			for w in range(36):
				weights[w] = weightsCopy[w] - alpha*(sum(features*weightsCopy) - totalTDReturn)*features[w] 
			
			#print(str(x) + ',' + str(y) + ',' + str(z) + ' was ' + str(Q[x][y][z].value))
			#print('Changed to: ' + str(Q[x][y][z].value))
			#input("")
				
		#print(S0)
		#print(totalRewardTillNow)
		
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(1, 21, 1)
	y = np.arange(1, 10, 1)
	X, Y = np.meshgrid(x, y)
	
	Q = [[[QNode() for x in range(2)] for y in range(11)] for z in range(22)]
	for x in range(22):
			for y in range(11):
				for z in range(2):
					Q[x][y][z].value = sum(getFeatures(State(x,y),Action(z)) * weights)
								
	zs = np.array([(Q[x][y][0].value if Q[x][y][0].value > Q[x][y][1].value else Q[x][y][1].value) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)

	ax.plot_surface(Y, X, Z,cmap=cm.coolwarm)

	ax.set_ylabel('Player Sum')
	ax.set_xlabel('Dealer Card')
	ax.set_zlabel('Value')
	ax.set_title('V* using Sarsa with Linear Function Approximation')
	fig = plt.gcf()
	fig.savefig('ApproxSarsa.png')
	#plt.show()


MonteCarloControl()
SarsaControl()
SarsaControlFA()
	
	
	
	
	
	