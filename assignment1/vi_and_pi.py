### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
from numpy import linalg as LA
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)
def policy_rewards(P, nS, policy):
	R = np.zeros(nS)	
	for state in range(0, nS):
		touples = P[state][policy[state]]
		R[ state ] = 0
		for touple in touples:
			R[ state ] += touple[0] * touple[2]
		 			
		
	return R
def policy_probs( P, nS, policy):
		P_new = np.zeros((nS, nS))
		for s in range(0, nS):
			determined_action = policy[s]
			det_sto_state = P[s][determined_action]
			for entry in det_sto_state:
				s_prime = entry[1]
				prob = entry[0]
				P_new[s][ s_prime ] += prob 
		return P_new
				
def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	V = np.zeros(nS)
	R = policy_rewards(P, nS, policy)
	probs = policy_probs(P, nS, policy)
	V_old = np.repeat([ 2 * tol ] , nS)
	for  _ in range(0, max_iteration):
		done = (np.abs(V_old - V) < tol).all()
		if done: 
			break
		V_old = np.copy(V)

		for s in range(0, nS):
			V[s] = R[s] 
			for s_prime in range(0, nS):
				V[s] += gamma * probs[s][s_prime] * V_old[s_prime]
		


	#################done###########
	return V

def state_action_probs(P, nS, state, action ):
	policy = np.zeros(nS)
	policy[state] = action
	return policy_probs( P, nS, policy)[state]
def state_action_reward(P, state, action):
	tuples = P[state][action]
	rewards = np.array([ tup[0] * tup[2] for tup in tuples])
	return np.sum(rewards)
def state_rewards(P,  nA, state):
	rewards = np.zeros(nA)
	for a in range(0, nA):
		rewards[ a ] = state_action_reward(P, state, a)
	return rewards
	
def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
 	Q = np.zeros((nS, nA))
	for s in range(0, nS):
		for a in range(0, nA):
			x = state_action_reward(P, s, a) + gamma * np.matmul(state_action_probs(P, nS, s, a) , value_from_policy)				
			Q[s][a] = x	
	new_policy = [  np.argmax(Q[s]) for s in range(0, nS) ] 	
	############################
	return new_policy 

def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	old_policy = np.random.randint(0, nA, nS)
	############################
	for _ in range(max_iteration):
		V = policy_evaluation(P, nS, nA, old_policy, gamma, 1000, tol)
		new_policy = policy_improvement(P, nS, nA, V, old_policy, gamma)
		if ( old_policy - new_policy > 0).all():
			break
		old_policy = np.copy(new_policy)
	############################
	return V, new_policy

def Bellman_backup( P, nS, nA, V,  gamma = 0.9 ): 
	new_Values = np.zeros( nS)
	policy = np.zeros( nS, dtype = int ) 
	for s in range(0, nS):
		values_for_action =   [ state_action_reward(P, s, a) + gamma * np.matmul( state_action_probs( P, nS, s, a) , V) for a in range(0, nA)   ] 
		policy[ s ] = np.argmax( values_for_action )
		new_Values[ s ] = values_for_action[ policy[s] ] 

	return policy, new_Values
def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	old_V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	new_V = 0
	############################
	for _ in range(0, max_iteration):
		policy, new_V = Bellman_backup( P, nS, nA, old_V, gamma ) 
		if( (np.abs( new_V - old_V ) < tol ).all() ):
			break
		old_V = new_V
	############################
	return new_V , policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print( "Episode reward: %f" % episode_reward )


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print env.__doc__
	#example(env)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
	#V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)

	render_single(env, p_vi ) 
