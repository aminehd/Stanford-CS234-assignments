import unittest
import numpy as np
from vi_and_pi import (policy_evaluation, policy_rewards, policy_probs, state_action_probs,
      state_action_reward)

class PolicyEvaluation(unittest.TestCase):

  def setUp(self):
    self.P = {
      0: {
        0: [
          (1, 0, 1, False)
        ],
        1: [
          (1, 0, 0, False)
        ]
      }
    }
    self.nS = 1
    self.nA = 2

  def test_numpy_array(self):
    n = np.array([1])
    self.assertEqual(n[0], 1)

  def test_numpy_shape(self):
    arr1 = np.zeros(1)
    arr2 = np.zeros((1, 3))
    self.assertEqual(np.matmul(arr1, arr2).shape, (3,))
    self.assertEqual(arr2[0][2], 0)

  def test_policy_reward(self):
    policy1 = np.array([0])
    policy2 = np.array([1])

    rewards1 = policy_rewards(self.P, self.nS, policy1)
    self.assertTrue((rewards1 == np.array([1.])).all())

    rewards2 = policy_rewards(self.P, self.nS, policy2)
    self.assertTrue((rewards2 == np.array([0.])).all())

  def test_policy_probs(self):
    P = {
      0: {
        0: [
          (.5, 0),
          (.5, 1)
        ]
      },
      1: {
        0: [
          (.2, 0),
          (.8, 1)
        ]
      }
    }
    nS = 2
    policy = np.array([0, 0])
    probs = policy_probs(P, nS, policy)
    self.assertTrue((probs == np.array([
      [.5, .5],
      [.2, .8]
    ])).all())

  def test_one_state_two_actions(self):
    policy1 = np.array([0])
    policy2 = np.array([1])
    gamma = 0.9

    value1 = policy_evaluation(self.P, self.nS, self.nA, policy1, gamma)
    value2 = policy_evaluation(self.P, self.nS, self.nA, policy2, gamma)

    self.assertTrue(np.abs(np.array([1/(1 - gamma)]) - value1) < 1e-2)
    self.assertTrue(np.abs(np.array([0]) - value2) < 1e-2)

  def test_two_states_one_action(self):
    P = {
      0: { 0: [(1., 0, 1., False)] },
      1: { 0: [(1., 0, 0., False)] }
    }
    nS = 2
    nA = 1
    policy = np.array([0, 0])
    gamma = .9
    
    value = policy_evaluation(P, nS, nA, policy, gamma)
    self.assertTrue((np.abs(value - np.array([1/(1 - gamma), gamma/(1 - gamma)])) < 1e-2).all())

class PolicyImprovment(unittest.TestCase):
  def test_multiple_test_case(self):
    self.assertTrue(True) 

  def test_state_aciton_probs(self):
    P = {
      0: {
        0: [
          (.5, 0),
          (.5, 1)
        ]
      },
      1: {
        0: [
          (.2, 0),
          (.8, 1)
        ]
      }
    }
    probs0 = state_action_probs(P, 2, 0, 0)
    self.assertTrue((probs0 == np.array([.5, .5])).all())
    probs1 = state_action_probs(P, 2, 1, 0)
    self.assertTrue((probs1 == np.array([.2, .8])).all())

  def test_state_action_single_action_available(self):
    P = {
      0: {
        0: [
          (1, 0),
        ]
      },
      1: {
        0: [
          (.2, 0),
          (.8, 1)
        ]
      }
    }
    probs0 = state_action_probs(P, 2, 0, 0)
    self.assertTrue((probs0 == np.array([1., 0.])).all())

  def test_state_action_reward(self):
    P = {
      0: {
        0: [
          (.5, 0, 2, False),
          (.5, 1, 0, False)
        ]
      },
      1: {
        0: [
          (.2, 0, 10, False),
          (.8, 1, 5, False)
        ]
      }
    }   
    reward0 = state_action_reward(P, 0, 0)
    self.assertEqual(reward0, 1.)

    reward1 = state_action_reward(P, 1, 0)
    self.assertEqual(reward1, 6)

if __name__ == '__main__':
  unittest.main()
