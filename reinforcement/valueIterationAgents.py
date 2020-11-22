# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp
import util
import random

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        max_values = self.values.copy()
        for i in range(self.iterations):
            for s in self.mdp.getStates()[1:]:
                poss_act = []
                for possible_action in self.mdp.getPossibleActions(s):
                    total = 0
                    for transition in self.mdp.getTransitionStatesAndProbs(s, possible_action):
                        total += transition[1] * (self.mdp.getReward(
                            s, possible_action, transition[0]) + self.discount*self.values[transition[0]])
                    poss_act.append(total)
                max_values[s] = max(poss_act)
            self.values = max_values.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        total = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            total += transition[1] * (self.mdp.getReward(state, action,
                                                         transition[0]) + self.discount * self.values[transition[0]])
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        total = 0
        qvalues = []
        for action in self.mdp.getPossibleActions(state):
            for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                total += transition[1] * (self.mdp.getReward(
                    state, action, transition[0]) + self.discount * self.values[transition[0]])
            qvalues.append(total)
            total = 0
        q = max(qvalues)
        maxi = []
        for i in range(len(qvalues)):
            if qvalues[i] == q:
                maxi.append(i)
        index = random.choice(maxi)

        return self.mdp.getPossibleActions(state)[index]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            index = i % len(self.mdp.getStates())
            state = self.mdp.getStates()[index]
            if not self.mdp.isTerminal(state):
                self.values[state] = max(
                    [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        all_states = self.mdp.getStates()

        p_queue = util.PriorityQueue()

        predecessor = collections.defaultdict(set)

        for state in all_states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for trans_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        predecessor[trans_state].add(state)

        for state in all_states:
            if not self.mdp.isTerminal(state):
                diff = self.values[state] - max([self.computeQValueFromValues(state, action)
                                                 for action in self.mdp.getPossibleActions(state)])
                diff = abs(diff)
                p_queue.update(state, -diff)

        counter = 0
        while counter < self.iterations and not p_queue.isEmpty():
            local_state = p_queue.pop()
            if not self.mdp.isTerminal(local_state):
                self.values[local_state] = max([self.computeQValueFromValues(
                    local_state, action) for action in self.mdp.getPossibleActions(local_state)])
            for p in predecessor[local_state]:
                if not self.mdp.isTerminal(local_state):
                    diff_compare_delta = abs(self.values[p] - max([self.computeQValueFromValues(p, action)
                                                                   for action in self.mdp.getPossibleActions(p)]))

                    if diff_compare_delta > self.theta:
                        p_queue.update(p, -diff_compare_delta)
            counter += 1
