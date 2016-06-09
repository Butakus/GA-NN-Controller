#!/usr/bin/python2

# Copyright (C) 2016  Francisco Miguel Moreno
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Neural Network with 2 layers of perceptrons to control a system

Network inputs: 6
    System current state (position, peed, acceleration)
    System target state (position, peed, acceleration)
Network Outputs: 1
    System control variable (theta)
"""

import numpy as np
from time import time
from ga_trainer import GATrainer
from system_model import State, SystemModel

# Maximum theta value allowed
MAX_THETA = 10.0
# Initial theta value
INITIAL_THETA = 0.0


def sigmoid(x):
    """ Sigmoid activation function """
    return .5 * (1 + np.tanh(.5 * x))


def cost_function(current_state, target_state):
    """ Return the ponderated sum of absolute errors of position, speed and acceleration """
    error = 5*abs(current_state.position - target_state.position)
    error += 2*abs(current_state.speed - target_state.speed)
    error += abs(current_state.accel - target_state.accel)
    return error


class NNController():
    """ Neural Network Controller """
    def __init__(self, system_model, initial_state, target_state, hidden_units=12):
        # Model to control
        self.system = system_model
        self.initial_state = initial_state
        self.target_state = target_state

        self.hidden_units = hidden_units

        # Trainer to optimize the weights
        # Params: population=100, mut_rate=5%, input_length=7 (6 from state + bias), hidden_length=hidden_units + bias, error_function
        self.trainer = GATrainer(50, 0.05, 7, self.hidden_units+1, self.compute_error)
        # initialize the trainer object (random initialization)
        self.trainer.init_population()
        # Update (get) the best weights from the trainer (random right now)
        self.weights = self.trainer.get_best_elemment()

    def save(self, filename):
        # TODO: Save weights and other params to file
        print "WARNING: Save method not implemented yet"
        pass

    def load(self, filename):
        # TODO: load weights and other params from file
        print "WARNING: load method not implemented yet"
        pass

    def l1_step(self, input_state):
        """ Compute the l1 step for all hidden units """
        l1_values = []
        for i in xrange(self.hidden_units):
            #print "input_state: {}".format(input_state)
            #print "self.weights[0][i]: {}".format(self.weights[0][:,i])
            #print "inner: {}".format(np.inner(input_state, self.weights[0][:,i]))
            value = sigmoid(np.inner(input_state, self.weights[0][:,i]))
            l1_values.append(value)
        return l1_values

    def l2_step(self, input_state):
        """ Compute the l2 step """
        l2_value = sigmoid(np.inner(input_state, self.weights[1]))
        return l2_value

    def action(self, current_state, target_state):
        """ Forward step. Compute the network output for the current weights """
        # Network input: current_state,  target_state and 1 for the bias weight
        x = np.zeros(7)
        x[:3] = current_state.list()
        x[3:6] = current_state.list()
        x[6] = 1
        # Compute the outputs of the hidden level units
        l1_values = self.l1_step(x)
        # Append a 1 at the end for the bias
        l1_values.append(1)
        # Compute the network output (final control value)
        new_theta = self.l2_step(l1_values)

        # Limit theta value
        if new_theta > MAX_THETA:
            new_theta = MAX_THETA
        elif new_theta < -MAX_THETA:
            new_theta = -MAX_THETA

        return new_theta

    def compute_error(self, weights, sim_steps=100):
        """ Simulate the system using the given weights and return the final error """
        # Set network weights
        self.weights = weights
        #print "weights: {}".format(self.weights)
        # Simulate the system
        current_theta = INITIAL_THETA
        current_state = self.initial_state
        self.system.reset_state(current_state)
        for i in xrange(sim_steps-1):
            next_theta = self.action(current_state, self.target_state)
            next_state = self.system.next(next_theta)
            current_theta = next_theta
            current_state = next_state

        # Get the error in the last state after simulating
        error = cost_function(current_state, self.target_state)
        return error

    def train(self, error_threshold=1e-6, max_iterations=10000):
        """ Train the network using the trainer object to optimize the weights """
        t = time()
        for i in xrange(max_iterations):
            # Compute the next generation of weights
            self.trainer.next_generation()

            # Check stop condition
            current_error, best_weights = self.trainer.get_best_elemment()
            if current_error < error_threshold:
                break

            # Show progress (only each 1000 iterations)
            if i % 10 == 0:
                print "Iter {it}. Current error: {err}".format(it=i, err=current_error)
                #print "Weights: {}".format(best_weights)
                print "Time elapsed: {}".format(time()-t)
                t = time()
        best_error, best_weights = self.trainer.get_best_elemment()
        print "Training finished."
        print "Error: {}".format(best_error)
        print "Weights: {}".format(best_weights)


if __name__ == '__main__':
    """ Main program to train the network and show the results """
    # Create the model
    dt = 0.1
    initial_state = State(0.0, 0.0, 0.0)
    system_model = SystemModel(initial_state, dt)

    # Try to control the system to reach the desired final state
    target_state = State(3.0, 0.0, 0.0) # Final position in x = 3

    nn = NNController(system_model, initial_state, target_state)


    t1 = time()
    nn.train()
    print "Time elapsed: {}".format(time() - t1)
