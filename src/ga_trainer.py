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

""" Genetic Algorithm implementation to train a Neural Network """

import numpy as np


class GATrainer():
    """ GATrainer """
    def __init__(self, pop_size, mutation_rate, input_length, hidden_layer_length, cost_function):
        # Algorithm parameters
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        # Seed for the random number generator
        np.random.seed(1)
        # Variables to know how the weights matrix is shaped
        self.input_length = input_length
        self.hidden_layer_length = hidden_layer_length
        # Population lists
        self.population = []
        self.population_fitness = []
        # Function to call and compute the fitness
        self.cost_function = cost_function

    def chromosome_to_weights(self, chromosome):
        """ Reshape the array to get the weights """
        w0_length = self.input_length * self.hidden_layer_length
        w0 = chromosome[:w0_length].reshape((self.input_length, self.hidden_layer_length))
        w1 = chromosome[w0_length:]
        return np.array([w0, w1])
        

    def get_best_elemment(self):
        """ Return the best set of weights in the population """
        # Get the best element in the population
        best_pop_index = np.argmin(self.population_fitness)
        best_pop = self.population[best_pop_index]
        # Reshape the element to get the weights
        return chromosome_to_weights(best_pop)

    def init_population(self):
        """ Randomly initialize the population """
        # Length of each chromosome: Elements in L0 weights + elements in L1 weights
        chromosome_length = self.input_length * self.hidden_layer_length + self.hidden_layer_length
        # Initial values random between [-1,1]
        self.population = 2*np.random.random((self.population_size, chromosome_length)) - 1
        self.population_fitness = np.zeros(self.population_size)


    def selection(self):
        # TODO: Return the best elements from the current population (Tournament)
        pass

    def offspring_generation(self, winners):
        # TODO: Apply a crossover process to the winners subset and generate the offspring for the new generation
        pass

    def mutation(self):
        """ Randomly mutate some elements in the current population """
        for i in xrange(len(self.population)):
            if np.random.random() < mutation_rate:
                # Multiply the current value by a random value from a normal distribution with mean=0 and std=2
                self.population[i] = self.population[i] * np.random.normal(0,2,1)[0]

    def next_generation(self):
        """ Evaluate the fitness of the current generation and compute the next """
        # Compute fitness
        for i in xrange(self.population_size):
            w = chromosome_to_weights(self.population[i])
            self.population_fitness[i] = self.cost_function(w)
        # Select the best elements
        winners = selection()
        # Generate the population offspring
        self.population = offspring_generation(winners)
        # Mutate some elements
        mutation()