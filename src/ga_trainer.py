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

""" Genetic Algorithm implementation to train a Neural Network """

import numpy as np

RANDOM_SEED = 1


class GATrainer():
    """ GATrainer """
    def __init__(self, pop_size, mutation_rate, input_length, hidden_layer_length, cost_function):
        # Algorithm parameters
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        # Seed for the random number generator
        np.random.seed(RANDOM_SEED)
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
        
    def compute_fitness(self):
        """ Compute and update the fitness of each element """
        for i in xrange(self.population_size):
            w = self.chromosome_to_weights(self.population[i])
            self.population_fitness[i,0] = self.cost_function(w)


    def get_best_elemment(self):
        """ Return the best set of weights in the population """
        # Get the best element in the population
        best_pop_index = np.argmin(self.population_fitness[:,0])
        print "best_pop_index: {}".format(best_pop_index)
        best_pop = self.population[best_pop_index]
        print "best_pop: {}".format(best_pop)
        # Reshape the element to get the weights
        return (self.population_fitness[best_pop_index,0], self.chromosome_to_weights(best_pop))

    def init_population(self):
        """ Randomly initialize the population """
        # Length of each chromosome: Elements in L0 weights + elements in L1 weights
        chromosome_length = self.input_length * self.hidden_layer_length + self.hidden_layer_length
        # Initial values random between [-2,2]
        self.population = 4*np.random.random((self.population_size, chromosome_length)) - 2
        self.population_fitness = np.zeros((self.population_size, 2)) # First col = value, second col = index
        self.population_fitness[:,1] = range(self.population_size)
        self.compute_fitness()

    def tournament(self, tournament_size):
        """ Tournament process. Get a random subset of the population and return the best element """
        # Pick the selected elements indexes
        selected = np.random.choice(range(self.population_size), tournament_size, replace=False)
        winner = np.argmin(self.population_fitness[selected, 0]) # Winner index (in selected array)
        winner = self.population_fitness[selected[winner], 1] # Actual index (in global population array)
        return winner

    def selection(self):
        """ Return the best elements from the current population (by tournament) """
        # 20% of the new population will be elements from the current population
        number_of_winners = int(round(0.2 * self.population_size))
        winners = np.zeros(number_of_winners)
        # But the best element will automatically pass (VIP access)
        winners[0] = np.argmin(self.population_fitness[:,0])
        # Each round in the tournament will pick a 10% of the population
        tournament_size = int(round(0.1 * self.population_size))
        for i in xrange(1, number_of_winners):
            winner = self.tournament(tournament_size)
            while winner in winners:
                # Repeat until it founds an element not chosen already
                winner = self.tournament(tournament_size)
            winners[i] = winner
        return self.population[[int(x) for x in winners]]

    def global_mutation(self):
        """ Randomly mutate some elements in the current population """
        for i in xrange(self.population_size):
            for j in xrange(len(self.population[i])):
                if np.random.random() < self.mutation_rate:
                    # Multiply the current value by a random value from a normal distribution with mean=0 and std=2
                    self.population[i][j] = self.population[i][j] * np.random.normal(0,2,1)[0]

    def element_mutation(self, element):
        # TODO: Randomly mutate the given element
        pass

    def offspring_generation(self, winners):
        """ Apply a crossover process to the winners subset and generate the offspring for the new generation """
        offspring_length = self.population_size - len(winners)
        offspring = np.zeros((offspring_length, winners.shape[1]))
        # Populate the offspring array. Just copy and repeat the winners
        for i in xrange(offspring_length):
            # Copy a row fron winners subset
            offspring[i] = winners[i % len(winners)]
            # Mating process. Shuffle the offspring columns.
            # The generated weights are a combination of the winners weights
            np.random.shuffle(offspring[i])
        new_population = np.zeros((self.population_size, winners.shape[1]))
        new_population[:len(winners)] = winners
        new_population[len(winners):] = offspring
        return new_population        

    def next_generation(self):
        """ Compute the next generation and evaluate its fitness """
        # Select the best elements
        winners = self.selection()
        # Generate the population offspring
        self.population = self.offspring_generation(winners)
        # Mutate some elements
        #self.mutation()
        # Compute fitness
        self.compute_fitness()

def cost(x):
    """ Just to test some functions """
    return abs(np.mean(x[0]) + np.mean(x[1]))

if __name__ == '__main__':
    """ Playground (Test area) """
    tr = GATrainer(10,0.05,2,3,cost)
    tr.init_population()
    print "population: {}".format(tr.population)
    print "fitness: {}".format(tr.population_fitness)

    """
    for i in xrange(10):
        w = tr.chromosome_to_weights(tr.population[i])
        tr.population_fitness[i,0] = cost(w)

    print "fitness: {}".format(tr.population_fitness)
    winners = tr.selection()
    print "winners: {}".format(winners)

    tr.population = tr.offspring_generation(winners)
    print "new pop: {}".format(tr.population)

    tr.mutation()
    print "mutated pop: {}".format(tr.population)
    """
    tr.next_generation()
    print "population: {}".format(tr.population)
    print "population_fitness: {}".format(tr.population_fitness)
    print "best: {}".format(tr.get_best_elemment())
    tr.next_generation()
    print "population: {}".format(tr.population)
    print "best: {}".format(tr.get_best_elemment())
