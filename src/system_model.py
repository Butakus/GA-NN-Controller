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
Model of the system to control.
The states are computed step by step depending on the control variable.
"""

from math import sin, pi
from matplotlib.pyplot import *

# Globals
G =  9.80665


class State:
    """ System state variables """
    def __init__(self, position, speed, accel):
        self.position = position
        self.speed = speed
        self.accel = accel

    def __str__(self):
        return "Position, Speed, Accel: {}, {}, {}".format(self.position, self.speed, self.accel)

    def list(self):
        return [self.position, self.speed, self.accel]
        

class SystemModel:
    """ System model """
    def __init__(self, initial_state, delta_t):
        # Simulation time step
        self.delta_t = delta_t
        self.states = [initial_state]

    def reset_state(self, initial_state):
        """ Reset the list of states from the previous simulation """
        self.states = [initial_state]

    def next(self, theta):
        """ Compute and get the next system state after the delta_t time has passed """
        last_state = self.states[len(self.states) - 1]
        new_accel = G * sin(pi*theta/180)
        new_speed = new_accel * self.delta_t + last_state.speed
        new_position = new_accel * self.delta_t**2 / 2 + new_speed * self.delta_t + last_state.position
        new_state = State(new_position, new_speed, new_accel)
        self.states.append(new_state)
        return new_state

    def simulate(self, steps, theta_control):
        """ Make a full simulation from a list with the values of the control variable """
        # Sanitize inputs
        if len(theta_control) >= steps:
            theta_control = theta_control[:steps]
        elif len(theta_control) < steps:
            steps = len(theta_control)

        sim_states = [self.states[0]]
        for i in xrange(steps):
            sim_states.append(self.next(theta_control[i]))

        return sim_states

if __name__ == '__main__':
    """ Main function to test the behaviour of the SystemModel class. """
    # Create the model
    dt = 0.1
    state_0 = State(0.0, 0.0, 0.0)
    system_model = SystemModel(state_0, dt)

    # Simulate the system with different theta variations:
    steps = 100
    t = [float(i*dt) for i in range(0,steps)]
    #print "t: {}".format(t)

    # Theta variating
    theta_0 = [5.0]*int(steps/3)
    theta_0 += [-5.0]*(steps-len(theta_0))
    #print "theta_0: {}".format(theta_0)
    sim_0_states = system_model.simulate(steps, theta_0)
    positions = []
    for i in xrange(steps):
        #print sim_0_states[i]
        positions.append(sim_0_states[i].position)

    # Plot the result
    plot(t, positions)
    xlabel('Time (sec)')
    ylabel('Position (m)')
    title('Positions / time')
    #legend(('$x$ (m)', '$\dot{x}$ (m)'))
    show()
