import random
from environment import Agent, Environment, TrafficLight
from planner import RoutePlanner
from simulator import Simulator
import csv

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Create variable for distance traveled
        self.traveled = 0
        # Store all valid actions
        self.valid_actions = self.env.valid_actions[:]
        # Create Q Table for State, Action - Create a dictionary for all possible states - waypoints, light, oncoming, left, right,
        self.q_table = {}
        # waypoints: self.env.valid_actions[1:]
        # light: TrafficLight.valid_states
        # oncoming: self.env.valid_actions[:]
        # left: self.env.valid_actions[:]
        # right: self.env.valid_actions[:]
        # actions: self.env.valid_actions[:] 
        for w in self.env.valid_actions[1:]:
            for t in ['red','green']:
                for o in self.env.valid_actions[:]:
                    for l in self.env.valid_actions[:]:
                        for r in self.env.valid_actions[:]:
                            # Create unique key for the state
                            state_key = (w,t,o,l,r)
                            for a in self.env.valid_actions[:]:
                                # Use the action as second key in tuple
                                self.q_table[(state_key,a)] = 0
        # create variables for state, action, state_prime, action_prime
        s = ''
        a = ''
        r = ''
        s_prime = ''
        a_prime = ''
                                
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        
        # Reset distance traveled
        self.traveled = 0

    def update(self, t):
        '''
        Using Q-Learning method 2
        1) Sense the environment (see what changes occur naturally in the environment) - store it as state_0
        2) Take an action/reward - store as action_0 & reward_0
        
        In the next iteration
        1) Sense environment (see what changes occur naturally and from an action) - store as state_1
        2) Update the Q-table using state_0, action_0, reward_0, state_1
        3) Take an action - get a reward
        4) Repeat
        '''
        ### Step 3 - Implement a Q-Learning Driving Agent ###
        
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        
        # Sense the Environment        
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        

        # TODO: Learn policy based on state, action, reward

        # Establish Constants for the Q Learning update equation
        alpha = self.env.alpha
        gamma = self.env.gamma
        
        if t == 0:
            # States: waypoints, light, oncoming, left, right
            # Define the state
            self.s  = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
            # Choose an action            
            self.a = self.next_waypoint 
            # Execute action and get reward
            self.r = self.env.act(self, self.a)
        else:   
            # Define the new state
            self.s_prime  = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])
            # Find Max Q
            max_q = self.maximum_q(self.s_prime, self.valid_actions)
            #max_q = [0.0,1.0]
            # Calculate Q
            Q_value = (1.0-alpha)*self.q_table[(self.s, self.a)] + alpha*(self.r + gamma * max_q[1])
            # Update the Q table            
            self.q_table[(self.s, self.a)] = Q_value
            
            # TODO: Select action according to your policy
            # Take an action
            self.a = max_q[0]
            # Execute action and get reward
            self.r = self.env.act(self, self.a)
            # Replace s, a, r
            self.s = self.s_prime
            
        self.state = self.s


        # Add conditionals to capture distance traveled
        if self.next_waypoint != None:
            self.traveled += 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def maximum_q(self, state, possible_actions):
        result = [possible_actions[0], -100000.0]
        for a in possible_actions:
            if self.q_table[(state, a)] > result[1]:
                result = (a, self.q_table[(state, a)]) 
        return result

def run():
    """Run the agent for a finite number of trials."""
    
    # Define list of alphas (learning rate) to test 
    alphas = [0.25, 0.5, 0.75, 1.0] 
    #alphas = [0.25]
    # Define list of gammas (future discount factor) to test
    gammas = [0.0, 0.25, 0.5, 0.75, 1.0]
    #gammas = [0.25]

    # Create an array to capture all trial data
    trial_data = []
    trial_data.append(['reached','goal_distance', 'distance_traveled', 'steps', 'alpha', 'gamma'])
    
    for a in alphas:
        for g in gammas:
            # Set up environment and agent
            e = Environment(trial_data, alpha=a,gamma=g)  # create environment (also adds some dummy traffic - cyan)
            smartcab = e.create_agent(LearningAgent)  # create agent - red
            e.set_primary_agent(smartcab, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
            
            # Now simulate it
            sim = Simulator(e, update_delay=0.01, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False
        
            sim.run(n_trials=100)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
            
    with open('trials_a+g.csv', 'wb') as mycsvfile:
        thedatawriter = csv.writer(mycsvfile, delimiter=',')
        for row in trial_data:
            thedatawriter.writerow(row)

if __name__ == '__main__':
    run()
