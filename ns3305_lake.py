import random
import math
import sys
import numpy


class FrozenLake(object):

    def __init__(self, width, height, start, targets, blocked, holes):
        self.initial_state = start 
        self.width = width
        self.height = height
        self.targets = targets
        self.holes = holes
        self.blocked = blocked

        self.actions = ('n', 's', 'e', 'w')
        self.states = set()
        for x in range(width):
            for y in range(height):
                if (x,y) not in self.targets and (x,y) not in self.holes and (x,y) not in self.blocked:
                    self.states.add((x,y))

        # Parameters for the simulation
        self.gamma = 0.9
        self.success_prob = 0.8
        self.hole_reward = -5.0
        self.target_reward = 1.0
        self.living_reward = -0.1

    #### Internal functions for running policies ###

    def get_transitions(self, state, action):
        """
        Return a list of (successor, probability) pairs that
        can result from taking action from state
        """
        result = []
        x,y = state
        remain_p = 0.0

        if action=="n":
            success = (x,y-1)
            fail = [(x+1,y), (x-1,y)]
        elif action=="s":
            success =  (x,y+1)
            fail = [(x+1,y), (x-1,y)]
        elif action=="e":
            success = (x+1,y)
            fail= [(x,y-1), (x,y+1)]
        elif action=="w":
            success = (x-1,y)
            fail= [(x,y-1), (x,y+1)]
          
        if success[0] < 0 or success[0] > self.width-1 or \
           success[1] < 0 or success[1] > self.height-1 or \
           success in self.blocked: 
                remain_p += self.success_prob
        else: 
            result.append((success, self.success_prob))
        
        for i,j in fail:
            if i < 0 or i > self.width-1 or \
               j < 0 or j > self.height-1 or \
               (i,j) in self.blocked: 
                    remain_p += (1-self.success_prob)/2
            else: 
                result.append(((i,j), (1-self.success_prob)/2))
           
        if remain_p > 0.0: 
            result.append(((x,y), remain_p))
        return result

    def move(self, state, action):
        """
        Return the state that results from taking this action
        """
        transitions = self.get_transitions(state, action)
        new_state = random.choices([i[0] for i in transitions], weights=[i[1] for i in transitions])
        return new_state[0]

    def simple_policy_rollout(self, policy):
        """
        Return (Boolean indicating success of trial, total rewards) pair
        """
        state = self.initial_state
        rewards = 0
        while True:
            if state in self.targets:
                return (True, rewards+self.target_reward)
            if state in self.holes:
                return (False, rewards+self.hole_reward)
            state = self.move(state, policy[state])
            rewards += self.living_reward

    def QValue_to_value(self, Qvalues):
        """
        Given a dictionary of q-values corresponding to (state, action) pairs,
        return a dictionary of optimal values for each state
        """
        values = {}
        for state in self.states:
            values[state] = -float("inf")
            for action in self.actions:
                values[state] = max(values[state], Qvalues[(state, action)])
        return values


    #### Some useful functions for you to visualize and test your MDP algorithms ###

    def test_policy(self, policy, t=500):
        """
        Following the policy t times, return (Rate of success, average total rewards)
        """
        numSuccess = 0.0
        totalRewards = 0.0
        for i in range(t):
            result = self.simple_policy_rollout(policy)
            if result[0]:
                numSuccess += 1
            totalRewards += result[1]
        return (numSuccess/t, totalRewards/t)

    def get_random_policy(self):
        """
        Generate a random policy.
        """
        policy = {}
        for i in range(self.width):
            for j in range(self.height):
                policy[(i,j)] = random.choice(self.actions)
        return policy

    def gen_rand_set(width, height, size):
        """
        Generate a random set of grid spaces.
        Useful for creating randomized maps.
        """
        mySet = set([])
        while len(mySet) < size:
            mySet.add((random.randint(0, width), random.randint(0, height)))
        return mySet


    def print_map(self, policy=None):
        """
        Print out a map of the frozen pond, where * indicates start state,
        T indicates target states, # indicates blocked states, and O indicates holes.
        A policy may optimally be provided, which will be printed out on the map as well.
        """
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")
        for j in range(self.height):
            sys.stdout.write("|")
            for i in range(self.width):
                if (i, j) in self.targets:
                    sys.stdout.write("T\t")
                elif (i, j) in self.holes:
                    sys.stdout.write("O\t")
                elif (i, j) in self.blocked:
                    sys.stdout.write("#\t")
                else:
                    if policy and (i, j) in policy:
                        a = policy[(i, j)]
                        if a == "n":
                            sys.stdout.write("^")
                        elif a == "s":
                            sys.stdout.write("v")
                        elif a == "e":
                            sys.stdout.write(">")
                        elif a == "w":
                            sys.stdout.write("<")
                        sys.stdout.write("\t")
                    elif (i, j) == self.initial_state:
                        sys.stdout.write("*\t")
                    else:
                        sys.stdout.write(".\t")
            sys.stdout.write("|")
            sys.stdout.write("\n")
        sys.stdout.write(" ")
        for i in range(2*self.width):
            sys.stdout.write("--")
        sys.stdout.write("\n")

    def print_values(self, values):
        """
        Given a dictionary {state: value}, print out the values on a grid
        """
        for j in range(self.height):
            for i in range(self.width):
                if (i, j) in self.holes:
                    value = self.hole_reward
                elif (i, j) in self.targets:
                    value = self.target_reward
                elif (i, j) in self.blocked:
                    value = 0.0
                else:
                    value = values[(i, j)]
                print("%10.2f" % value, end='')
            print()


    #### Your code starts here ###

    def value_iteration(self, threshold=0.001):
        """
        The value iteration algorithm to iteratively compute an optimal
        value function for all states.
        """
        values = dict((state, 0.0) for state in self.states)
        print("Below are the targets")


        while True:
            delta=0
            values_copy=values.copy()

            for s in self.states:
                final_value = []
                r=self.living_reward
                g=self.gamma

                for a in self.actions:
                    transition=lake.get_transitions(s,a)
                    #print(transition)
                    tot_value = 0

                    for cor,prob in transition:
                        state_reward=0
                        if cor in self.targets:
                            state_reward=self.target_reward
                        elif cor in self.holes:
                            state_reward=self.hole_reward
                        elif cor in self.blocked:
                            state_reward=0
                        else:
                            state_reward=values_copy[cor]

                        tot_value+= prob*(self.living_reward+g*state_reward)
                    final_value.append(tot_value)
                best_value = max(final_value)
                values[s] = best_value
                delta = max(delta,abs(values[s]-values_copy[s]))
                if delta < threshold:
                    #print(values)
                    return values_copy
        call_values=values
        return values

    def extract_policy(self, values):

        policy = {}
        total_value = 0
        for state in self.states:
            action_sum = []
            for action in self.actions:
                state_optimal = lake.move(state, action)
                transition_states = lake.get_transitions(state, action)
                trans_sum = 0
                for (next_state, probability) in transition_states:
                    state_reward = 0
                    if next_state in self.targets:
                        state_reward = self.target_reward
                    elif next_state in self.holes:
                        state_reward = self.hole_reward
                    else:
                        state_reward = values[next_state]

                    trans_sum += probability * (self.living_reward + self.gamma * state_reward)
                action_sum.append(trans_sum)
            max_action_index = numpy.argmax(action_sum)
            policy[state] = self.actions[max_action_index]

        return policy


    def Qlearner(self, alpha, epsilon, num_robots):
        """
        Implement Q-learning with the alpha and epsilon parameters provided.
        Runs number of episodes equal to num_robots.
        """
        robot_count=num_robots
        local_epsilon=epsilon
        local_alpha=alpha

        Qvalues = {}
        for state in self.states:
            for action in self.actions:
                Qvalues[(state, action)] = 0

        s=self.initial_state
        while robot_count > 0:

            #s= self.initial_state
            best_results=lake.get_best(Qvalues,s)
            max_action=best_results[0]

            random_action = random.choices(self.actions)[0]
            a = random.choices([random_action, max_action], [local_epsilon, 1-local_epsilon])[0]
            s_prime = lake.move(s,a)

            if s_prime not in self.states:

                if s_prime in self.targets:
                    max_value = self.target_reward
                elif s_prime in self.holes:
                    max_value = self.hole_reward


                Qvalues[(s, a)] = (1 - local_alpha) * Qvalues[(s, a)] + \
                                                  local_alpha * (self.living_reward + self.gamma*max_value)
                robot_count -= 1

                if robot_count==0.8*(num_robots):
                    local_epsilon=epsilon-(0.2*epsilon)
                    local_alpha=alpha-(0.2*alpha)
                elif robot_count==0.6*(num_robots):
                    local_epsilon=epsilon-(0.4*epsilon)
                    local_alpha=alpha-(0.4*alpha)
                elif robot_count==0.4*(num_robots):
                    local_epsilon=epsilon-(0.6*epsilon)
                    local_alpha=alpha-(0.6*alpha)
                elif robot_count==0.2*(num_robots):
                    epsilon=epsilon-(0.8*epsilon)
                    alpha=alpha-(0.8*alpha)

                s=self.initial_state

            else:
                best_value = lake.get_best(Qvalues, s_prime)
                max_value = best_value[1]
                Qvalues[(s, a)] = (1 - alpha) * Qvalues[(s, a)] + \
                                                  alpha * (self.living_reward + self.gamma*max_value)
                #print(Qvalues[(s, a)])
                s = s_prime

        return Qvalues

    def get_best(self, Qvalues,s):

        vdict={}

        for a in self.actions:
            vdict[a]=Qvalues[(s,a)]

        max_action=max(vdict.keys(), key=(lambda k: vdict[k]))
        max_value = vdict[max_action]
        return max_action, max_value







if __name__ == "__main__":
   
    # Create a lake simulation
    width = 8
    height = 8
    start = (0,0)
    targets = set([(3,4)])
    blocked = set([(3,3), (2,3), (2,4)])
    holes = set([(4, 0), (4, 1), (3, 0), (3, 1), (6, 4), (6, 5), (0, 7), (0, 6), (1, 7)])
    lake = FrozenLake(width, height, start, targets, blocked, holes)

    rand_policy = lake.get_random_policy()
    lake.print_map()
    lake.print_map(rand_policy)
    print(lake.test_policy(rand_policy))

    opt_values = lake.value_iteration()
    lake.print_values(opt_values)
    opt_policy = lake.extract_policy(opt_values)
    lake.print_map(opt_policy)
    print(lake.test_policy(opt_policy))
    #
    Q_values = lake.Qlearner(alpha=0.5, epsilon=0.5, num_robots=70)
    learned_values = lake.QValue_to_value(Q_values)
    learned_policy = lake.extract_policy(learned_values)
    lake.print_map(learned_policy)
    print(lake.test_policy(learned_policy))