from abc import ABC, abstractmethod
import numpy as np
import csv

class MarkovProcessDataGenerator(ABC):
    """
    - emission_params: the list of parameters for different states in order as a np array of np arrays
    - states: the labels of hidden states/regimes, passed as a np array of numbers 
      starting at 0
    - initial_probs: the initial probabilities for the Markov chain
    - trans_matrix: the transitional matrix for the Markov chain
    - data_count: the number of data points to be generated
    """
    def __init__(self, emission_params, states, initial_probs, trans_matrix, data_count):
        self.emission_params = emission_params
        self.states = states
        self.initial_probs = initial_probs
        self.trans_matrix = trans_matrix
        self.data_count = data_count

    def sample_categorical(self, curr_state=0, init=False):
        if not init:
            trans_probs = self.trans_matrix[curr_state]
            sample_state = np.random.choice(self.states,p=trans_probs)
            return sample_state
        return np.random.choice(self.states,p=self.initial_probs)

    @abstractmethod
    def sample_dist(self, parameters):
        pass
    
    def generate_data(self):
        rows = []
        current_state = self.sample_categorical(init=True)
        for t in range(self.data_count):
            if t > 0:
                current_state = self.sample_categorical(
                    curr_state=current_state
                )
            parameters = self.emission_params[current_state]
            data = self.sample_dist(parameters)
            rows.append((t, current_state, data))
        return rows

    def write_csv(self, filename, rows):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "state", "data"])
            writer.writerows(rows)

class GaussianEmissionGenerator(MarkovProcessDataGenerator):
    def __init__(self, emission_params, states, initial_probs, trans_matrix, data_count):
        super().__init__(emission_params, states, initial_probs, trans_matrix, data_count)

    def sample_dist(self, parameters):
        mean, std = parameters
        return np.random.normal(loc=mean,scale=std)

    



        