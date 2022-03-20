import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class SIRSModel(object):
    def __init__(self, N, prob_1, prob_2, prob_3, frac) -> None:
        self.N = N
        self.prob_1 = prob_1
        self.prob_2 = prob_2
        self.prob_3 = prob_3
        self.frac = frac # fraction of immunity

        # Initialise 2D array of state consisting of 0, 1 and 2
        if frac == 0:
            self.state = np.random.choice(3, size=(self.N, self.N))
        else:
            self.state = self.make_immune_sites()

    def get_infected_nn(self, r, c):
        # Returns true if any neighbour has value 1
        # Faster than np.roll if just need to check for single cell
        left = self.state[r][(c-1) % self.N]
        right = self.state[r][(c+1) % self.N]
        top = self.state[(r-1) % self.N][c]
        bottom = self.state[(r+1) % self.N][c]
        return (left == 1) or (right == 1) or (top == 1) or (bottom == 1)

    def update_state(self):
        r = random.randrange(0, self.N)
        c = random.randrange(0, self.N)

        status = self.state[r][c]
        has_infected_neighbour = self.get_infected_nn(r, c)
        roll = np.random.uniform()

        if status == 0 and roll < self.prob_1: #susceptible
            if has_infected_neighbour:
                self.state[r][c] = 1  
        elif status == 1 and roll < self.prob_2: #infected
            self.state[r][c] = 2
        elif status == 2 and roll < self.prob_3: #recovered
            self.state[r][c] = 0

        return

    def sweep(self): # perform a single sweep over NxN random cells
        for _ in range(self.N*self.N):
            self.update_state()

    def run(self, n_sweeps):
        # Initialise figure, uncomment to plot
        fig = plt.figure()
        im = plt.imshow(self.state, animated=True, vmin=0, vmax=3)

        for i in range(n_sweeps):
            self.sweep()
            
            if (i % 5 == 0):
                #print(f"Step {i:d}")
                plt.cla()
                im = plt.imshow(self.state, animated=True, vmin=0, vmax=3)
                plt.draw()
                plt.pause(1e-10)
        return

    def calc_infected_sites(self):
        return np.sum(self.state == 1)

    def measure(self, n_sweeps):
        for _ in range(100): #equilibration
            self.sweep()

        values_I = []
        for _ in range(n_sweeps):
            self.sweep()
            I = self.calc_infected_sites()
            values_I.append(I)
        return np.array(values_I)

    def make_immune_sites(self):
        self.state = np.random.choice(3, size=(self.N, self.N))
        immune_sites = np.zeros(self.N ** 2, dtype=int)
        total_immune_sites = int(self.N ** 2 * self.frac)
        indices = np.random.choice(self.N ** 2, total_immune_sites, False)
        for i in indices:
            immune_sites[i] = 3
        immune_sites = np.reshape(immune_sites, (self.N,self.N))
        np.copyto(self.state, immune_sites, where=immune_sites != 0)
        #not_immune_indices = list(set(np.arange(self.N ** 2)) - set(indices)) # could be used to update faster
        return self.state

# Main entry point of the program
if __name__ == "__main__":
    
    # Read input arguments
    args = sys.argv
    if (len(args) != 5):
        print("Usage SIRS.py N prob1 prob2 prob3")
        sys.exit(1)

    N = int(args[1])
    prob_1 = float(args[2])
    prob_2 = float(args[3])
    prob_3 = float(args[4])

    # Set up and run the visualisation
    model = SIRSModel(N, prob_1, prob_2, prob_3, 0.3)
    model.run(1000)