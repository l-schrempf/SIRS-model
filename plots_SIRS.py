import sys
import numpy as np
import matplotlib.pyplot as plt
from SIRS import SIRSModel
import pandas as pd
from scipy import stats

class SIRSPlots(object):

    def __init__(self, N):
        # Set up the model
        self.N = N
        self.N2 = N * N # total number of sites
        
    def calc_I_var(self, I_values):
        avg_I = np.mean(I_values)
        norm_avg_I = avg_I / self.N2

        I_sq = I_values ** 2
        avg_I_sq = np.mean(I_sq)
        var = (avg_I_sq - avg_I ** 2) / self.N2
        return norm_avg_I, var
    
    def calc_error_var(self, I):
        vars = []
        for _ in range(1000):
            sample = np.random.choice(I, I.shape[0])
            _, var = self.calc_I_var(sample)
            vars.append(var)
        vars = np.array(vars)
        avg_var = np.mean(vars)
        vars_sq = vars ** 2
        avg_var_sq = np.mean(vars_sq)
        st_dev = np.sqrt(avg_var_sq - avg_var ** 2)
        return st_dev
        
    def run(self, ps):
        # Append data to lists
        Is = []
        vars = []
        model = SIRSModel(self.N, ps[0], 0.5, ps[0], 0) #initialise model
        model.prob_2 = 0.5
        for p1 in ps:
            model.prob_1 = p1
            for p3 in ps:
                model.prob_3 = p3
                I = model.measure(1000)
                # Append average and variance
                avg_I, var = self.calc_I_var(I)
                Is.append(avg_I)
                vars.append(var)
        return Is, vars
    
    def run_var(self, ps):
        vars = []
        var_errs = []
        model = SIRSModel(self.N, ps[0], 0.5, 0.5, 0) #initialise model
        for p1 in ps:
            model.prob_1 = p1
            I = model.measure(10000)
            # Append variance
            _, var = self.calc_I_var(I)
            vars.append(var)
            var_error = self.calc_error_var(I)
            var_errs.append(var_error)
        return vars, var_errs

    def run_perm_immun(self, fs):
        I_avg = []
        for frac in fs:
            model = SIRSModel(self.N, 0.5, 0.5, 0.5, frac)
            I = model.measure(1000)
            #print(I)
            avg_I = np.mean(I) / self.N2
            I_avg.append(avg_I)
        return I_avg


# Main entry point of the program
if __name__ == "__main__":
    # Set up and run the measurements and plotting script
    plots = SIRSPlots(50)

    # Plot contour for infected sites and variance
    ps = np.linspace(0,1,21)
    Is, vars = plots.run(ps)
    s = int(np.sqrt(len(Is)))
    Is = np.reshape(Is, (s,s))
    vars = np.reshape(vars, (s,s))

    df = pd.DataFrame(Is, index=[f'p1={p1}' for p1 in ps], columns=[f'p3={p3}' for p3 in ps])
    df.to_csv('data_Is.csv')

    df = pd.DataFrame(vars, index=[f'p1={p1}' for p1 in ps], columns=[f'p3={p3}' for p3 in ps])
    df.to_csv('data_vars.csv')
    
    X, Y = np.meshgrid(ps, ps)
    fig1, ax1 = plt.subplots()
    contour1 = ax1.contourf(X, Y, Is)
    cbar = fig1.colorbar(contour1)
    cbar.ax.set_ylabel('Fraction of infected sites')
    ax1.set_title("Phase Diagram of Infected Sites for Varying p1 and p3")
    ax1.set_ylabel("p_3")
    ax1.set_xlabel("p_1")
    plt.show()

    fig2, ax2 = plt.subplots()
    contour2 = ax2.contourf(X, Y, vars)
    cbar = fig2.colorbar(contour1)
    cbar.ax.set_ylabel('Normalised variance')
    ax2.set_title("Variance of Infected Sites for Varying p1 and p3")
    ax2.set_ylabel("p_3")
    ax2.set_xlabel("p_1")
    plt.show()

    # Set up for variance slice
    ps_var = np.linspace(0.2,0.5,13)
    vars, errs = plots.run_var(ps_var)

    df = pd.DataFrame({'variance': vars, 'errors': errs}, index=[f'p1={p1}' for p1 in ps_var])
    df.to_csv('data_vars_slice.csv')

    plt.scatter(ps_var, vars, marker='.', color="darkorchid")
    plt.errorbar(ps_var, vars, errs, color="darkorchid")
    plt.title("Slice of Normalised Variance of Infected Sites for p_2 = p_3 = 0.5")
    plt.ylabel("Variance")
    plt.xlabel("p_1")
    plt.show()

    # Run immunity
    num_steps = 11
    num_iter = 5
    fs = np.linspace(0, 1, num_steps)
    avg_Is = np.zeros((num_iter, *fs.shape))
    for i in range(num_iter):
        print(f'Running iteration {i}...')
        avg_I = plots.run_perm_immun(fs) # returns 11 vals in list
        avg_Is[i,] = avg_I
    avg_Is_mean = np.mean(avg_Is, axis=0)
    avg_Is_sem = stats.sem(avg_Is, axis=0)

    df = pd.DataFrame({'avg_I_mean': avg_Is_mean, 'sem': avg_Is_sem}, index=[f'frac={f1}' for f1 in fs])
    df.to_csv('data_avgI_fs.csv')
    
    plt.scatter(fs, avg_Is_mean, marker='.', color="darkorchid")
    plt.errorbar(fs, avg_Is_mean, avg_Is_sem, color="darkorchid")
    plt.title("Average Infected Sites vs Immunity Fraction")
    plt.ylabel("Infected sites")
    plt.xlabel("Fraction of immunity")
    plt.show()
