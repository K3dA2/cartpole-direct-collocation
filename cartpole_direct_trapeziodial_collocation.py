import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#form mlp_toolkits.mplot3d import Axes3D

# Define the dynamics of the mass-spring-damper system
def dynamics(x, u):
    dx = np.zeros_like(x)
    q2 = x[:,1]
    q2_d = x[:,3]
    U = np.append(u, 0)
    dx[:,0] = x[:,2]
    dx[:,1] = x[:,3]
    dx[:,2] = ((l*m2*np.sin(q2)*q2_d**2)+U+(m2*g*np.cos(q2)*np.sin(q2)))/(m1 + m2*(1-(np.cos(q2))**2))
    dx[:,3] = -((l*m2*np.cos(q2)*np.sin(q2)*q2_d**2)+(U*np.cos(q2)) + ((m1+m2)*g*np.sin(q2)))/(l*m1 + l*m2*(1-(np.cos(q2))**2))
    print(dx.shape)
    return dx

# Define the objective function to minimize control effort
def objective(u):
    return np.sum(u**2)

# Define the dynamics defects
def dynamics_defects(decision_variables):
    u = decision_variables[:N]
    x = decision_variables[N:].reshape((N+1, states_dim))
    
    # Calculate the dynamics
    x_dot = dynamics(x, u)
    
    # Calculate the approximation of integral using trapezoidal quadrature
    integral = (x_dot[:-1] + x_dot[1:]) / 2 * dt
    
    # Calculate the state defects
    defects = []
    for i in range(N):
        defects.append(x[i+1] - x[i] - integral[i])
    return np.sum(np.array(defects))


# Define the direct collocation optimization problem
def optimization_problem(x0, xf, N):
    
    # Initial guess for control inputs
    u_init = np.zeros(N)
    u_init = u_init
    
    # Initial guess for states
    x_init = np.zeros((N+1, states_dim))
    x_init[:, 0] = np.linspace(x0[0], xf[0], N+1)
    x_init[:, 1] = np.linspace(0, np.pi, N+1)
    # Concatenate control inputs and states into a single decision variable
    initial_guess = np.concatenate([u_init, x_init.flatten()])
    
    # Define the optimization problem
    def problem(decision_variables):
        u = decision_variables[:N]
       
        obj_value = objective(u)
        
        return obj_value
    
    # Define the bounds for the decision variables
    bounds = [(-20, 20)] * N 
    state_bounds = [(None, None)] * (states_dim * (N+1))

    for i in range(0,N+1):
        state_bounds[4*i] = (l_b,u_b)

    bounds = bounds + state_bounds    
    print(np.array(bounds).shape)

    #Enforcing Bound constraint on initial and final states
    bounds[10] = (0.0,0.0)
    bounds[11] = (0.0,0.0)
    bounds[12] = (0.0, 0.0)
    bounds[13] = (0.0,0.0)
    bounds[50] = (1.0,1.0)
    bounds[51] = (np.pi,np.pi)
    bounds[52] = (0.0,0.0)
    bounds[53] = (0.0,0.0)

    # Define the constraints
    constraints = {'type': 'eq', 'fun': dynamics_defects}
    
    # Solve the optimization problem
    result = minimize(problem, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result

# Define the initial and final states
x0 = [0.0, 0.0, 0.0, 0.0]  # initial position and velocity
xf = [1, np.pi, 0.0, 0.0]  # final position and velocity
states_dim = 4


# Define the constants
l = 0.5
m1 = 1
m2 = 0.3
g = 9.81
T = 2

#bounds
u_b = 5
l_b = -5

# Define the number of time steps
N = 10
t = np.linspace(0, T, N+1)  # time grid
dt = t[1] - t[0]  # time step

# Solve the optimization problem
result = optimization_problem(x0, xf, N)

# Extract the optimal control inputs and states
u_opt = result.x[:N]

x_opt = result.x[N:].reshape((N+1, states_dim))

# Print the optimal control inputs and states
print("Optimal control inputs:")
print(u_opt)
print("Optimal states:")
print(x_opt)

t_ = np.linspace(0, T, N+1)  # time grid
plt.scatter(t_,x_opt[:,1])
#plt.plot(t[:-1], u_opt)
plt.show()
