
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, sin, cos, lambdify
from matplotlib.animation import FuncAnimation

# Constants
mass1 = 0.5
mass2 = 0.5
length1 = 1.
length2 = 1.
acc_gravity = 9.8

# Initial conditions
theta1 = np.radians(-150)
theta2 = np.radians(15)
v1, v2 = 0.0, 0.0

state = np.array([theta1, theta2, v1, v2], dtype=float)


# 1. Create symbolic variables for the formulas
th1_s, th2_s, w1_s, w2_s = symbols(
    'theta1 theta2 omega1 omega2'
)
a1_s, a2_s = symbols('a1 a2')

theta_change = th1_s - th2_s

# 2. Define the equations symbolically
eq1 = Eq(
    a1_s * (mass1 + mass2) * length1**2
    + mass2 * length1 * length2 * a2_s * cos(theta_change)
    + mass2 * length1 * length2 * w2_s**2 * sin(theta_change)
    + (mass1 + mass2) * acc_gravity * length1 * sin(th1_s),
    0
)

eq2 = Eq(
    a2_s * mass2 * length2**2
    + mass2 * length1 * length2 * a1_s * cos(theta_change)
    - mass2 * length1 * length2 * w1_s**2 * sin(theta_change)
    + mass2 * acc_gravity * length2 * sin(th2_s),
    0
)

# 3. Solve once, then convert the formulas into Python functions
solution = solve((eq1, eq2), (a1_s, a2_s))

acceleration1 = lambdify(
    (th1_s, th2_s, w1_s, w2_s),
    solution[a1_s],
    "numpy"
)

acceleration2 = lambdify(
    (th1_s, th2_s, w1_s, w2_s),
    solution[a2_s],
    "numpy"
)


# 4. Calculate derivatives using the CURRENT state
def derivatives(state, t):
    theta1 = state[0]
    theta2 = state[1]
    vel1 = state[2]
    vel2 = state[3]

    a1 = acceleration1(theta1, theta2, vel1, vel2)
    a2 = acceleration2(theta1, theta2, vel1, vel2)

    return np.array([vel1, vel2, a1, a2], dtype=float)


# 5. RK4 integrator
def RK4(t, state_0, h):
    times = np.arange(0, t, h)
    states = [state_0.copy()]

    for i in range(len(times) - 1):
        current = states[i]

        k1 = derivatives(current, times[i])
        k2 = derivatives(current + h*k1/2, times[i] + h/2)
        k3 = derivatives(current + h*k2/2, times[i] + h/2)
        k4 = derivatives(current + h*k3, times[i] + h)

        state_next = current + h*(k1 + 2*k2 + 2*k3 + k4)/6
        states.append(state_next)

    return times, np.array(states)


times, test = RK4(30, state, 0.01)


# 6. Convert angles into Cartesian coordinates
x1_values = length1 * np.sin(test[:, 0])
y1_values = -length1 * np.cos(test[:, 0])

x2_values = x1_values + length2 * np.sin(test[:, 1])
y2_values = y1_values - length2 * np.cos(test[:, 1])


# 7. Animation
fig, ax = plt.subplots(figsize=(8, 8))

xmin = min(x1_values.min(), x2_values.min()) - 0.3
xmax = max(x1_values.max(), x2_values.max()) + 0.3
ymin = min(y1_values.min(), y2_values.min()) - 0.3
ymax = max(y1_values.max(), y2_values.max()) + 0.3

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect("equal")
ax.set_title("Double Pendulum")
ax.set_xlabel("x")
ax.set_ylabel("y")

pivot, = ax.plot(
    [0], 
    [0], 
    "ko"
    )

rod1, = ax.plot(
    [], 
    [], 
    "-", 
    color="black"
    )

rod2, = ax.plot(
    [], 
    [], 
    "-", 
    color="black"
    )

m1, = ax.plot(
    [], 
    [], 
    "o", 
    markersize=10, 
    color="tab:blue"
    )

m2, = ax.plot(
    [], 
    [], 
    "o", 
    markersize=10, 
    color="tab:green"
    )


trail, = ax.plot(
    [], 
    [], 
    "--", 
    color="tab:green", 
    alpha=0.6
    )


def update(frame):
    x1, y1 = x1_values[frame], y1_values[frame]
    x2, y2 = x2_values[frame], y2_values[frame]

    rod1.set_data([0, x1], [0, y1])
    rod2.set_data([x1, x2], [y1, y2])

    m1.set_data([x1], [y1])
    m2.set_data([x2], [y2])

    trail.set_data(x2_values[:frame+1], y2_values[:frame+1])

    return rod1, rod2, m1, m2, pivot, trail


animation = FuncAnimation(
    fig,
    update,
    frames=len(test),
    interval=10,
    blit=True
)

animation.save(
    "Double Pendulum.gif",
    writer="pillow",
    fps=30
)

plt.show()
