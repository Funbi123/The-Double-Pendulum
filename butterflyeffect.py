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

# Run two simulations with almost identical initial conditions
times, test = RK4(50, state, 0.01)

state_ = state.copy()
state_[0] += 1e-6  # Change only theta1 by a tiny amount

times, test2 = RK4(50, state_, 0.01)


# Convert both simulations into Cartesian coordinates
x1 = length1 * np.sin(test[:, 0])
y1 = -length1 * np.cos(test[:, 0])
x2 = x1 + length2 * np.sin(test[:, 1])
y2 = y1 - length2 * np.cos(test[:, 1])

x1_2 = length1 * np.sin(test2[:, 0])
y1_2 = -length1 * np.cos(test2[:, 0])
x2_2 = x1_2 + length2 * np.sin(test2[:, 1])
y2_2 = y1_2 - length2 * np.cos(test2[:, 1])


# Set up the plot
fig, ax = plt.subplots(figsize=(9, 9))

ax.set_xlim(
    min(x1.min(), x2.min(), x1_2.min(), x2_2.min()) - 0.3,
    max(x1.max(), x2.max(), x1_2.max(), x2_2.max()) + 0.3
)
ax.set_ylim(
    min(y1.min(), y2.min(), y1_2.min(), y2_2.min()) - 0.3,
    max(y1.max(), y2.max(), y1_2.max(), y2_2.max()) + 0.3
)

ax.set_aspect("equal")
ax.set_title("Double Pendulum: Butterfly Effect")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot([0], [0], "ko", markersize=7)

# First pendulum
rod1, = ax.plot([], [], color="tab:blue")
rod2, = ax.plot([], [], color="tab:blue")
m1, = ax.plot([], [], "o", color="tab:blue", markersize=8)
m2, = ax.plot([], [], "o", color="tab:blue", markersize=8)
trail1, = ax.plot([], [], "--", color="tab:blue", alpha=0.6)

# Second pendulum
rod1b, = ax.plot([], [], color="tab:orange")
rod2b, = ax.plot([], [], color="tab:orange")
m1b, = ax.plot([], [], "o", color="tab:orange", markersize=8)
m2b, = ax.plot([], [], "o", color="tab:orange", markersize=8)
trail2, = ax.plot([], [], "--", color="tab:orange", alpha=0.6)


def update(frame):
    # First simulation
    rod1.set_data([0, x1[frame]], [0, y1[frame]])
    rod2.set_data([x1[frame], x2[frame]],
                  [y1[frame], y2[frame]])
    m1.set_data([x1[frame]], [y1[frame]])
    m2.set_data([x2[frame]], [y2[frame]])
    trail1.set_data(x2[:frame+1], y2[:frame+1])

    # Second simulation
    rod1b.set_data([0, x1_2[frame]], [0, y1_2[frame]])
    rod2b.set_data([x1_2[frame], x2_2[frame]],
                   [y1_2[frame], y2_2[frame]])
    m1b.set_data([x1_2[frame]], [y1_2[frame]])
    m2b.set_data([x2_2[frame]], [y2_2[frame]])
    trail2.set_data(x2_2[:frame+1], y2_2[:frame+1])

    return (
        rod1, rod2, m1, m2, trail1,
        rod1b, rod2b, m1b, m2b, trail2
    )


animation = FuncAnimation(
    fig,
    update,
    frames=range(0, len(test), 5),
    interval=10,
    blit=True
)

animation.save(
    "butterflyeffect.gif",
    writer="pillow",
    fps=30
)

plt.show()
