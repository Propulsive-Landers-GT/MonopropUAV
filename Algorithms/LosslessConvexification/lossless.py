import numpy as np
import cvxpy as cp
import glob
import os
import csv
import matplotlib.pyplot as plt

class LosslessConvexTaylorSolver:
    def __init__(
        self,
        landing_point,
        initial_position,
        initial_velocity,
        glide_slope,
        max_velocity,
        dry_mass,
        fuel_mass,
        alpha,
        lower_thrust_bound,
        upper_thrust_bound,
        tvc_range_rad,
        delta_t,
        pointing_direction=np.array([0, 0, 1]),
        N=100
    ):
        self.g = np.array([0, 0, -9.80665])
        self.landing_point = landing_point
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.glide_slope = glide_slope
        self.max_velocity = max_velocity
        self.dry_mass = dry_mass
        self.fuel_mass = fuel_mass
        self.alpha = alpha
        self.lower_thrust_bound = lower_thrust_bound
        self.upper_thrust_bound = upper_thrust_bound
        self.tvc_range = tvc_range_rad
        self.delta_t = delta_t
        self.pointing_direction = pointing_direction / np.linalg.norm(pointing_direction)
        self.N = N

    def solve(self):
        m0 = self.dry_mass + self.fuel_mass
        m_dry = self.dry_mass

        # Normalized thrust bounds (conservative)
        sigma_min = self.lower_thrust_bound / m0
        sigma_max = self.upper_thrust_bound / m_dry

        # Decision variables
        x = cp.Variable((3, self.N + 1))
        v = cp.Variable((3, self.N + 1))
        w = cp.Variable(self.N + 1)  # log-mass
        u = cp.Variable((3, self.N))
        sigma = cp.Variable(self.N)

        constraints = []

        # Initial conditions
        constraints += [
            x[:, 0] == self.initial_position,
            v[:, 0] == self.initial_velocity,
            w[0] == np.log(m0)
        ]

        # Final conditions
        constraints += [
            x[:, -1] == self.landing_point,
            v[:, -1] == np.zeros(3),
            w[-1] >= np.log(m_dry)
        ]

        # Dynamics and constraints
        for k in range(self.N):
            # Discrete-time dynamics
            constraints += [
                x[:, k + 1] == x[:, k] + self.delta_t * v[:, k],
                v[:, k + 1] == v[:, k] + self.delta_t * (u[:, k] + self.g),
                w[k + 1] == w[k] - self.delta_t * self.alpha * sigma[k]
            ]
            
            # Thrust magnitude constraints
            z_0 = np.log(m0 + self.alpha * self.upper_thrust_bound * self.delta_t * k)
            constraints += [
                cp.norm(u[:, k], 2) <= sigma[k],
                self.lower_thrust_bound * np.exp(-z_0) * (1 - (w[k] - z_0) + ((w[k] - z_0**2)/2)) <= sigma[k],
                sigma[k] <= self.upper_thrust_bound * np.exp(-z_0) * (1 - (w[k] - z_0))
            ]

            # Thrust pointing constraint
            constraints += [
                cp.norm(u[:, k] - sigma[k] * self.pointing_direction, 2)
                <= sigma[k] * np.sin(self.tvc_range)  # Uses sin instead of tan
            ]
            # Max velocity constraint (optional)
            constraints += [
                cp.norm(v[:, k], 2) <= self.max_velocity
            ]

        # # Glide slope constraint (cone)
        # for k in range(self.N + 1):
        #     constraints += [
        #         cp.norm(x[:2, k], 2) <= (1.0 / self.glide_slope) * x[2, k]
        #     ]

        # Objective: minimize total fuel used (equivalent to maximizing final mass)
        objective = cp.Minimize(cp.sum(sigma) * self.delta_t)

        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.ECOS)

        if prob.status != cp.OPTIMAL:
            print("Problem status:", prob.status)
            raise RuntimeError("No feasible solution found.")

        # Recover mass trajectory
        mass_traj = np.exp(w.value)

        return x.value, v.value, mass_traj, u.value, sigma.value

if __name__ == "__main__":
    solver = LosslessConvexTaylorSolver(
        landing_point=np.array([0, 0, 0]),
        initial_position=np.array([0, 0, 50]),
        initial_velocity=np.array([0, 0, 0]),
        # landing_point=np.array([0, 0, 50]),
        # initial_position=np.array([0, 0, 0]),
        # initial_velocity=np.array([0, 0, 0]),
        glide_slope=-0.05,
        max_velocity=5,
        dry_mass=50,
        # fuel_mass=60,
        fuel_mass=22.59,
        alpha=1/(9.81 * 180),
        lower_thrust_bound=1000 * 0.4,
        upper_thrust_bound=1000,
        tvc_range_rad=np.radians(15),
        #delta_t=0.05,
        delta_t=0.5,
        pointing_direction=np.array([0, 0, 1]),
        N=100
    )

    t_min = solver.dry_mass * np.linalg.norm(solver.initial_velocity) / solver.upper_thrust_bound
    t_max = solver.fuel_mass / (solver.alpha * solver.lower_thrust_bound)
    
    N_min = int(t_min / solver.delta_t) + 1
    N_max = int(t_max / solver.delta_t)

    print(f"Time bounds: t_min = {t_min:.2f}s, t_max = {t_max:.2f}s")
    print(f"Number of time steps: N_min = {N_min}, N_max = {N_max}")
    
    min_fuel_used = float('inf')
    best_solution = None
    for N in range(N_min, N_max + 1):
        print(f"Trying with N = {N} time steps...")
        solver.N = N
        
        try:
            x, v, m, u, sigma = solver.solve()
            total_fuel_used = m[0] - m[-1]
            if total_fuel_used < min_fuel_used:
                min_fuel_used = total_fuel_used
                best_solution = (x, v, m, u, sigma)
            elif total_fuel_used > min_fuel_used:
                break
        except RuntimeError as e:
            print(f"Failed with N = {N}: {e}")
            continue

    # print("Optimal trajectory (positions):\n", best_solution[0])
    # print("Optimal velocities:\n", best_solution[1])
    # print("Optimal mass profile:\n", best_solution[2])
    # print("Optimal specific thrusts:\n", best_solution[3])
    # print("Optimal normalized thrust magnitudes:\n", best_solution[4])
    # print("Total fuel used:", total_fuel_used)
    # print(N, "time steps computed.")

    # if best_solution is not None:
    #     x, v, m, u, sigma = best_solution
    #     time = np.linspace(0, solver.N * solver.delta_t, solver.N)

    #     # Plot position trajectory in 3D
    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot(x[0], x[1], x[2], '-o', color='C0', lw=2, markersize=4)
    #     ax.set_title("Position Trajectory")
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")
    #     ax.legend()
    #     plt.grid()
    #     plt.tight_layout()
    #     plt.savefig("trajectory.png", dpi=150, bbox_inches="tight")
    #     # Save view from multiple angles
    #     angles = [
    #         (45, 45),   # Corner view
    #         (45, 135),  # Another corner
    #         (45, 225),  # Another corner 
    #         (45, 315),  # Another corner
    #         (0, 0),     # Side view
    #         (0, 90),    # Front view
    #         (90, 0),    # Top view
    #     ]
        
    #     for elevation, azimuth in angles:
    #         ax.view_init(elev=elevation, azim=azimuth)
    #         plt.savefig(f"trajectory_elev{elevation}_azim{azimuth}.png", 
    #                dpi=150, bbox_inches="tight")
    #     # # Plot velocity trajectory
    #     # plt.figure(figsize=(10, 6))
    #     # plt.plot(time, v[0, :], label="vx (East)")
    #     # plt.plot(time, v[1, :], label="vy (North)")
    #     # plt.plot(time, v[2, :], label="vz (Vertical)")
    #     # plt.title("Velocity Trajectory")
    #     # plt.xlabel("Time (s)")
    #     # plt.ylabel("Velocity (m/s)")
    #     # plt.legend()
    #     # plt.grid()

    #     # # Plot mass profile
    #     # plt.figure(figsize=(10, 6))
    #     # plt.plot(time, m, label="Mass")
    #     # plt.title("Mass Profile")
    #     # plt.xlabel("Time (s)")
    #     # plt.ylabel("Mass (kg)")
    #     # plt.legend()
    #     # plt.grid()

    #     # # Plot thrust magnitudes
    #     # thrust_time = np.linspace(0, solver.N * solver.delta_t, solver.N)
    #     # plt.figure(figsize=(10, 6))
    #     # plt.plot(thrust_time, sigma, label="Normalized Thrust Magnitude")
    #     # plt.title("Thrust Magnitude Profile")
    #     # plt.xlabel("Time (s)")
    #     # plt.ylabel("Normalized Thrust")
    #     # plt.legend()
    #     # plt.grid()

    #     # plt.show()
    # else:
    #     print("No solution found to graph.")

    print("Calculating optimal trajectory with finer resolution...")
    
    previous_solver_delta_t = solver.delta_t
    solver.delta_t = 0.05
    solver.N = N * int(previous_solver_delta_t / solver.delta_t)
        
    try:
        x, v, m, u, sigma = solver.solve()
        total_fuel_used = m[0] - m[-1]
        min_fuel_used = total_fuel_used
        best_solution = (x, v, m, u, sigma)
    except RuntimeError as e:
        print(f"Failed with N = {N}: {e}")

    print("Optimal trajectory (positions):\n", best_solution[0])
    print("Optimal velocities:\n", best_solution[1])
    print("Optimal mass profile:\n", best_solution[2])
    print("Optimal specific thrusts:\n", best_solution[3])
    print("Optimal normalized thrust magnitudes:\n", best_solution[4])
    print("Total fuel used:", total_fuel_used)
    print(N, "time steps computed.")

    if best_solution is not None:
        x, v, m, u, sigma = best_solution
        time = np.linspace(0, solver.N * solver.delta_t, solver.N + 1)
        
        # Save trajectory data to CSV (including actual thrust = mass * u)
        traj_filename = "trajectory_data.csv"

        # Compute actual thrust vector (N) at each timestep: thrust = mass * u
        # u has length N, while time/m has length N+1. Set thrust at final time to zero.
        thrust_vec = np.zeros((3, len(time)))
        for k in range(len(time) - 1):
            thrust_vec[:, k] = (u[:, k] * m[k])  # elementwise: accel * mass = force
        thrust_mag = np.linalg.norm(thrust_vec, axis=0)

        with open(traj_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
            "Time", "X", "Y", "Z",
            "Vx", "Vy", "Vz",
            "Mass",
            "Thrust_X", "Thrust_Y", "Thrust_Z", "Thrust_Mag"
            ])
            for i in range(len(time)):
                writer.writerow([
                    float(time[i]),
                    float(x[0, i]), float(x[1, i]), float(x[2, i]),
                    float(v[0, i]), float(v[1, i]), float(v[2, i]),
                    float(m[i]),
                    float(thrust_vec[0, i]), float(thrust_vec[1, i]), float(thrust_vec[2, i]),
                    float(thrust_mag[i])
                ])

        print(f"Trajectory data saved to {traj_filename}")


        # Plot position trajectory in 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[0], x[1], x[2], '-o', color='C0', lw=2, markersize=4)
        ax.set_title("Position Trajectory")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("trajectory.png", dpi=150, bbox_inches="tight")
        # Save view from multiple angles
        angles = [
            (45, 45),   # Corner view
            (45, 135),  # Another corner
            (45, 225),  # Another corner 
            (45, 315),  # Another corner
            (0, 0),     # Side view
            (0, 90),    # Front view
            (90, 0),    # Top view
        ]
        
        for elevation, azimuth in angles:
            ax.view_init(elev=elevation, azim=azimuth)
            plt.savefig(f"trajectory_elev{elevation}_azim{azimuth}.png", 
                   dpi=150, bbox_inches="tight")
        # # Plot velocity trajectory
        # plt.figure(figsize=(10, 6))
        # plt.plot(time, v[0, :], label="vx (East)")
        # plt.plot(time, v[1, :], label="vy (North)")
        # plt.plot(time, v[2, :], label="vz (Vertical)")
        # plt.title("Velocity Trajectory")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Velocity (m/s)")
        # plt.legend()
        # plt.grid()

        # # Plot mass profile
        # plt.figure(figsize=(10, 6))
        # plt.plot(time, m, label="Mass")
        # plt.title("Mass Profile")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Mass (kg)")
        # plt.legend()
        # plt.grid()

        # # Plot thrust magnitudes
        # thrust_time = np.linspace(0, solver.N * solver.delta_t, solver.N)
        # plt.figure(figsize=(10, 6))
        # plt.plot(thrust_time, sigma, label="Normalized Thrust Magnitude")
        # plt.title("Thrust Magnitude Profile")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Normalized Thrust")
        # plt.legend()
        # plt.grid()

        # plt.show()
    else:
        print("No solution found to graph.")
