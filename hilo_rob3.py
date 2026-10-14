import time
import numpy as np
from rl_manipulation_obstacles.mpc_controller import *


def run_nmpc(
    controller,
    get_joint_positions,
    send_joint_velocities,
    stop_robot,
    max_steps=1000,
    control_dt=None,
    position_tolerance=0.01,
    orientation_tolerance=0.05,
    settle_steps=5,
):
    """
    Execute receding-horizon NMPC.

    Parameters
    ----------
    controller : dict
        Output of drop_NMPC_setup().

    get_joint_positions : callable
        Returns current joint positions, shape (6,), in radians.

    send_joint_velocities : callable
        Sends joint velocity commands, shape (6,), in rad/s.

    stop_robot : callable
        Stops the robot or sends zero joint velocities.

    max_steps : int
        Maximum number of control iterations.

    control_dt : float or None
        Control period. Defaults to the NMPC model's dt.

    position_tolerance : float
        Cartesian position error tolerance in meters.

    orientation_tolerance : float
        Orientation error tolerance in radians.

    settle_steps : int
        Number of consecutive successful measurements required
        before declaring convergence.

    Returns
    -------
    history : dict
        Recorded joint positions, velocity commands, and pose errors.
    """

    opti = controller["opti"]
    X = controller["X"]
    U = controller["U"]

    q_parameter = controller["q_measured"]
    u_previous_parameter = controller["u_previous"]

    T_ref = controller["T_ref"]

    dt = (
        controller["dt"]
        if control_dt is None
        else control_dt
    )

    if dt <= 0:
        raise ValueError("control_dt must be positive.")

    horizon = controller["horizon"]

    # Previous applied velocity command.
    u_previous = np.zeros(6)

    # Initial solver guesses.
    X_guess = np.tile(
        np.asarray(get_joint_positions(), dtype=float).reshape(6, 1),
        (1, horizon),
    )

    U_guess = np.zeros((6, horizon - 1))

    history = {
        "q": [],
        "u": [],
        "position_error": [],
        "orientation_error": [],
        "solve_time": [],
    }

    consecutive_successes = 0

    try:
        for step in range(max_steps):

            loop_start = time.perf_counter()

            # ------------------------------------------
            # 1. Read the latest robot state
            # ------------------------------------------
            q_measured = np.asarray(
                get_joint_positions(),
                dtype=float,
            ).reshape(6)

            if not np.all(np.isfinite(q_measured)):
                raise ValueError(
                    "Received invalid joint positions."
                )

            # ------------------------------------------
            # 2. Update NMPC parameters
            # ------------------------------------------
            opti.set_value(
                q_parameter,
                q_measured,
            )

            opti.set_value(
                u_previous_parameter,
                u_previous,
            )

            # ------------------------------------------
            # 3. Warm-start the optimization
            # ------------------------------------------
            opti.set_initial(X, X_guess)
            opti.set_initial(U, U_guess)

            # ------------------------------------------
            # 4. Solve NMPC
            # ------------------------------------------
            solve_start = time.perf_counter()

            solution = opti.solve()

            solve_time = time.perf_counter() - solve_start

            # ------------------------------------------
            # 5. Extract the first optimal command
            # ------------------------------------------
            U_opt = solution.value(U)
            X_opt = solution.value(X)

            u_cmd = np.asarray(
                U_opt[:, 0],
                dtype=float,
            ).reshape(6)

            if not np.all(np.isfinite(u_cmd)):
                raise RuntimeError(
                    "NMPC generated a non-finite command."
                )

            # ------------------------------------------
            # 6. Evaluate Cartesian pose error
            # ------------------------------------------
            pose_error = np.asarray(
                controller["pose_error"](q_measured).full()
            ).reshape(6)

            position_error = np.linalg.norm(
                pose_error[:3]
            )

            orientation_error = np.linalg.norm(
                pose_error[3:]
            )

            # ------------------------------------------
            # 7. Record results
            # ------------------------------------------
            history["q"].append(q_measured.copy())
            history["u"].append(u_cmd.copy())

            history["position_error"].append(
                position_error
            )

            history["orientation_error"].append(
                orientation_error
            )

            history["solve_time"].append(
                solve_time
            )

            print(
                f"Step {step:04d} | "
                f"position error={position_error:.4f} m | "
                f"orientation error={orientation_error:.4f} rad | "
                f"solve={solve_time * 1000:.1f} ms"
            )

            # ------------------------------------------
            # 8. Check convergence
            # ------------------------------------------
            if (
                position_error < position_tolerance
                and orientation_error < orientation_tolerance
            ):
                consecutive_successes += 1
            else:
                consecutive_successes = 0

            if consecutive_successes >= settle_steps:
                print("Target reached.")
                break

            # ------------------------------------------
            # 9. Apply the first optimal command
            # ------------------------------------------
            send_joint_velocities(u_cmd)

            # Remember the command for the next acceleration
            # constraint.
            u_previous = u_cmd.copy()

            # ------------------------------------------
            # 10. Shift the solution for warm-starting
            # ------------------------------------------
            X_guess = np.hstack([
                X_opt[:, 1:],
                X_opt[:, -1:],
            ])

            U_guess = np.hstack([
                U_opt[:, 1:],
                U_opt[:, -1:],
            ])

            # ------------------------------------------
            # 11. Maintain the control period
            # ------------------------------------------
            elapsed = time.perf_counter() - loop_start
            remaining = dt - elapsed

            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("NMPC interrupted by user.")

    except Exception as exc:
        print(f"NMPC stopped due to an error: {exc}")
        raise

    finally:
        # Always request the robot to stop when exiting.
        stop_robot()

    # Convert history lists into arrays.
    for key in history:
        history[key] = np.asarray(history[key])

    return history


controller = drop_NMPC_setup(
    ini=[0.0] * 6,
    ref=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0],
    dt=0.01,
    horizon=15,
)

history = run_nmpc(
    controller=controller,
    get_joint_positions=get_joint_positions,
    send_joint_velocities=send_joint_velocities,
    stop_robot=stop_robot,
    max_steps=1000,
    control_dt=0.01,
)