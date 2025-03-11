import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

FILES = ["evaluation_DQ_small.json", "evaluation_EULER_small.json",
         "evaluation_DQ_norm.json", "evaluation_EULER_norm.json",
         "evaluation_DQ_cyl.json", "evaluation_EULER_cyl.json"]
# FILES = ["evaluation_DQ_small2.json", "evaluation_EULER_small2.json",
#          "evaluation_DQ_norm2.json", "evaluation_EULER_norm2.json",
#          "evaluation_DQ_cyl2.json", "evaluation_EULER_cyl2.json"]
SEL = [[2,7,10,14,20,27,48], [20,35,45,47,48],
       [7,10,42,46,48,51,59,60,62,68,69], [6,9,20,28,68,70,74,76],
       [4,6,7,9,10,29,31,33,40,46,56], [7,10,22,29,39,49,58,71]]
# SEL = [[7,20,22,24,25], [1,10,12,24,34],
#        [2,4,6,8,9,11,14], [0,20,22,25,59],
#        [5,15,16,20,24,31], [7,11,16,20,34,53]]
COLORS = ['red', 'green', 'violet', 'green', 'violet', 'violet']
LABEL = ["DQ small prism", "EULER small prism", 
         "DQ normal prism", "EULER normal prism",
         "DQ cylinder", "EULER cylinder"]
MAGN = [r"$d_{DQ}$",r"$||\vec t_1 - \vec t_2||_2$"  + " (m)", r"$||\mathcal{P}(\mathbf{\hat q_{{diff}}} - \mathbf{\hat I})||_2$"]

LS = ['-', "--"]
exp = 0
magn = 0

# FILES = ["evaluation_EULER_cyl_both.json", "evaluation_EULER_norm_both.json", "evaluation_EULER_small_both.json"]
# SEL = [[1,3,19,21,26,30,32,57,65,71,74], [5,7,27,31,35,37,45,52,55,57,60], [2,6,8,41,46,52,58,60,71]]
# MAGN = [r"$d_{EULER}$",r"$||\vec t_1 - \vec t_2||_2$"  + " (m)", r"$||\vec e_1 - \vec e_2||_2$"  + " (rad)"]
# LABEL = ["EULER w. Cylinder", "EULER w. Original Prism", "EULER w. Small Prism"]

FILES = ["evaluation_DQ_cyl.json", "evaluation_DQ_norm.json", "evaluation_DQ_small.json"]
SEL = [[4,6,7,9,10,29,31,33,40,46,56], [7,10,42,46,48,51,59,60,62,68,69], [2,7,10,14,20,27,48]]
MAGN = [r"$d_{DQ}$",r"$||\vec t_1 - \vec t_2||_2$"  + " (m)", r"$||\mathcal{P}(\mathbf{\hat q_{{diff}}} - \mathbf{\hat I})||_2$"]
LABEL = ["DQ w. Cylinder", "DQ w. Original Prism", "DQ w. Small Prism"]




sel = SEL[exp] # Lo que tengo apuntado está empezando en 1

for k in range(3):
    for i in range(3):
        # Load JSON
        with open(os.path.join("logs/2025-02-14_08-38-27", FILES[i]), "r") as file:
            data = json.load(file)

        # Extract trajectories
        trajectories = data["distances"]

        max_length = max(len(traj) for traj in trajectories)

        # Interpolate all trajectories to the max length
        resampled_trajectories = []
        for s in SEL[i]:

            x_old = np.linspace(0, 1, len(trajectories[s]))  # Original indices normalized
            x_new = np.linspace(0, 1, max_length)  # Target indices
            a = []
            for j in trajectories[s]:
                a.append(j[k])

            
            
            interpolator = interp1d(x_old, a, kind='linear')  # Linear interpolation
            resampled_trajectories.append(interpolator(x_new))  # Resample trajectory

        # Compute the mean trajectory
        mean_trajectory = np.mean(resampled_trajectories, axis=0)
        std_trajectory = np.std(resampled_trajectories, axis=0)

        start = 2
        mean_trajectory = mean_trajectory[start:]
        std_trajectory = std_trajectory[start:]

        plt.plot(range(max_length-start), mean_trajectory, linewidth=3, label=LABEL[i], color=COLORS[i], ls=LS[i%2])
        plt.fill_between(range(max_length-start),
                 mean_trajectory - std_trajectory,
                 mean_trajectory + std_trajectory,
                 color=COLORS[i],
                 alpha=0.1)

    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    

    plt.xlabel("Time Step", fontsize=40)
    plt.ylabel(MAGN[k], fontsize=40)

    plt.legend(fontsize=28)
    plt.show()


# Load JSON
with open(os.path.join("logs/2025-02-14_08-38-27", "evaluation.json"), "r") as file:
    data = json.load(file)

# Extract trajectories
trajectories = data["distances"]
traj_ = data["phase"]
sel = range(80)

# Plot each trajectory
plt.figure(figsize=(8, 5))
for i, sel_ in enumerate(sel):
    a = []

    b = []
    b_r = []
    b_t = []
    for j in trajectories[sel_]:
        a.append(j[2])

    for j in traj_[sel_]:
        b.append(j[0])
        b_t.append(j[1])
        b_r.append(j[2])
    
    a.remove(a[0])
    b.remove(b[0])
    
    plt.plot(range(len(a)), a, label=f'Trajectory {i+1}')
    # plt.plot(range(len(b)), b, label=f'Trajectory DQ {i+1}')
    # plt.plot(range(len(b_t)), b_t, label=f'Trajectory T {i+1}')
    plt.plot(range(len(b_r)), b_r, label=f'Trajectory R {i+1}')

    plt.xlabel("Time Step")
    plt.ylabel("Distance")
    plt.title("Trajectory Distances Over Time")
    plt.legend()
    plt.show()
