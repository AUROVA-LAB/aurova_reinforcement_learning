import subprocess

# File to modify
file_path = "/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/rl_manipulation_direct_env_cfg.py"

command = """/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/rl_manipulation/train/train.py \
--task Isaac-RL-Manipulation-Direct-reach-v0 \
--num_envs 1024 \
--enable_cameras \
--video \
--headless
"""


def modify_cfg(representation, mapping, distance):
    # New values to set
    new_values = {
        "representation": representation,
        "mapping": mapping,
        "distance": distance,
    }

    # Read and modify
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("representation ="):
                file.write(f"    representation = {new_values['representation']}\n")
            elif line.strip().startswith("mapping ="):
                file.write(f"    mapping = {new_values['mapping']}\n")
            elif line.strip().startswith("distance ="):
                file.write(f"    distance = {new_values['distance']}\n")
            else:
                file.write(line)


if __name__ == "__main__":
    possib_repr = ["DQ", "EULER", "QUAT", "MAT"]
    possib_map =  [1, 1, 3, 1]
    possib_dist = [3, 1, 1, 1]

    for repr, map, dist in zip(possib_repr, possib_map, possib_dist):
        
        for m in range(map):
            for d in range(dist):
                
                m_ = m
                if repr == "DQ":
                    m_ += 2
                
                if repr == "QUAT":
                    m_ += 1
                if repr == "MAT":
                    m_ += 1

                modify_cfg("MAT", 0, 1)

                print("--- RUN: ", "MAT", " ", 0, " ", 1)
                subprocess.run(command, shell = True, check = True)
                
                raise
            raise
        raise








    
