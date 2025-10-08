# Lie Algebra mapping for robotic Reinforcement Learning manipulation

This repository contains setup instructions and training pipelines for the **Lie Algebra mapping for robotic Reinforcement Learning manipulation** (Reach & Pick&Place).  

---

## 🤖 Projects
### Install Reach Task
```bash
git clone https://github.com/AUROVA-LAB/aurova_reinforcement_learning -b rl_manipulation_reach_v4.2
```

### Install Pick&Place:
```bash
git clone https://github.com/AUROVA-LAB/aurova_reinforcement_learning -b rl_manipulation_pcikplace_v4.2
```

Pretrained REACH models available on [Google Drive](https://drive.google.com/drive/folders/1un_rO9T07DCe0Gp4Fy-j5i9ecMWgr7WW?usp=drive_link).  


#### Training (both)

In the **rl_manipulation_direct_env_cfg.py** there are different flags to change the representation:
| **Representation** | **Mapping** | **Distance** |
| --- | --- | --- |
| DQ | 0 - non map | 0 - DQ LOAM distance |
|  | 1 - Bruno | 1 - Geodesic |
| EULER | 0 - non map | 0 - Geodesic |
| QUAT | 0 - non map |  |
|  | 1 - Stereographic | 0 - Geodesic |
| MAT | 0 - non map |  |
|  | 1 - SE(3) mapping | 0 - Geodesic |

```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1
```

Headless:
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1 --headless
```

Server mode (many envs, video enabled):
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1024 --enable_cameras --video --headless
```

## 🧪 Testing

Move trained models to:
```
/workspace/isaaclab/source/extensions/.../rl_manipulation/train/
```

Run evaluation:
```bash
./isaaclab.sh -p source/extensions/.../train/eval.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1 --enable_cameras --model_dir <MODEL_PATH>
```

---

## ⚙️ Real Robot Deployment

Requires:
- PyTorch  
- Stable Baselines3  
- ROS Noetic  

### Run Agent
```bash
roslaunch lie_agent agent.launch
```

### Nodes
- `agent.py` → loads RL model, subscribes to `/agent/observation`, publishes `/agent/action`  
- `pre_agent.py` → computes pose differences and transforms them into velocities (`/agent/twist`)  

---

## 🦾 Importing New Assets
Follow [IsaacLab URDF Import Tutorial](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html).  

Example conversion:
```bash
rosrun xacro xacro --inorder -o model.urdf model.urdf.xacro
./isaaclab.sh -p source/standalone/tools/convert_urdf.py model.urdf model.usd --merge-joints --make-instanceable --fix-base
```

---

## 📚 References
- [IsaacLab Docs](https://isaac-sim.github.io/IsaacLab/main/index.html)  
- [Docker Compose Notes](https://www.notion.so/IsaacLab-s-Docker-Compose-Explanation-28188e8cb85581478009d2f654ef3707)  
- [Training & Tutorials](https://www.notion.so/Tutorials-28188e8cb855810ca9d6ee5b8671475f)  


---
## 📑 Citations
If you use this work in your research, please cite the following articles:



