# IsaacLab Tutorial

This repository contains setup instructions and training pipelines for **IsaacLab** using NVIDIA Isaac Sim. It includes reinforcement learning (RL) environments such as **Bimanual Handover** and **Lie Manipulation** (Reach & Pick&Place).  

---

## 📦 Installation

### 1. Prerequisites
- [NVIDIA NGC account](https://ngc.nvidia.com/signin)  
- Docker with GPU support  
- API key from NGC (`Setup → Generate API Key`)  

### 2. Login to NGC
```bash
docker login nvcr.io
Username: $oauthtoken
Password: <API_KEY>
```

### 3. Download IsaacLab
```bash
git clone https://github.com/NVIDIA-Omniverse/IsaacLab.git
cd IsaacLab/docker/
```

Or download from [Google Drive](https://drive.google.com/drive/folders/155kTFGM4nkJq7VBDJaTDch0ApARkK9Jb?usp=drive_link).

### 4. Launch Docker
```bash
chmod +x ./isaaclab.sh
sudo python3 container.py start
sudo python3 container.py enter
```

To stop:
```bash
sudo python3 container.py stop
```

### 5. Install Dependencies
```bash
pip install wandb
```

---

## 🤖 RL Environments

### Clone Tasks
```bash
cd IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic
git clone https://github.com/AUROVA-LAB/aurova_reinforcement_learning
```

For Pick&Place:
```bash
git clone https://github.com/AUROVA-LAB/aurova_reinforcement_learning -b rl_manipulation_pcikplace_v4.2
```

---

## 🚀 Training

### Bimanual Handover
With GUI:
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1
```

Headless:
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1 --headless
```

Server mode (many envs, video enabled):
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1024 --enable_cameras --video --headless
```

### Lie Manipulation Reach
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1
```

Headless:
```bash
./isaaclab.sh -p source/extensions/.../train/train.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1 --headless
```

---

## 🧪 Testing

Move trained models to:
```
/workspace/isaaclab/source/extensions/.../rl_manipulation/train/
```

Run evaluation:
```bash
./isaaclab.sh -p source/extensions/.../train/eval.py --task Isaac-RL-Manipulation-Direct-reach-v0 --num_envs 1 --enable_cameras --model_dir <MODEL_PATH>
```

Pretrained models available on [Google Drive](https://drive.google.com/drive/folders/1un_rO9T07DCe0Gp4Fy-j5i9ecMWgr7WW?usp=drive_link).  

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

## 🐳 Docker Base (For Servers)

### Build (v4.2.0 - recommended)
```bash
sudo docker build -f docker/Dockerfile.base \
    --build-arg ISAACSIM_VERSION_ARG=4.2.0 \
    --build-arg ISAACSIM_ROOT_PATH_ARG=/isaac-sim \
    --build-arg ISAACLAB_PATH_ARG=/workspace/isaaclab \
    --build-arg DOCKER_USER_HOME_ARG=/root \
    -t isaac-lab-base .
```

### Run
```bash
sudo docker run --rm -it --gpus all \
    -e ACCEPT_EULA=Y \
    -e ISAACSIM_VERSION=4.2.0 \
    -v $(pwd)/source:/workspace/isaaclab/source \
    --network host --name my-isaac-lab-base --entrypoint /bin/bash \
    isaac-lab-base
```

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
