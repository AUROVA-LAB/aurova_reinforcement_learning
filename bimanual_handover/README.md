# Learning Dexterous Handover

This repository contains setup instructions and training pipelines for the **Learning Dexterous Handover**.  

---

## 📦 Installation

```bash
cd IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic
git clone https://github.com/AUROVA-LAB/aurova_reinforcement_learning -b bimanual_handover
```

## 🤖 Training
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


## 🧪 Testing

Move trained models to:
```
/workspace/isaaclab/source/extensions/.../rl_manipulation/train/
```

Run evaluation:
```bash
./isaaclab.sh -p source/extensions/.../train/eval.py --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1 --enable_cameras --model_dir <MODEL_PATH>
```

---

## 📚 References
- [IsaacLab Docs](https://isaac-sim.github.io/IsaacLab/main/index.html)  
- [Docker Compose Notes](https://www.notion.so/IsaacLab-s-Docker-Compose-Explanation-28188e8cb85581478009d2f654ef3707)  
- [Training & Tutorials](https://www.notion.so/Tutorials-28188e8cb855810ca9d6ee5b8671475f)  


---
## 📑 Citations
If you use this work in your research, please cite the following articles:

### Learning Dexterous Handover

```bibtex
@article{frau2025learning,
  title={Learning Dexterous Object Handover},
  author={Frau-Alfaro, Daniel and Casta{\~n}o-Amoros, Julio and Puente, Santiago and Gil, Pablo and Calandra, Roberto},
  journal={arXiv preprint arXiv:2506.16822},
  year={2025}
}


