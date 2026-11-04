import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import torch

names = ["DQ_1", "DQ_2", "DQ1", "DQ2", "MAT1", "MAT2", "TQ1", "TQ2"]

for name in names:
    # tensor: (N, 6)
    data = torch.load("evol_" + name + ".pth").cpu().numpy()

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

    for i in range(6):
        axes[i].plot(abs(data[:, i]))
        axes[i].set_ylabel(f'Component {i+1}')
        axes[i].grid(True)

    axes[-1].set_xlabel('Sample')

    plt.tight_layout()
    plt.savefig("evolution_" + name + ".png", dpi=300, bbox_inches="tight")
    plt.close()