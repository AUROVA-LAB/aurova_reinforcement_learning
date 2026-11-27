"""Compara políticas usando sus episodios originales, sin emparejar archivos.

Guarda este archivo junto a evaluar_episodios_organizado.py.
Ejemplo:
    python comparar_acoplamiento.py --root /ruta/exp7

Las acciones deben estar escaladas a radianes/metros. La métrica compara
decodificadores para una misma acción; no mide el error físico del robot.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_exp import (
    coupling_samples, evaluate_episode,
)


DEFAULT_ROOT = (
    "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/"
    "manager_based/aurova_reinforcement_learning/rl_manipulation/exp7"
)


def load_representation(folder, experiment, dt, action_order):
    files = sorted(p for p in folder.glob("*.pickle") if p.name != "0_results.pickle")
    if not files:
        raise ValueError(f"No hay episodios .pickle en {folder}")
    episodes, absolute, relative = [], [], []
    for path in files:
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        plot_data = []
        try:
            metrics = evaluate_episode(sample, experiment, plot_data, dt, action_order)
            actions = plot_data[0]["actions"]
            if actions is None:
                raise ValueError("Se necesitan saved_actions; utiliza un experimento >= 7")
            values = coupling_samples(actions)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Error en {path}: {exc}") from exc
        episodes.append({"representation": folder.name, "filename": path.name, **metrics})
        absolute.append(values["c"] * 1000)
        valid = values["translation"] > 1e-10
        relative.append(100 * values["c"][valid] / values["translation"][valid])
    return {"episodes": episodes, "absolute_mm": np.concatenate(absolute),
            "relative_percent": np.concatenate(relative)}


def ecdf(ax, values, label, color):
    values = np.sort(values[np.isfinite(values)])
    if len(values):
        ax.step(values, np.arange(1, len(values) + 1) / len(values),
                where="post", label=label, color=color, linewidth=1.8)


def save_figure(fig, output, name):
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def plot_action_distributions(datasets, colors, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for (name, data), color in zip(datasets.items(), colors):
        ecdf(axes[0], data["absolute_mm"], name, color)
        ecdf(axes[1], data["relative_percent"], name, color)
    axes[0].set_xlabel("Diferencia entre decodificadores c [mm]")
    axes[1].set_xlabel("Diferencia relativa por acción: 100 c / ‖ρ‖ [%]")
    for ax in axes:
        ax.set_ylabel("Fracción acumulada de acciones")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Sensibilidad geométrica de las acciones de cada política\n"
                 "Se agrupan acciones: los episodios largos aportan más muestras")
    save_figure(fig, output, "01_comparacion_acciones")


def plot_episode_distributions(datasets, colors, output):
    specs = [
        ("coupling_mean_mm", 1, "Diferencia media por episodio [mm]"),
        ("coupling_relative", 100, "100 Σc / Σ‖ρ‖ por episodio [%]"),
        ("rotation_per_action_mean_deg", 1, "Giro medio por acción [°]"),
        ("axis_angle_mean_deg", 1, "Ángulo medio entre ρ y eje de φ [°]"),
        ("end_ep", 1000, "Error final de posición [mm]"),
        ("end_er", 180 / np.pi, "Error final de orientación [°]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    rng = np.random.default_rng(14)
    for ax, (key, factor, ylabel) in zip(axes.flat, specs):
        for index, ((name, data), color) in enumerate(zip(datasets.items(), colors), 1):
            values = np.array([p[key] for p in data["episodes"]]) * factor
            values = values[np.isfinite(values)]
            if len(values):
                ax.boxplot([values], positions=[index], widths=0.4, showfliers=False)
                ax.scatter(index + rng.uniform(-0.12, 0.12, len(values)), values,
                           s=12, alpha=0.5, color=color)
        ax.set_xticks(range(1, len(datasets) + 1), list(datasets))
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Comparación por episodio: cada punto tiene el mismo peso\n"
                 "Las cajas muestran mediana e intervalo intercuartílico")
    save_figure(fig, output, "02_comparacion_episodios")


def plot_performance(datasets, colors, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        (axes[0, 0], "end_ep", 1000, "Error final de posición [mm]"),
        (axes[0, 1], "end_er", 180 / np.pi, "Error final de orientación [°]"),
        (axes[1, 0], "straightness_p", 1, "Rectitud de traslación hasta el corte"),
    ]
    rates, names = [], []
    for (name, data), color in zip(datasets.items(), colors):
        rows = data["episodes"]
        success = np.array([p["success"] for p in rows], dtype=bool)
        x = np.array([p["coupling_relative"] for p in rows]) * 100
        for ax, key, factor, ylabel in specs:
            y = np.array([p[key] for p in rows]) * factor
            for mask, marker, status in ((success, "o", "éxito"), (~success, "x", "fallo")):
                valid = mask & np.isfinite(x) & np.isfinite(y)
                if valid.any():
                    ax.scatter(x[valid], y[valid], marker=marker, s=24, alpha=0.65,
                               color=color, label=f"{name}: {status}")
            ax.set(xlabel="100 Σc / Σ‖ρ‖ por episodio [%]", ylabel=ylabel)
        names.append(f"{name}\n{int(success.sum())}/{len(rows)}")
        rates.append(100 * success.mean())
    for ax, *_ in specs:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1, 1].bar(names, rates, color=colors)
    axes[1, 1].set(ylabel="Éxito [%]", ylim=(0, 105))
    for index, rate in enumerate(rates):
        axes[1, 1].text(index, rate + 1, f"{rate:.1f}%", ha="center")
    fig.suptitle("Acoplamiento relativo y rendimiento: un punto por episodio\n"
                 "Asociación descriptiva; no demuestra causalidad ni compara semillas de entrenamiento")
    save_figure(fig, output, "03_acoplamiento_y_rendimiento")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(datasets):
    rows = []
    for name, data in datasets.items():
        episodes = data["episodes"]
        relative = np.array([p["coupling_relative"] for p in episodes]) * 100
        relative = relative[np.isfinite(relative)]
        c = data["absolute_mm"]
        row = {
            "representation": name,
            "episodes": len(episodes),
            "actions": len(c),
            "success_percent": 100 * np.mean([p["success"] for p in episodes]),
            "action_c_median_mm": float(np.median(c)),
            "action_c_p95_mm": float(np.percentile(c, 95)),
            "episode_relative_median_percent": float(np.median(relative)) if len(relative) else np.nan,
            "episode_relative_defined_count": len(relative),
            "episode_end_position_median_mm": float(np.median([p["end_ep"] for p in episodes]) * 1000),
            "episode_end_rotation_median_deg": float(np.degrees(np.median([p["end_er"] for p in episodes]))),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT), help="Carpeta que contiene SE3/ y T+SO3/")
    parser.add_argument("--representations", nargs="+", default=["SE3", "T+SO3"])
    parser.add_argument("--experiment", default="7")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--action-order", choices=["phi_rho", "rho_phi"], default="phi_rho")
    parser.add_argument("--output", type=Path, help="Por defecto: ROOT/comparacion_acoplamiento")
    args = parser.parse_args()
    if args.dt <= 0 or int(args.experiment) < 7:
        parser.error("Se necesita dt > 0 y un experimento >= 7 con saved_actions")
    if len(args.representations) < 2 or len(set(args.representations)) != len(args.representations):
        parser.error("Indica al menos dos representaciones distintas")
    datasets = {}
    for name in args.representations:
        datasets[name] = load_representation(args.root / name, args.experiment, args.dt, args.action_order)
    output = args.output or args.root / "comparacion_acoplamiento"
    output.mkdir(parents=True, exist_ok=True)
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(datasets))]
    plot_action_distributions(datasets, colors, output)
    plot_episode_distributions(datasets, colors, output)
    plot_performance(datasets, colors, output)
    write_csv(output / "metricas_por_episodio.csv",
              [episode for data in datasets.values() for episode in data["episodes"]])
    summary = summarize(datasets)
    write_csv(output / "resumen_comparacion.csv", summary)
    for row in summary:
        print(f"{row['representation']}: {row['episodes']} episodios; "
              f"éxito={row['success_percent']:.1f}%; "
              f"mediana c={row['action_c_median_mm']:.5f} mm; "
              f"mediana relativa por episodio={row['episode_relative_median_percent']:.3f}%")
    print("Comparación guardada en:", output)


if __name__ == "__main__":
    main()