"""Evalúa episodios y guarda cinco figuras en la subcarpeta graficas.

Configura el experimento, el periodo de muestreo y el orden de las acciones en main().
Las bandas temporales son percentiles entre episodios, no intervalos de confianza.
"""

import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_numpy(value):
    """Acepta arrays NumPy y tensores, también si están en GPU."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def sample_rows(values, width):
    if isinstance(values, (list, tuple)):
        values = np.stack([to_numpy(value) for value in values])
    else:
        values = to_numpy(values)
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    if values.ndim != 2 or values.shape[1] != width or not len(values):
        raise ValueError(f"Se esperaba un array no vacío [T, {width}]; recibido {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Hay NaN o infinito en los datos")
    return values


def qmul(a, b):
    """Producto de Hamilton, escalar primero; admite lotes."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), axis=-1)


def qconj(q):
    return q * np.array([1., -1., -1., -1.])


def rotation_vectors(q):
    """Log SO(3) de menor ángulo; q y -q dan el mismo giro salvo eje a pi."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1)
    theta = 2 * np.arctan2(s, np.clip(q[..., 0], 0., 1.))
    factor = np.divide(theta, s, out=np.full_like(s, 2.), where=s > 1e-12)
    return v * factor[..., None]


def pose_errors(poses, target):
    ep = np.linalg.norm(poses[:, :3] - target[:3], axis=1)
    er = np.linalg.norm(rotation_vectors(qmul(poses[:, 3:], qconj(target[3:]))), axis=1)
    return ep, er


def error(p1, p2):
    return pose_errors(to_numpy(p1), to_numpy(p2).reshape(7))


def norm_stats(out, name, vectors, weights, unit):
    magnitudes = np.linalg.norm(vectors, axis=1)
    keys = ("mean", "rms", "p95", "max")
    if not len(magnitudes):
        out.update({f"{name}_{k}_{unit}": math.nan for k in keys})
        out[f"{name}_squared_integral"] = math.nan
        out[f"{name}_support_s"] = 0.
        return
    support = float(np.sum(weights))
    out[f"{name}_mean_{unit}"] = float(np.sum(weights*magnitudes)/support)
    out[f"{name}_rms_{unit}"] = float(np.sqrt(np.sum(weights*magnitudes**2)/support))
    out[f"{name}_p95_{unit}"] = float(np.percentile(magnitudes, 95))
    out[f"{name}_max_{unit}"] = float(np.max(magnitudes))
    out[f"{name}_squared_integral"] = float(np.sum(weights*magnitudes**2))
    out[f"{name}_support_s"] = support


def coupling_samples(actions, tau_p=1e-5, tau_r=1e-4):
    """
    actions: [T, 6], orden usado aquí: [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z].
    Comprueba que saved_actions utiliza este orden y está escalado a unidades físicas.
    rho: metros; phi: vector de rotación en radianes.

    tau_p y tau_r sólo identifican acciones combinadas significativas.
    """
    actions = np.asarray(actions, dtype=float)

    if actions.ndim != 2 or actions.shape[1] != 6 or len(actions) == 0:
        raise ValueError("Se necesita un array no vacío de forma [T, 6]")

    if not np.all(np.isfinite(actions)):
        raise ValueError("Las acciones contienen NaN o infinito")

    phi = actions[:, :3]
    rho = actions[:, 3:]

    translation = np.linalg.norm(rho, axis=1)
    theta = np.linalg.norm(phi, axis=1)

    # A = (1-cos(theta))/theta², estable incluso para theta = 0.
    A = 0.5 * np.sinc(theta / (2 * np.pi))**2

    # B = (theta-sin(theta))/theta³.
    B = np.empty_like(theta)
    small = theta < 1e-4

    B[small] = (
        1/6
        - theta[small]**2 / 120
        + theta[small]**4 / 5040
    )
    B[~small] = (
        (theta[~small] - np.sin(theta[~small]))
        / theta[~small]**3
    )

    cross = np.cross(phi, rho)

    difference = (
        A[:, None] * cross
        + B[:, None] * np.cross(phi, cross)
    )

    c = np.linalg.norm(difference, axis=1)

    combined = (translation > tau_p) & (theta > tau_r)

    # Ángulo con el eje, entre 0° y 90°:
    # paralelo y antiparalelo cuentan ambos como alineados.
    axis_angles = np.full(len(actions), np.nan)
    if np.any(combined):
        cos_angle = (
            np.sum(rho[combined] * phi[combined], axis=1)
            / (translation[combined] * theta[combined])
        )
        axis_angle = np.degrees(
            np.arccos(np.clip(np.abs(cos_angle), 0, 1))
        )
        axis_angles[combined] = axis_angle

    return {"translation": translation, "theta": theta, "c": c,
            "combined": combined, "axis_angle": axis_angles}


def coupling_metrics(actions, tau_p=1e-5, tau_r=1e-4):
    values = coupling_samples(actions, tau_p, tau_r)
    translation, theta, c, combined, angles = (
        values[key] for key in ("translation", "theta", "c", "combined", "axis_angle")
    )
    alignment = float(np.mean(angles[combined])) if np.any(combined) else np.nan

    denominator = np.sum(translation)

    return {
        "coupling_mean_mm": float(np.mean(c) * 1000),
        "coupling_p95_mm": float(np.percentile(c, 95) * 1000),
        "coupling_relative": (
            float(np.sum(c) / denominator)
            if denominator > 1e-10 else np.nan
        ),
        "rotation_per_action_mean_deg": float(
            np.degrees(np.mean(theta))
        ),
        "rotation_per_action_p95_deg": float(
            np.degrees(np.percentile(theta, 95))
        ),
        "combined_actions_fraction": float(np.mean(combined)),
        "axis_angle_mean_deg": alignment,
    }


def motion_metrics(poses, t):
    """Diferencias en intervalos: v en puntos medios; a y jerk sucesivamente.

    Los cuadrados se integran sobre el soporte disponible de cada derivada.
    Para angular se diferencian vectores omega en el mismo marco fijo;
    diferenciar únicamente la norma de omega perdería cambios de dirección.
    """
    dp = np.diff(poses[:, :3], axis=0)
    dr = rotation_vectors(qmul(poses[1:, 3:], qconj(poses[:-1, 3:])))
    lp = float(np.sum(np.linalg.norm(dp, axis=1)))
    lr = float(np.sum(np.linalg.norm(dr, axis=1)))
    direct_p = float(np.linalg.norm(poses[-1, :3] - poses[0, :3]))
    direct_r = float(np.linalg.norm(rotation_vectors(qmul(poses[-1, 3:], qconj(poses[0, 3:])))))
    out = {"path_translation_m": lp, "path_angle_rad": lr,
           "endpoint_displacement_m": direct_p, "endpoint_rotation_rad": direct_r,
           "translation_straightness": direct_p/lp if lp > 1e-12 else math.nan,
           "angle_straightness": direct_r/lr if lr > 1e-12 else math.nan,
           "excess_translation_m": max(0., lp-direct_p),
           "excess_angle_rad": max(0., lr-direct_r)}
    dt = np.diff(t)
    vt = (t[1:]+t[:-1])/2
    at = (vt[1:]+vt[:-1])/2
    adt, jdt = np.diff(vt), np.diff(at)
    for name, increment, length in (("linear", dp, lp), ("angular", dr, lr)):
        v = increment/dt[:, None]
        a = np.diff(v, axis=0)/adt[:, None]
        j = np.diff(a, axis=0)/jdt[:, None]
        unit = "m" if name == "linear" else "rad"
        norm_stats(out, f"{name}_speed", v, dt, f"{unit}_s")
        norm_stats(out, f"{name}_acceleration", a, adt, f"{unit}_s2")
        norm_stats(out, f"{name}_jerk", j, jdt, f"{unit}_s3")
        out[f"{name}_velocity_total_variation_{unit}_s"] = float(np.sum(np.linalg.norm(np.diff(v, axis=0), axis=1))) if len(v) >= 2 else math.nan
        energy = out[f"{name}_jerk_squared_integral"]
        out[f"{name}_dimensionless_jerk_path"] = float(t[-1]**5*energy/length**2) if length > 1e-12 and np.isfinite(energy) else math.nan
    return out


def evaluate_episode(sample, n_experiment, plot_data=None, dt=0.01,
                     action_order="phi_rho"):
    """Calcula una vez todas las métricas de un episodio."""
    results = {}
    results["success"] = bool(to_numpy(sample["success"]).item())

    # Error a lo largo de la trayectoria
    trajectory = sample_rows(sample["trajectory"], 7)
    ep, er = error(trajectory, sample["tgt_pose"])

    # Error final
    results["end_ep"] = ep[-1]
    results["end_er"] = er[-1]

    # Error mínimo
    results["min_ep"] = np.min(ep)   
    results["min_er"] = np.min(er)    

    # Error máximo
    results["max_ep"] = np.max(ep)   
    results["max_er"] = np.max(er)

    # Se conserva el corte exclusivo trajectory[:step_reach]. Si vale cero,
    # se utiliza la trayectoria disponible en lugar de asumir 899 muestras.
    step_reach = int(to_numpy(sample["step_reach"]).item())
    stop = min(step_reach, len(trajectory)) if step_reach > 0 else len(trajectory)
    if stop < 1 or dt <= 0:
        raise ValueError("Se necesita al menos una pose y dt > 0")
    t = np.arange(stop) * dt
    motion = motion_metrics(trajectory[:stop], t)
    results.update(motion)

    results["straightness_p"] = motion["translation_straightness"]
    results["straightness_r"] = motion["angle_straightness"]
    results["distance_travelled"] = motion["path_translation_m"]
    results["rotation_travelled"] = motion["path_angle_rad"]
    results["endpoint_displacement_m"] = motion["endpoint_displacement_m"]
    results["endpoint_rotation_rad"] = motion["endpoint_rotation_rad"]



    actions = None
    if int(n_experiment) >= 3:
        actions = sample_rows(sample["saved_actions"], 6)[:stop]
        if action_order == "rho_phi":
            actions = np.concatenate((actions[:, 3:], actions[:, :3]), axis=1)
        elif action_order != "phi_rho":
            raise ValueError("action_order debe ser phi_rho o rho_phi")
        c_metrics = coupling_metrics(actions)

        results["coupling_mean_mm"] = c_metrics["coupling_mean_mm"]
        results["coupling_p95_mm"] = c_metrics["coupling_p95_mm"]
        results["coupling_relative"] = c_metrics["coupling_relative"]
        results["rotation_per_action_mean_deg"] = c_metrics["rotation_per_action_mean_deg"]
        results["rotation_per_action_p95_deg"] = c_metrics["rotation_per_action_p95_deg"]
        results["combined_actions_fraction"] = c_metrics["combined_actions_fraction"]
        results["axis_angle_mean_deg"] = c_metrics["axis_angle_mean_deg"]
    if plot_data is not None:
        plot_data.append({
            "ep": ep, "er": er, "trajectory": trajectory[:stop],
            "target": to_numpy(sample["tgt_pose"]).reshape(7),
            "success": results["success"], "actions": actions,
        })
    return results


def aggregate_results(episode_results):
    """Calcula las medias entre episodios, después de terminar el bucle."""
    if not episode_results:
        raise ValueError("No hay episodios para evaluar")

    return {
        key: float(np.mean([episode[key] for episode in episode_results]))
        for key in episode_results[0]
    }


def save_results(results, exp_path):
    """Guarda el resumen una sola vez, después de calcular las medias."""
    with open(os.path.join(exp_path, "0_results.pickle"), "wb") as handle:
        pickle.dump(results, handle)


def save_figure(fig, folder, name):
    """PNG para inspección y PDF vectorial para exportar."""
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(folder, name + "." + extension), dpi=180)
    plt.close(fig)


def temporal_band(ax, series, dt, scale, ylabel):
    """Sin rellenar el final de episodios cortos ni normalizar el tiempo."""
    count = max(map(len, series))
    values = np.full((len(series), count), np.nan)
    for index, row in enumerate(series):
        values[index, :len(row)] = np.asarray(row) * scale
    low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
    time = np.arange(count) * dt
    ax.fill_between(time, low, high, alpha=0.25, label="P25–P75 entre episodios")
    ax.plot(time, median, label="Mediana")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    return np.sum(np.isfinite(values), axis=0)


def plot_errors(plot_data, dt, folder, label):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    counts = temporal_band(axes[0], [p["ep"] for p in plot_data], dt, 1000,
                           "Error de posición [mm]")
    temporal_band(axes[1], [p["er"] for p in plot_data], dt, 180 / np.pi,
                  "Error de orientación [°]")
    axes[2].step(np.arange(len(counts)) * dt, counts, where="post")
    axes[2].set_ylabel("Episodios disponibles")
    axes[2].set_xlabel("Tiempo físico desde el inicio [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(label + " · Errores en la trayectoria completa registrada")
    save_figure(fig, folder, "01_errores_tiempo")


def metric_panels(episode_results, specifications, folder, name, title):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    success = np.array([p["success"] for p in episode_results], dtype=bool)
    for ax, (key, scale, ylabel) in zip(axes.flat, specifications):
        values = np.array([p.get(key, np.nan) for p in episode_results]) * scale
        for position, mask, color in ((1, ~success, "#c55a38"),
                                      (2, success, "#258276")):
            group = values[mask & np.isfinite(values)]
            if len(group):
                ax.boxplot([group], positions=[position], widths=0.4,
                           showfliers=False)
                # Desplazamiento horizontal determinista, sin alterar los valores.
                offsets = np.random.default_rng(14).uniform(-0.12, 0.12, len(group))
                ax.scatter(position + offsets, group, s=10, alpha=0.5, color=color)
        ax.set_xticks([1, 2], ["Fallidos", "Éxitos"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        if not np.any(np.isfinite(values)):
            ax.text(0.5, 0.5, "Sin datos definidos", transform=ax.transAxes,
                    ha="center")
    fig.suptitle(title)
    save_figure(fig, folder, name)


def plot_trajectories(plot_data, folder, label, max_episodes=60):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    indices = np.linspace(0, len(plot_data) - 1,
                          min(max_episodes, len(plot_data)), dtype=int)
    seen = set()
    all_points = []
    for index in indices:
        item = plot_data[index]
        xyz, target = item["trajectory"][:, :3], item["target"][:3]
        status = "Éxito" if item["success"] else "Fallo"
        color = "#258276" if item["success"] else "#c55a38"
        ax.plot(*xyz.T, color=color, alpha=0.45, linewidth=0.8,
                label=status if status not in seen else None)
        seen.add(status)
        ax.scatter(*target, marker="*", color="black", s=30,
                   label="Objetivo" if "target" not in seen else None)
        seen.add("target")
        ax.scatter(*xyz[0], marker="o", color=color, s=8)
        all_points.extend((xyz, target[None, :]))
    # Una misma escala física en los tres ejes.
    points = np.concatenate(all_points)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2, 1e-3)
    for setter, midpoint in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(midpoint - radius, midpoint + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    ax.legend()
    fig.suptitle(f"{label} · Trayectorias hasta el corte ({len(indices)} episodios)")
    save_figure(fig, folder, "04_trayectorias_3d")


def plot_coupling(plot_data, folder, label):
    actions = [p["actions"] for p in plot_data if p["actions"] is not None]
    if not actions:
        return
    values = coupling_samples(np.concatenate(actions))
    c = values["c"] * 1000
    theta = np.degrees(values["theta"])
    translation = values["translation"] * 1000
    angles = values["axis_angle"]
    # Todas las acciones entran en la ECDF; los scatter usan hasta 20 000 puntos.
    indices = np.linspace(0, len(c) - 1, min(len(c), 20000), dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].step(np.sort(c), np.arange(1, len(c) + 1) / len(c), where="post")
    axes[0, 0].set(xlabel="‖V(φ)ρ − ρ‖ [mm]", ylabel="Fracción acumulada de acciones")
    axes[0, 1].scatter(theta[indices], c[indices], s=5, alpha=0.2)
    axes[0, 1].set(xlabel="‖φ‖ [°]", ylabel="‖V(φ)ρ − ρ‖ [mm]")
    valid = indices[np.isfinite(angles[indices])]
    axes[1, 0].scatter(angles[valid], c[valid], s=5, alpha=0.2)
    axes[1, 0].set(xlabel="Ángulo entre ρ y el eje de φ [°; 0–90]",
                   ylabel="‖V(φ)ρ − ρ‖ [mm]")
    axes[1, 1].scatter(translation[indices], theta[indices], s=5, alpha=0.2)
    axes[1, 1].set(xlabel="‖ρ‖ [mm]", ylabel="‖φ‖ [°]")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle(label + " · Diferencia geométrica SE(3) frente a T+SO(3)\n"
                 "Acciones agrupadas; no mide la desviación ejecutada por el robot")
    save_figure(fig, folder, "05_acoplamiento_acciones")


def make_plots(episode_results, plot_data, exp_path, dt, label):
    """Se llama UNA vez, cuando ya están cargados todos los episodios."""
    folder = os.path.join(exp_path, "graficas")
    os.makedirs(folder, exist_ok=True)
    plot_errors(plot_data, dt, folder, label)
    success_rate = 100 * np.mean([p["success"] for p in episode_results])
    metric_panels(episode_results, [
        ("end_ep", 1000, "Error final de posición [mm]"),
        ("end_er", 180 / np.pi, "Error final de orientación [°]"),
        ("distance_travelled", 1, "Longitud recorrida [m]"),
        ("rotation_travelled", 180 / np.pi, "Rotación acumulada [°]"),
        ("straightness_p", 1, "Rectitud de traslación"),
        ("straightness_r", 1, "Rectitud angular"),
    ], folder, "02_distribuciones_metricas",
        f"{label} · {len(episode_results)} episodios · Éxito: {success_rate:.1f}%")
    metric_panels(episode_results, [
        ("linear_acceleration_rms_m_s2", 1, "Aceleración lineal RMS [m/s²]"),
        ("linear_jerk_squared_integral", 1, "Integral de jerk lineal² [m²/s⁵]"),
        ("linear_dimensionless_jerk_path", 1, "Jerk lineal adimensional"),
        ("angular_acceleration_rms_rad_s2", 1, "Aceleración angular RMS [rad/s²]"),
        ("angular_jerk_squared_integral", 1, "Integral de jerk angular² [rad²/s⁵]"),
        ("angular_dimensionless_jerk_path", 1, "Jerk angular adimensional"),
    ], folder, "03_suavidad_episodios", label + " · Suavidad por episodio hasta el corte")
    plot_trajectories(plot_data, folder, label)
    plot_coupling(plot_data, folder, label)
    return folder


def main():
    # Configuración e inicialización: se ejecutan una sola vez.
    n_experiment = "7"
    representation = "SE3"
    dt = 0.01
    action_order = "phi_rho"  # O "rho_phi"; acciones ya escaladas a radianes/metros.
    exp_path = (
        "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/"
        "manager_based/aurova_reinforcement_learning/rl_manipulation/exp"
        + n_experiment + "/" + representation
    )

    experiments = sorted(
        filename for filename in os.listdir(exp_path)
        if filename.endswith(".pickle") and filename != "0_results.pickle"
    )
    episode_results = []
    plot_data = []

    # Dentro del bucle: cargar y evaluar el episodio actual.
    for sample_name in experiments:
        with open(os.path.join(exp_path, sample_name), "rb") as data:
            sample = pickle.load(data)

        results = evaluate_episode(sample, n_experiment, plot_data, dt, action_order)
        episode_results.append(results)

    # Fuera del bucle: promediar, mostrar y guardar una sola vez.
    my_results = aggregate_results(episode_results)
    for key, value in my_results.items():
        print(key, ": ", value)

    save_results(my_results, exp_path)
    folder = make_plots(episode_results, plot_data, exp_path, dt,
                        f"Experimento {n_experiment} · {representation}")
    print("Gráficas guardadas en:", folder)
    return my_results


if __name__ == "__main__":
    main()