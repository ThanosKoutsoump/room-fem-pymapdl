import glob
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from jp_regions import find_regions, region_for_index

JP_REF = 1e-12
P_REF = 20e-6


def to_db(pa, p_ref=P_REF):
    p_rms = np.abs(pa) / np.sqrt(2)
    return 20 * np.log10(np.clip(p_rms, 1e-12, None) / p_ref)


def find_unit_fields_file(explicit_path=None):
    if explicit_path:
        return explicit_path
    candidates = sorted(glob.glob("plots_*/unit_fields/unit_fields_control.npz"))
    if not candidates:
        raise SystemExit(
            "Could not find unit_fields_control.npz under plots_*/unit_fields/ "
            "-- run room_acoustics_control.py first, or pass the path explicitly: "
            "python ga_optimize.py path/to/unit_fields_control.npz")
    return candidates[-1]

RUN_POLARITY_SEARCH = False   # set False to only do the fixed-polarity-0 pass
RUN_PER_FREQUENCY_CEILING = False   # set False to skip the per-frequency optimum reference curve


# ==== superposition reconstruction ==========================================

def reconstruct_jp(p_grid_primary_row, p_grid_dipole_rows, room_volume, rho0, c0,
                    l1_db, l2_db, polarity):
    a1 = 10 ** (l1_db / 20.0)
    s2 = 1.0 if polarity == 0 else -1.0
    a2 = s2 * 10 ** (l2_db / 20.0)
    p_total = (p_grid_primary_row + a1 * p_grid_dipole_rows[0]
               + a2 * p_grid_dipole_rows[1])
    n_pts = p_total.shape[-1]
    return (room_volume / (4.0 * rho0 * c0 ** 2 * n_pts)) * np.sum(np.abs(p_total) ** 2)


def jp_to_db(jp):
    return 10 * np.log10(max(float(jp), 1e-30) / JP_REF)


def brute_force_best(p_grid_primary_row, p_grid_dipole_rows, room_volume, rho0, c0,
                      l_min, l_max, max_diff, polarity, n_grid=300):
   
    l1_vals = np.linspace(l_min, l_max, n_grid)
    l2_vals = np.linspace(l_min, l_max, n_grid)
    s2 = 1.0 if polarity == 0 else -1.0
    n_pts = p_grid_primary_row.shape[-1]
    best_jp = np.inf
    best_l1, best_l2 = l1_vals[0], l2_vals[0]
    for l1 in l1_vals:
        mask = np.abs(l1 - l2_vals) <= max_diff
        l2_valid = l2_vals[mask]
        if l2_valid.size == 0:
            continue
        a1 = 10 ** (l1 / 20.0)
        a2 = s2 * 10 ** (l2_valid / 20.0)
        p_total = (p_grid_primary_row[None, :] + a1 * p_grid_dipole_rows[0][None, :]
                   + a2[:, None] * p_grid_dipole_rows[1][None, :])
        jp_vals = (room_volume / (4.0 * rho0 * c0 ** 2 * n_pts)) * np.sum(np.abs(p_total) ** 2, axis=-1)
        idx = int(np.argmin(jp_vals))
        if jp_vals[idx] < best_jp:
            best_jp = float(jp_vals[idx])
            best_l1, best_l2 = l1, float(l2_valid[idx])
    return best_l1, best_l2, best_jp


# ==== genetic algorithm ======================================================

def genetic_algorithm(fitness, l_min, l_max, max_diff, pop_size=60, generations=150,
                       rng=None, elite_frac=0.1, tournament_k=3,
                       mutation_sigma0=3.0, mutation_sigma_decay=0.97):
    
    rng = rng or np.random.default_rng()

    def constrain(ind):
        ind = np.clip(ind, l_min, l_max)
        diff = ind[0] - ind[1]
        if abs(diff) > max_diff:
            excess = (abs(diff) - max_diff) / 2.0
            if diff > 0:
                ind[0] -= excess; ind[1] += excess
            else:
                ind[0] += excess; ind[1] -= excess
            ind = np.clip(ind, l_min, l_max)
        return ind

    pop = np.array([constrain(rng.uniform(l_min, l_max, size=2)) for _ in range(pop_size)])
    fit = np.array([fitness(ind) for ind in pop])
    n_elite = max(1, int(pop_size * elite_frac))
    sigma = mutation_sigma0
    history = []

    for _ in range(generations):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        history.append(fit[0])

        new_pop = [pop[i].copy() for i in range(n_elite)]
        while len(new_pop) < pop_size:
            i1 = rng.integers(0, pop_size, size=tournament_k)
            i2 = rng.integers(0, pop_size, size=tournament_k)
            p1 = pop[i1[np.argmin(fit[i1])]]
            p2 = pop[i2[np.argmin(fit[i2])]]
            alpha = rng.uniform(0.0, 1.0, size=2)
            child = alpha * p1 + (1.0 - alpha) * p2
            child = child + rng.normal(0.0, sigma, size=2)
            new_pop.append(constrain(child))
        pop = np.array(new_pop)
        fit = np.array([fitness(ind) for ind in pop])
        sigma *= mutation_sigma_decay

    order = np.argsort(fit)
    return pop[order[0]], float(fit[order[0]]), history


# ==== per-region optimization ================================================

def optimize_regions(data, ga_cfg, polarities):
    freqs = data["freqs"]
    jp_n = data["Jp_n"]
    jp_n_db = np.array([jp_to_db(v) for v in jp_n])
    regions = find_regions(freqs, jp_n_db,
                            drop_db=ga_cfg["region_drop_db"],
                            min_peak_prominence_db=ga_cfg["region_min_peak_prominence_db"])

    room_volume = float(data["room_volume_m3"])
    rho0 = float(data["rho0"])
    c0 = float(data["c0"])

    results = []
    for region in regions:
        idx = region.idx_rep
        p_grid_primary_row = data["p_grid_primary"][idx]
        p_grid_dipole_rows = data["p_grid_dipole"][:, idx, :]

        best = None
        best_history = None
        for pol in polarities:
            def fitness(ind, pol=pol):
                return reconstruct_jp(p_grid_primary_row, p_grid_dipole_rows,
                                       room_volume, rho0, c0, ind[0], ind[1], pol)
            rng = np.random.default_rng(ga_cfg.get("seed", 0) + region.idx_rep + pol)
            ind, jp_val, history = genetic_algorithm(
                fitness, ga_cfg["l_min_db"], ga_cfg["l_max_db"],
                ga_cfg["max_l1_l2_diff_db"], pop_size=ga_cfg["population_size"],
                generations=ga_cfg["generations"], rng=rng)
            if best is None or jp_val < best[2]:
                best = (ind, pol, jp_val)
                best_history = history

        ind, pol, jp_val = best

        jp_val_db = jp_to_db(jp_val)
        results.append(dict(
            region=region, l1_db=float(ind[0]), l2_db=float(ind[1]), polarity=pol,
            jp_n_db=float(jp_n_db[idx]), jp_c_db=jp_val_db,
            reduction_db=float(jp_n_db[idx] - jp_val_db),
            ga_history=best_history))
    return regions, jp_n_db, results


def reconstruct_full_curves(data, regions, results):
    freqs = data["freqs"]
    n = len(freqs)
    jp_c_db = np.zeros(n)
    listener_c = np.zeros(n, dtype=complex)
    room_volume = float(data["room_volume_m3"])
    rho0 = float(data["rho0"]); c0 = float(data["c0"])
    by_region = {id(r["region"]): r for r in results}

    for i in range(n):
        region = region_for_index(regions, i)
        res = by_region[id(region)]
        a1 = 10 ** (res["l1_db"] / 20.0)
        s2 = 1.0 if res["polarity"] == 0 else -1.0
        a2 = s2 * 10 ** (res["l2_db"] / 20.0)

        p_grid_i = (data["p_grid_primary"][i] + a1 * data["p_grid_dipole"][0, i]
                    + a2 * data["p_grid_dipole"][1, i])
        n_pts = p_grid_i.shape[-1]
        jp_i = (room_volume / (4.0 * rho0 * c0 ** 2 * n_pts)) * np.sum(np.abs(p_grid_i) ** 2)
        jp_c_db[i] = jp_to_db(jp_i)

        listener_c[i] = (data["listener_pressure_primary"][i]
                          + a1 * data["listener_pressure_dipole"][0, i]
                          + a2 * data["listener_pressure_dipole"][1, i])
    return jp_c_db, listener_c


def optimize_per_frequency(data, l_min, l_max, max_diff, polarity=0, n_grid=150):
    
    freqs = data["freqs"]
    room_volume = float(data["room_volume_m3"])
    rho0 = float(data["rho0"])
    c0 = float(data["c0"])
    jp_ceiling_db = np.zeros(len(freqs))
    for i in range(len(freqs)):
        p_grid_primary_row = data["p_grid_primary"][i]
        p_grid_dipole_rows = data["p_grid_dipole"][:, i, :]
        _, _, jp = brute_force_best(p_grid_primary_row, p_grid_dipole_rows,
                                     room_volume, rho0, c0, l_min, l_max, max_diff,
                                     polarity, n_grid=n_grid)
        jp_ceiling_db[i] = jp_to_db(jp)
    return jp_ceiling_db


def plot_convergence(results, out_path="ga_convergence.png", n_regions=4):
    
    peaks = [r for r in results if r["region"].kind == "peak"]
    picks = sorted(peaks, key=lambda r: r["jp_n_db"], reverse=True)[:n_regions] or results[:n_regions]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for res in picks:
        hist_db = [jp_to_db(v) for v in res["ga_history"]]
        r = res["region"]
        ax.plot(hist_db, label=f"{r.f_lo:.0f}-{r.f_hi:.0f} Hz ({r.kind})")
    ax.set_xlabel("GA generation")
    ax.set_ylabel("Best Jp,c found so far (dB)")
    ax.set_title("GA convergence -- should flatten out well before the last generation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def freq_to_region(regions, freqs, target_freq):
    
    idx = int(np.argmin(np.abs(freqs - target_freq)))
    return region_for_index(regions, idx)


def plot_controlled_plane_maps(data, regions, results, out_prefix="anc_plane"):
    
    freqs = data["freqs"]
    plot_freqs = data["plot_freqs"]
    plane_heights = data["plane_heights_z_m"]
    by_region = {id(r["region"]): r for r in results}
    vmin = float(data["global_spl_min_db"])
    vmax = float(data["global_spl_max_db"])
    levels = np.linspace(vmin, vmax, 101)

    saved = []
    for plane_z in plane_heights:
        tag = f"{plane_z:.3f}".replace(".", "p")
        xy = data[f"plane_xy_z{tag}"]
        field_primary = data[f"plane_field_primary_z{tag}"]   # (n_plot_freqs, n_nodes) complex
        field_dipole = data[f"plane_field_dipole_z{tag}"]       # (2, n_plot_freqs, n_nodes) complex
        pts_x, pts_y = xy[:, 0], xy[:, 1]

        for i, f in enumerate(plot_freqs):
            region = freq_to_region(regions, freqs, f)
            res = by_region[id(region)]
            a1 = 10 ** (res["l1_db"] / 20.0)
            s2 = 1.0 if res["polarity"] == 0 else -1.0
            a2 = s2 * 10 ** (res["l2_db"] / 20.0)

            p_before = field_primary[i]
            p_after = p_before + a1 * field_dipole[0, i] + a2 * field_dipole[1, i]
            spl_before = np.clip(to_db(p_before), vmin, vmax)
            spl_after = np.clip(to_db(p_after), vmin, vmax)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
            for ax, spl, title in ((axes[0], spl_before, "before ANC"),
                                    (axes[1], spl_after, "after ANC")):
                tpc = ax.tricontourf(pts_x, pts_y, spl, levels=levels, cmap="jet")
                ax.set_xlabel("x (m)")
                ax.set_aspect("equal")
                ax.set_title(title)
            axes[0].set_ylabel("y (m)")
            fig.colorbar(tpc, ax=axes, label="SPL (dB)", format="%.0f", shrink=0.85)
            phi_str = "0" if res["polarity"] == 0 else "pi"
            fig.suptitle(f"f={f:.2f} Hz, z={plane_z:.3f} m -- "
                         f"L1={res['l1_db']:.2f} dB, L2={res['l2_db']:.2f} dB, phi={phi_str} "
                         f"(region {region.f_lo:.2f}-{region.f_hi:.2f} Hz)")
            fname = f"{out_prefix}_{f:.0f}Hz_z{tag}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(fname)
            print(f"[plot] saved {fname}")
    return saved


def print_table(regions, results, label):
    print(f"\n=== {label} ===")
    print(f"{'region (Hz)':>16} {'f_rep':>8} {'kind':>7} {'L1(dB)':>8} {'L2(dB)':>8} "
          f"{'phi':>5} {'Jp,n(dB)':>9} {'Jp,c(dB)':>9} {'reduct.(dB)':>11}")
    for res in results:
        r = res["region"]
        phi_str = "0" if res["polarity"] == 0 else "pi"
        print(f"{r.f_lo:7.2f}-{r.f_hi:6.2f} {r.f_rep:8.2f} {r.kind:>7} "
              f"{res['l1_db']:8.2f} {res['l2_db']:8.2f} {phi_str:>5} "
              f"{res['jp_n_db']:9.2f} {res['jp_c_db']:9.2f} {res['reduction_db']:11.2f}")


def save_results_csv(path, results):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f_lo_hz", "f_hi_hz", "f_rep_hz", "kind", "L1_dB", "L2_dB",
                    "polarity", "Jp_n_dB", "Jp_c_dB", "reduction_dB"])
        for res in results:
            r = res["region"]
            w.writerow([r.f_lo, r.f_hi, r.f_rep, r.kind, res["l1_db"], res["l2_db"],
                        "0" if res["polarity"] == 0 else "pi",
                        res["jp_n_db"], res["jp_c_db"], res["reduction_db"]])


# ==== main ====================================================================

if __name__ == "__main__":
    with open("room_config.json") as f:
        cfg = json.load(f)
    ga_cfg = cfg.get("ga", {"l_min_db": -10.0, "l_max_db": 40.0, "max_l1_l2_diff_db": 10.0,
                              "population_size": 60, "generations": 150, "seed": 0,
                              "region_drop_db": 6.0, "region_min_peak_prominence_db": 1.0})

    npz_path = find_unit_fields_file(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[load] {npz_path}")
    data = np.load(npz_path)
    freqs = data["freqs"]

    # ---- Pass 1: polarity fixed at 0
    regions, jp_n_db, results_fixed = optimize_regions(data, ga_cfg, polarities=[0])
    print_table(regions, results_fixed, "Pass 1: polarity fixed at 0")
    jp_c_db_fixed, listener_c_fixed = reconstruct_full_curves(data, regions, results_fixed)
    save_results_csv("ga_results_polarity_fixed.csv", results_fixed)
    plot_controlled_plane_maps(data, regions, results_fixed, out_prefix="anc_plane_phi0")
    plot_convergence(results_fixed, out_path="ga_convergence_phi0.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freqs, jp_n_db, color="black", label="Jp,n (uncontrolled)")
    ax.plot(freqs, jp_c_db_fixed, "--", color="tab:red", label="Jp,c (GA, region-held, phi=0)")
    if RUN_PER_FREQUENCY_CEILING:
        print("\n[ceiling] optimizing independently at every solved frequency "
              f"({len(freqs)} points) -- no region-holding, for reference only...")
        jp_ceiling_db = optimize_per_frequency(
            data, ga_cfg["l_min_db"], ga_cfg["l_max_db"], ga_cfg["max_l1_l2_diff_db"], polarity=0)
        ax.plot(freqs, jp_ceiling_db, ":", color="tab:gray",
                 label="Jp,c ceiling (per-frequency optimal, phi=0)")
        avg_gap = float(np.mean(jp_c_db_fixed - jp_ceiling_db))
        max_gap_region = float(np.max(jp_c_db_fixed - jp_ceiling_db))
        print(f"[ceiling] region-based Jp,c is on average {avg_gap:.2f} dB above the "
              f"per-frequency ceiling (worst point: {max_gap_region:.2f} dB) -- a large "
              "gap here means narrower regions (region_drop_db / "
              "region_min_peak_prominence_db in the \"ga\" config block) would help; "
              "a small gap means you're already close to what this source can do.")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Jp (dB)")
    ax.set_title("Longitudinal quadrupole control -- polarity fixed at 0")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("jp_comparison_polarity_fixed.png", dpi=150)
    plt.close(fig)

    # ---- Pass 2 : allow the GA to also pick polarity in {0, pi}
    if RUN_POLARITY_SEARCH:
        regions2, _, results_search = optimize_regions(data, ga_cfg, polarities=[0, 1])
        print_table(regions2, results_search, "Pass 2: polarity searched in {0, pi}")
        jp_c_db_search, listener_c_search = reconstruct_full_curves(data, regions2, results_search)
        save_results_csv("ga_results_polarity_search.csv", results_search)
        plot_controlled_plane_maps(data, regions2, results_search, out_prefix="anc_plane_phi_search")
        plot_convergence(results_search, out_path="ga_convergence_phi_search.png")

        print("\n=== polarity search vs. fixed-phi=0, per region ===")
        for r1, r2 in zip(results_fixed, results_search):
            gain = r1["jp_c_db"] - r2["jp_c_db"]
            flip = "" if r2["polarity"] == 0 else "  <-- polarity flipped to pi"
            print(f"  {r2['region'].f_lo:6.2f}-{r2['region'].f_hi:5.2f} Hz: "
                  f"{gain:+.2f} dB extra reduction{flip}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(freqs, jp_n_db, color="black", label="Jp,n (uncontrolled)")
        ax.plot(freqs, jp_c_db_fixed, "--", color="tab:red", label="Jp,c (GA, phi=0)")
        ax.plot(freqs, jp_c_db_search, ":", color="tab:blue", label="Jp,c (GA, phi searched)")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Jp (dB)")
        ax.set_title("Longitudinal quadrupole control -- fixed vs. searched polarity")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig("jp_comparison_polarity_search.png", dpi=150)
        plt.close(fig)

    # ---- listener SPL before/after, sanity check ----
    listener_spl_n = to_db(data["listener_pressure_primary"])
    listener_spl_c = to_db(listener_c_fixed)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(freqs, listener_spl_n, color="black", label="Listener SPL, uncontrolled")
    ax.plot(freqs, listener_spl_c, "--", color="tab:red", label="Listener SPL, controlled (phi=0)")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("SPL (dB)")
    ax.set_title("Listener SPL before/after control")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("listener_comparison.png", dpi=150)
    plt.close(fig)

    print("\nSaved: ga_results_polarity_fixed.csv, jp_comparison_polarity_fixed.png, "
          "listener_comparison.png, anc_plane_phi0_*.png (before/after SPL maps)"
          + (", ga_results_polarity_search.csv, jp_comparison_polarity_search.png, "
             "anc_plane_phi_search_*.png" if RUN_POLARITY_SEARCH else ""))