"""Harmonic FEM model of a rectangular room driven by a monopole noise
source, with a single listener point and a fixed-position longitudinal
quadrupole control source """

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pv.OFF_SCREEN = True

from room_geometry import RoomModel, load_config, rigid_room_modes, to_db, P_REF

# ==== OUTPUT FOLDERS =========================================================

_job_id = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = f"plots_{_job_id}"
DIR_3D_PRESSURE = os.path.join(OUTPUT_DIR, "3d_pressure")
DIR_3D_PLANE = os.path.join(OUTPUT_DIR, "3d_plane")
DIR_3D_DATA = os.path.join(OUTPUT_DIR, "3d_data")
DIR_MODEL = os.path.join(OUTPUT_DIR, "model")
DIR_UNIT_FIELDS = os.path.join(OUTPUT_DIR, "unit_fields")
for d in (DIR_3D_PRESSURE, DIR_3D_PLANE, DIR_3D_DATA, DIR_MODEL, DIR_UNIT_FIELDS):
    os.makedirs(d, exist_ok=True)

JOBNAME = "room_acoustics_control"

_T0 = time.perf_counter()
_lap_t = _T0


def lap(label):
    global _lap_t
    now = time.perf_counter()
    print(f"[timer] {label}: {now - _lap_t:.2f} s (total {now - _T0:.2f} s)")
    _lap_t = now


# ==== BUILD MODEL ============================================================

cfg = load_config("room_config.json")
model = RoomModel(cfg, OUTPUT_DIR, jobname=JOBNAME)
lap("build geometry + mesh + BCs")

mesh_grid = model.mapdl.mesh.grid.copy()
mesh_grid.save(os.path.join(DIR_3D_DATA, "mesh.vtu"))

model.mapdl.save(fname=JOBNAME, ext="db")
model.mapdl.download(f"{JOBNAME}.db", target_dir=DIR_MODEL)

# ==== SOLVE 1: PRIMARY SOURCE ONLY ==========================================

model.clear_all_bf()
model.apply_primary_bf()
solved_freqs = model.harmonic_solve(light_outres=False)
freqs = solved_freqs
model.mapdl.download(f"{JOBNAME}.rst", target_dir=DIR_MODEL)
lap("harmonic solve (primary)")

listener_pressure_primary = model.node_pressure_sweep(model.listener_node, len(freqs))
listener_spl = to_db(listener_pressure_primary)
listener_velocity = model.node_velocity_sweep(model.listener_node, len(freqs))
listener_vmag_rms = np.abs(listener_velocity[:, 3]) / np.sqrt(2)
lap("listener sweep (primary)")

jp_grid_xy, p_grid_primary = model.plane_grid_pressure_sweep(
    model.JP_PLANE_Z, model.jp_grid_xy, len(freqs), "primary")
Jp_n = model.jp_from_grid(p_grid_primary)
Jp_n_db = model.jp_db(Jp_n)
print(f"[jp] evaluated on a {model.JP_N_X}x{model.JP_N_Y}={jp_grid_xy.shape[0]}-point "
      f"grid at z={model.JP_PLANE_Z} m -- Jp,n ranges "
      f"{Jp_n_db.min():.1f} to {Jp_n_db.max():.1f} dB")
lap("jp grid sweep (primary)")

# ---- theoretical modes + band-stratified frequency picks for field maps ---

modes = rigid_room_modes(model.LX, model.LY, model.LZ, model.C0,
                          model.FREQ_MIN, model.FREQ_MAX)
N_BANDS = 4
PLANES_PER_BAND = 2
band_edges = np.linspace(model.FREQ_MIN, model.FREQ_MAX, N_BANDS + 1)
plot_freqs = []
for lo, hi in zip(band_edges[:-1], band_edges[1:]):
    band_modes = [m for m in modes if lo <= m < hi]
    if not band_modes:
        continue
    band_idx = [int(np.argmin(np.abs(freqs - m))) for m in band_modes]
    dedup = {}
    for m, idx in zip(band_modes, band_idx):
        dedup.setdefault(idx, m) 
    dedup_modes = list(dedup.values())
    band_spl = [listener_spl[idx] for idx in dedup.keys()]
    order = np.argsort(band_spl)[::-1][:PLANES_PER_BAND]
    plot_freqs.extend(dedup_modes[i] for i in order)
plot_freqs = sorted(plot_freqs)
print(f"[sweep] modes selected for field plots (Hz): "
      f"[{', '.join(f'{f:.1f}' for f in plot_freqs)}]")

# ---- listener plots ---------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, listener_spl, color="tab:orange")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("SPL, unweighted (dB)")
ax.set_title("Listener SPL response, primary source only")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "listener_sweep.png"), dpi=150)
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(freqs, listener_spl, color="tab:orange", label="SPL")
ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("SPL, unweighted (dB)", color="tab:orange")
ax1.tick_params(axis="y", labelcolor="tab:orange"); ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(freqs, listener_vmag_rms, color="tab:blue", label="Velocity")
ax2.set_ylabel("Particle velocity, RMS (m/s)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_title("Listener SPL and particle velocity, primary source only")
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "listener_spl_velocity_combined.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, Jp_n_db, color="tab:green")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Jp,n (dB)")
ax.set_title(f"Acoustic potential energy proxy Jp,n "
             f"({model.JP_N_X}x{model.JP_N_Y}-point grid, z={model.JP_PLANE_Z} m)")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTPUT_DIR, "jp_n_sweep.png"), dpi=150)
plt.close(fig)
lap("primary-field plots")

# ---- resonance field maps (primary field only) + cache for later reuse ---

def plot_3d(f, field, tag, vmin, vmax):
    grid = model.mapdl.mesh.grid.copy()
    grid.point_data["SPL (dB)"] = np.clip(to_db(field), vmin, vmax)
    grid.save(os.path.join(DIR_3D_DATA, f"spl_3d_{f:.0f}Hz_{tag}.vtu"))
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(grid, scalars="SPL (dB)", cmap="jet", show_edges=False,
                clim=[vmin, vmax], scalar_bar_args={"fmt": "%.0f"})
    pl.add_text(f"SPL field at {f:.1f} Hz ({tag})", font_size=12)
    pl.camera_position = "iso"
    pl.show(screenshot=os.path.join(DIR_3D_PRESSURE, f"spl_3d_{f:.0f}Hz_{tag}.png"))


def plot_plane(f, plane_z, field_re, field_im, pts_x, pts_y, tag, vmin, vmax):
    spl = np.clip(to_db(field_re + 1j * field_im), vmin, vmax)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    levels = np.linspace(vmin, vmax, 101)
    tpc = ax.tricontourf(pts_x, pts_y, spl, levels=levels, cmap="jet")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"SPL field at z={plane_z:.3f} m, f={f:.1f} Hz ({tag})")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="SPL (dB)", format="%.0f")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_3D_PLANE, f"spl_plane_{f:.0f}Hz_z{plane_z:.3f}m_{tag}.png"), dpi=150)
    plt.close(fig)


model.mapdl.allsel()
peak_cache = []              
global_min, global_max = np.inf, -np.inf
for target_freq in plot_freqs:
    f, field = model.nodal_pressure_at(solved_freqs, target_freq)
    peak_cache.append((f, field))
    db = to_db(field)
    n_bad = int((~np.isfinite(db)).sum())
    if n_bad:
        print(f"[warn] primary field at {f:.0f} Hz has {n_bad} non-finite SPL value(s) "
              "-- excluded from the color-range calculation (nanmin/nanmax)")
    global_min = min(global_min, np.nanmin(db))
    global_max = max(global_max, np.nanmax(db))
global_min, global_max = float(np.floor(global_min)), float(np.ceil(global_max))
lap("field extraction (primary)")

plane_node_xy = {}
plane_field_primary = {}   
for plane_z in model.PLANE_HEIGHTS_Z:
    model.mapdl.allsel()
    model.mapdl.nsel("S", "LOC", "Z", plane_z)
    n_pts = int(model.mapdl.get_value(entity="node", entnum=0, item1="count"))
    if n_pts == 0:
        raise RuntimeError(f"no nodes found at z={plane_z} m for plotting.")
    coords = model.mapdl.mesh.nodes
    plane_node_xy[plane_z] = coords[:, :2].copy()
    rows = []
    for f, _ in peak_cache:
        model.mapdl.set(time=f, kimg=0)
        p_re = model.mapdl.post_processing.nodal_pressure()
        model.mapdl.set(time=f, kimg=1)
        p_im = model.mapdl.post_processing.nodal_pressure()
        rows.append(p_re + 1j * p_im)
        plot_plane(f, plane_z, p_re, p_im, coords[:, 0], coords[:, 1],
                   "primary", global_min, global_max)
    plane_field_primary[plane_z] = np.array(rows)
    model.mapdl.allsel()

for f, field in peak_cache:
    plot_3d(f, field, "primary", global_min, global_max)
lap("peak field plots (primary)")

# ==== SOLVE 2 & 3: CONTROL DIPOLES AT UNIT LEVEL ============================

dipole_listener = []       # per dipole: complex pressure at listener, all freqs
dipole_p_grid = []         # per dipole: complex pressure on Jp grid, all freqs
dipole_field_full = []      # per dipole: (n_plot_freqs, n_mesh_nodes) complex, full mesh
dipole_plane_field = {plane_z: [] for plane_z in model.PLANE_HEIGHTS_Z}   # per dipole

for d in range(2):
    label = f"dipole {d}"
    model.clear_all_bf()
    model.apply_dipole_bf(d)   # unit reference level
    dipole_freqs = model.harmonic_solve(light_outres=True)
    if dipole_freqs.shape != freqs.shape or not np.allclose(dipole_freqs, freqs):
        raise RuntimeError(
            f"[{label}] solved frequencies differ from the primary solve's -- "
            f"got {dipole_freqs.shape[0]} points, expected {freqs.shape[0]}.")
    lap(f"harmonic solve ({label})")

    p_listener = model.node_pressure_sweep(model.listener_node, len(freqs))
    _, p_grid = model.plane_grid_pressure_sweep(
        model.JP_PLANE_Z, model.jp_grid_xy, len(freqs), label)
    dipole_listener.append(p_listener)
    dipole_p_grid.append(p_grid)

    # Reconstruct SPL via superposition with zero further MAPDL calls.
    full_rows = []
    for target_freq, _ in peak_cache:
        _, field = model.nodal_pressure_at(dipole_freqs, target_freq)
        full_rows.append(field)
    dipole_field_full.append(np.array(full_rows))

    for plane_z in model.PLANE_HEIGHTS_Z:
        model.mapdl.allsel()
        model.mapdl.nsel("S", "LOC", "Z", plane_z)
        coords = model.mapdl.mesh.nodes
        rows = []
        for f, _ in peak_cache:
            model.mapdl.set(time=f, kimg=0)
            p_re = model.mapdl.post_processing.nodal_pressure()
            model.mapdl.set(time=f, kimg=1)
            p_im = model.mapdl.post_processing.nodal_pressure()
            rows.append(p_re + 1j * p_im)
        dipole_plane_field[plane_z].append(np.array(rows))
        model.mapdl.allsel()

    print(f"[solve] {label}: cached listener + {jp_grid_xy.shape[0]}-point Jp grid "
          f"+ full-mesh unit response across {len(freqs)} frequencies")
    lap(f"unit-field extraction ({label})")

model.exit()
lap("mapdl exit")

# ==== SAVE UNIT FIELDS =======================================================

unit_fields_path = os.path.join(DIR_UNIT_FIELDS, "unit_fields_control.npz")
save_kwargs = dict(
    freqs=freqs,
    listener_pressure_primary=listener_pressure_primary,
    p_grid_primary=p_grid_primary,
    Jp_n=Jp_n,
    listener_pressure_dipole=np.stack(dipole_listener),
    p_grid_dipole=np.stack(dipole_p_grid),
    jp_grid_xy=jp_grid_xy,
    jp_plane_z=model.JP_PLANE_Z,
    room_volume_m3=model.ROOM_VOLUME_M3,
    rho0=model.RHO_AIR,
    c0=model.C0,
    mass_ref=model.MASS_REF,
    monopole_signs=np.array([-1.0, 1.0, 1.0, -1.0]),
    dipole_pairs=np.array([(0, 1), (2, 3)]),
    ctrl_monopole_xy=model.CTRL_MONOPOLE_XY,
    ctrl_z=model.CTRL_Z,
    length_x_m=model.LX, length_y_m=model.LY, height_z_m=model.LZ,
    src_x_m=model.SRC_X, src_y_m=model.SRC_Y, src_z_m=model.SRC_Z,
    listener_x_m=model.LISTENER_X, listener_y_m=model.LISTENER_Y, listener_z_m=model.LISTENER_Z,
    plot_freqs=np.array([f for f, _ in peak_cache]),
    field_primary=np.array([field for _, field in peak_cache]),
    field_dipole=np.stack(dipole_field_full),
)

for plane_z in model.PLANE_HEIGHTS_Z:
    tag = f"{plane_z:.3f}".replace(".", "p")
    save_kwargs[f"plane_xy_z{tag}"] = plane_node_xy[plane_z]
    save_kwargs[f"plane_field_primary_z{tag}"] = plane_field_primary[plane_z]
    save_kwargs[f"plane_field_dipole_z{tag}"] = np.stack(dipole_plane_field[plane_z])
save_kwargs["plane_heights_z_m"] = np.array(model.PLANE_HEIGHTS_Z)
save_kwargs["global_spl_min_db"] = global_min
save_kwargs["global_spl_max_db"] = global_max

np.savez_compressed(unit_fields_path, **save_kwargs)
print(f"[done] cached unit fields -> {unit_fields_path}")
print("Run ga_optimize.py against this file -- it needs no further MAPDL calls.")
lap("save unit fields")

print(f"[timer] TOTAL: {time.perf_counter() - _T0:.2f} s")