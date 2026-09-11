"""Harmonic FEM model of a rectangular room driven by a monopole speaker."""

import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

pv.OFF_SCREEN = True

from ansys.mapdl.core import launch_mapdl

# ==== OUTPUT FOLDERS ========================================================

OUTPUT_DIR = "plots"
DIR_3D_PRESSURE = os.path.join(OUTPUT_DIR, "3d_pressure")
DIR_3D_PLANE = os.path.join(OUTPUT_DIR, "3d_plane")
DIR_3D_DATA = os.path.join(OUTPUT_DIR, "3d_data")

os.makedirs(DIR_3D_PRESSURE, exist_ok=True)
os.makedirs(DIR_3D_PLANE, exist_ok=True)
os.makedirs(DIR_3D_DATA, exist_ok=True)

# ==== TIMER ================================================================

_T0 = time.perf_counter()
_lap_t = _T0


def lap(label):
    global _lap_t
    now = time.perf_counter()
    print(f"[timer] {label}: {now - _lap_t:.2f} s (total {now - _T0:.2f} s)")
    _lap_t = now


# ==== CONFIG ================================================================

CONFIG_PATH = "room_config.json"

try:
    with open(CONFIG_PATH) as _f:
        _cfg = json.load(_f)
except FileNotFoundError:
    raise SystemExit(
        f"Config file '{CONFIG_PATH}' not found. Create one alongside this "
        f"script (see room_config.json for the expected format) or point "
        f"CONFIG_PATH at an existing file.")

LX = _cfg["room"]["length_x_m"]
LY = _cfg["room"]["length_y_m"]
LZ = _cfg["room"]["height_z_m"]

SRC_X = _cfg["source"]["x_m"]
SRC_Y = _cfg["source"]["y_m"]
SRC_Z = _cfg["source"]["z_m"]
SRC_MASS_MAGNITUDE = _cfg["source"]["mass_source_kg_s"]


LISTENER_XYZ = (_cfg["listener"]["x_m"], _cfg["listener"]["y_m"],
                _cfg["listener"]["z_m"])

C0 = _cfg["acoustics"]["speed_of_sound_m_s"]
RHO_AIR = _cfg["acoustics"]["air_density_kg_m3"]
ALPHA_WALL = _cfg["acoustics"]["wall_absorption_coefficient"]

FREQ_MIN = _cfg["analysis"]["freq_min_hz"]
FREQ_MAX = _cfg["analysis"]["freq_max_hz"]
N_SUBSTEPS = _cfg["analysis"]["num_substeps"]
ELEMS_PER_WAVELENGTH = _cfg["analysis"]["elements_per_wavelength"]

PLANE_HEIGHTS_Z = _cfg["plotting"]["plane_heights_z_m"]
NUM_PEAKS = _cfg["plotting"]["num_peaks"]


ESIZE = C0 / (FREQ_MAX * ELEMS_PER_WAVELENGTH)
_r = np.sqrt(1.0 - ALPHA_WALL)
Z_WALL = RHO_AIR * C0 * (1.0 + _r) / (1.0 - _r)   # wall impedance (Pa*s/m)
P_REF = 20e-6                       # 0 dB SPL reference pressure (Pa)


def rigid_room_modes(lx, ly, lz, c, fmin, fmax, nmax=8):
    """Closed-form rigid-wall eigenfrequencies -- the theoretical modes to
    check the real (absorptive) room's response against."""
    modes = []
    for nx in range(nmax + 1):
        for ny in range(nmax + 1):
            for nz in range(nmax + 1):
                if nx == ny == nz == 0:
                    continue
                f = (c / 2) * np.sqrt((nx / lx) ** 2 + (ny / ly) ** 2
                                      + (nz / lz) ** 2)
                if fmin <= f <= fmax:
                    modes.append(f)
    return sorted(modes)


# ==== ROOM GEOMETRY ==========================================================

_job_id = os.environ.get("SLURM_JOB_ID", "local")
_run_base = os.environ.get("MAPDL_RUN_BASE", os.getcwd())
_mapdl_log_dir = os.path.join(_run_base, "mapdl_run", f"job_{_job_id}")
os.makedirs(_mapdl_log_dir, exist_ok=True)

mapdl = launch_mapdl(
    timeout=120,
    run_location=_mapdl_log_dir,
)
lap("launch_mapdl")
mapdl.clear()
mapdl.prep7()
mapdl.units("SI")

mapdl.mp("DENS", 1, RHO_AIR)
mapdl.mp("SONC", 1, C0)
mapdl.et(1, "FLUID221", kop2=1)   # quadratic acoustic element, pressure DOF

# room box corners
mapdl.k(1, 0, 0, 0)
mapdl.k(2, LX, 0, 0)
mapdl.k(3, LX, LY, 0)
mapdl.k(4, 0, LY, 0)
mapdl.k(5, 0, 0, LZ)
mapdl.k(6, LX, 0, LZ)
mapdl.k(7, LX, LY, LZ)
mapdl.k(8, 0, LY, LZ)

front_wall = mapdl.a(1, 4, 8, 5)
back_wall = mapdl.a(2, 3, 7, 6)
floor = mapdl.a(1, 2, 3, 4)
ceiling = mapdl.a(5, 6, 7, 8)
left_wall = mapdl.a(1, 2, 6, 5)
right_wall = mapdl.a(4, 3, 7, 8)

wall_areas = [front_wall, back_wall, floor, ceiling, left_wall, right_wall]
room_vol = mapdl.va(front_wall, back_wall, floor, ceiling, left_wall, right_wall)

# For local mesh refinement
src_kp = mapdl.k(100, SRC_X, SRC_Y, SRC_Z)

lap("geometry")

# ==== MESH ===================================================================

mapdl.type(1)
mapdl.mat(1)
mapdl.mshape(1, "3D")
mapdl.mshkey(0)
mapdl.kesize(src_kp, 0.03)   # finer mesh right at the source point
mapdl.esize(ESIZE)
mapdl.vmesh(room_vol)
print(f"[mesh] {mapdl.mesh.n_elem} elements, {mapdl.mesh.n_node} nodes, "
      f"esize={ESIZE:.3f} m")

# Portable mesh geometry export for local interactivity in PyVista/ParaView.
mesh_grid = mapdl.mesh.grid.copy()
mesh_grid.save(os.path.join(DIR_3D_DATA, "mesh.vtu"))

lap("mesh")

# ==== BOUNDARY CONDITIONS ====================================================

src_node = mapdl.queries.node(SRC_X, SRC_Y, SRC_Z)
mapdl.bf(src_node, "MASS", SRC_MASS_MAGNITUDE, 0.0)

mapdl.asel("S", "AREA", vmin=wall_areas[0])
for a in wall_areas[1:]:
    mapdl.asel("A", "AREA", vmin=a)
mapdl.nsla("S", 1)
mapdl.sf("all", "IMPD", Z_WALL)
mapdl.allsel()

lap("boundary conditions")

# ==== HARMONIC SOLVE =========================================================

mapdl.run("/SOLU")
mapdl.antype(3)
mapdl.harfrq(freqb=FREQ_MIN, freqe=FREQ_MAX)
mapdl.autots("off")
mapdl.nsubst(N_SUBSTEPS)
mapdl.kbc(0)
mapdl.outres("erase")
mapdl.outres("all", "none")
mapdl.outres("nsol", "all")
mapdl.solve()
mapdl.finish()
lap("harmonic solve")

mapdl.post1()
solved_freqs = np.unique(mapdl.post_processing.time_values)

# ==== POST-PROCESSING HELPERS ================================================

def nodal_pressure(target_freq):

    f = solved_freqs[np.argmin(np.abs(solved_freqs - target_freq))]
    mapdl.set(time=f, kimg=0)
    real = mapdl.get_array(entity="NODE", item1="PRES")
    mapdl.set(time=f, kimg=1)
    imag = mapdl.get_array(entity="NODE", item1="PRES")
    return f, real + 1j * imag


def to_db(pa):
    return 20 * np.log10(np.clip(np.abs(pa), 1e-12, None) / P_REF)


def listener_sweep(xyz, n_sets):
    
    probe = pv.PolyData(np.array([xyz]))
    grid = mapdl.mesh.grid.copy()

    amps = np.zeros(n_sets)
    for i in range(1, n_sets + 1):
        mapdl.set(lstep=1, sbstep=i, kimg=3)
        grid.point_data["PRES_AMP"] = mapdl.get_array(entity="NODE", item1="PRES")
        sampled = probe.sample(grid)
        amps[i - 1] = sampled["PRES_AMP"][0]
    return amps


# ==== LISTENER SWEEP =========================================================

freqs = solved_freqs
listener_pressure = listener_sweep(LISTENER_XYZ, len(freqs))
listener_spl = to_db(listener_pressure)
lap("listener sweep")


# ==== THEORETICAL MODES ======================================================

modes = rigid_room_modes(LX, LY, LZ, C0, FREQ_MIN, FREQ_MAX)

mode_spl = [listener_spl[np.argmin(np.abs(freqs - m))] for m in modes]
_mode_order = np.argsort(mode_spl)[::-1][:NUM_PEAKS]
plot_freqs = sorted(modes[i] for i in _mode_order)

print(f"[sweep] modes selected for field plots (Hz): {[round(f, 1) for f in plot_freqs]}")

# ==== LISTENER SPL PLOT ======================================================

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, listener_spl, color="tab:orange")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("SPL, unweighted (dB)")
ax.set_title("Listener SPL response")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "listener_sweep.png"), dpi=150)
lap("listener sweep plot")

# ==== RESONANCE FIELD MAPS ===================================================

def plot_3d(f, field, tag, vmin, vmax):
    grid = mapdl.mesh.grid.copy()
    grid.point_data["SPL (dB)"] = to_db(field)

    grid.save(os.path.join(DIR_3D_DATA, f"spl_3d_{f:.0f}Hz_{tag}.vtu"))

    pl_save = pv.Plotter(off_screen=True)
    pl_save.add_mesh(grid, scalars="SPL (dB)", cmap="jet",
                     show_edges=False, clim=[vmin, vmax],
                     scalar_bar_args={"fmt": "%.0f"})
    pl_save.add_text(f"SPL field at {f:.1f} Hz ({tag})", font_size=12)
    pl_save.camera_position = "iso"
    pl_save.show(screenshot=os.path.join(
        DIR_3D_PRESSURE, f"spl_3d_{f:.0f}Hz_{tag}.png"))


def plot_plane(f, field, plane_z, tag, vmin, vmax):
    grid = mapdl.mesh.grid.copy()
    grid.point_data["SPL (dB)"] = to_db(field)

    plane_slice = grid.slice(normal="z", origin=(0, 0, plane_z + 1e-4))
    pts = plane_slice.points
    spl = np.clip(plane_slice.point_data["SPL (dB)"], vmin, vmax)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    levels = np.linspace(vmin, vmax, 101)
    tpc = ax.tricontourf(pts[:, 0], pts[:, 1], spl, levels=levels, cmap="jet")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"SPL field at z={plane_z:.2f} m, f={f:.1f} Hz ({tag})")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="SPL (dB)", format="%.0f")
    fig.tight_layout()
    fig.savefig(os.path.join(
        DIR_3D_PLANE, f"spl_plane_{f:.0f}Hz_z{plane_z:.2f}m_{tag}.png"), dpi=150)


mapdl.allsel()

peak_cache = []
global_min, global_max = np.inf, -np.inf
for target_freq in plot_freqs:
    f, field = nodal_pressure(target_freq)
    peak_cache.append((f, field))
    db = to_db(field)
    global_min, global_max = min(global_min, db.min()), max(global_max, db.max())
global_min, global_max = float(np.floor(global_min)), float(np.ceil(global_max))
lap("field extraction")

for f, field in peak_cache:
    plot_3d(f, field, "peak", global_min, global_max)
    for plane_z in PLANE_HEIGHTS_Z:
        plot_plane(f, field, plane_z, "peak", global_min, global_max)
lap("peak field plots")

# ==== DONE ===================================================================
mapdl.exit()
lap("mapdl exit")
print(f"[timer] TOTAL: {time.perf_counter() - _T0:.2f} s")
