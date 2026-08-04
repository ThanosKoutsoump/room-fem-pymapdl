"""Harmonic FEM model of a rectangular room driven by a monopole speaker.

Produces the listener SPL sweep and SPL field maps (3D + plane) at the
room's strongest resonances (peaks) and deepest nulls (dips).

Pipeline: build room geometry -> mesh -> apply source/wall boundary
conditions -> harmonic solve -> extract listener sweep -> find peaks
and dips -> plot.

Room, source, listener, and analysis settings are read from
room_config.json alongside this script.
"""

import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.signal import find_peaks
from ansys.mapdl.core import launch_mapdl

# ==== OUTPUT FOLDERS ========================================================
OUTPUT_DIR = "plots"
DIR_3D_PRESSURE = os.path.join(OUTPUT_DIR, "3d_pressure")
DIR_3D_PLANE = os.path.join(OUTPUT_DIR, "3d_plane")

os.makedirs(DIR_3D_PRESSURE, exist_ok=True)
os.makedirs(DIR_3D_PLANE, exist_ok=True)

# ==== TIMER ================================================================
_T0 = time.perf_counter()
_lap_t = _T0


def lap(label):
    """Print elapsed time since the previous lap() call."""
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

# source patch is fixed to the x=0 wall; y_m/z_m position it on that wall
SRC_Y = _cfg["source"]["y_m"]
SRC_Z = _cfg["source"]["z_m"]
SRC_R = _cfg["source"]["radius_m"]
SRC_VELOCITY = _cfg["source"]["velocity_m_s"]

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
NUM_DIPS = _cfg["plotting"]["num_dips"]
MODE_MATCH_TOL = _cfg["plotting"]["mode_match_tolerance_hz"]

# sanity checks on the loaded config -- catches typos before a 150s+ solve
assert LX > 0 and LY > 0 and LZ > 0, "room dimensions must be positive"
assert SRC_R > 0, "source radius_m must be positive"
assert SRC_R < min(SRC_Y, LY - SRC_Y, SRC_Z, LZ - SRC_Z), (
    "source patch extends past the wall edges -- reduce radius_m or "
    "move y_m/z_m")
assert (0 <= LISTENER_XYZ[0] <= LX and 0 <= LISTENER_XYZ[1] <= LY
        and 0 <= LISTENER_XYZ[2] <= LZ), "listener position must be inside the room"
assert 0 < FREQ_MIN < FREQ_MAX, "freq_min_hz must be positive and less than freq_max_hz"
assert 0 <= ALPHA_WALL < 1, "wall_absorption_coefficient must be in [0, 1)"
assert ELEMS_PER_WAVELENGTH >= 1, "elements_per_wavelength must be at least 1"
assert len(PLANE_HEIGHTS_Z) >= 1, "plane_heights_z_m must contain at least one height"
assert all(0 <= z <= LZ for z in PLANE_HEIGHTS_Z), (
    "every value in plane_heights_z_m must be within [0, height_z_m]")

# derived -- computed from the config values above, not independent choices
ESIZE = C0 / (FREQ_MAX * ELEMS_PER_WAVELENGTH)
_r = np.sqrt(1.0 - ALPHA_WALL)
Z_WALL = RHO_AIR * C0 * (1.0 + _r) / (1.0 - _r)   # wall impedance (Pa*s/m)
P_REF = 20e-6                       # 0 dB SPL reference pressure (Pa)


def rigid_room_modes(lx, ly, lz, c, fmin, fmax, nmax=8):
    """Closed-form rigid-wall eigenfrequencies (validation reference)."""
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
mapdl = launch_mapdl(nproc=4)
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

# circular source patch cut into the front wall (built from 4 quarter arcs)
kc = mapdl.k(100, 0, SRC_Y, SRC_Z)
k1 = mapdl.k(101, 0, SRC_Y + SRC_R, SRC_Z)
k2 = mapdl.k(102, 0, SRC_Y, SRC_Z + SRC_R)
k3 = mapdl.k(103, 0, SRC_Y - SRC_R, SRC_Z)
k4 = mapdl.k(104, 0, SRC_Y, SRC_Z - SRC_R)

l1 = mapdl.larc(k1, k2, kc, SRC_R)
l2 = mapdl.larc(k2, k3, kc, SRC_R)
l3 = mapdl.larc(k3, k4, kc, SRC_R)
l4 = mapdl.larc(k4, k1, kc, SRC_R)

src_area = mapdl.al(l1, l2, l3, l4)
front_wall_remainder = mapdl.asba(front_wall, src_area, keep1="", keep2="KEEP")

back_wall = mapdl.a(2, 3, 7, 6)
floor = mapdl.a(1, 2, 3, 4)
ceiling = mapdl.a(5, 6, 7, 8)
left_wall = mapdl.a(1, 2, 6, 5)
right_wall = mapdl.a(4, 3, 7, 8)

wall_areas = [front_wall_remainder, back_wall, floor, ceiling, left_wall, right_wall]
room_vol = mapdl.va(front_wall_remainder, src_area, back_wall, floor, ceiling,
                    left_wall, right_wall)

mapdl.vplot(cpos="iso", background="white")
lap("geometry")

# ==== MESH ===================================================================
mapdl.type(1)
mapdl.mat(1)
mapdl.mshape(1, "3D")
mapdl.mshkey(0)
mapdl.aesize(src_area, 0.03)   # finer mesh right at the source
mapdl.esize(ESIZE)
mapdl.vmesh(room_vol)
print(f"[mesh] {mapdl.mesh.n_elem} elements, {mapdl.mesh.n_node} nodes, "
      f"esize={ESIZE:.3f} m")
mapdl.eplot(background="white", show_edges=True, cpos="iso")
lap("mesh")

# ==== BOUNDARY CONDITIONS ====================================================
# source: normal surface velocity on the piston patch
mapdl.asel("S", "AREA", vmin=src_area)
mapdl.nsla("S", 1)
mapdl.sf("all", "SHLD", SRC_VELOCITY)
mapdl.allsel()

# walls: finite impedance on every boundary area except the source patch
mapdl.asel("S", "AREA", vmin=wall_areas[0])
for a in wall_areas[1:]:
    mapdl.asel("A", "AREA", vmin=a)
mapdl.nsla("S", 1)
mapdl.sf("all", "IMPD", Z_WALL)
mapdl.allsel()

listener_node = mapdl.queries.node(*LISTENER_XYZ)
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
    """Full-mesh complex pressure at the nearest solved frequency."""
    f = solved_freqs[np.argmin(np.abs(solved_freqs - target_freq))]
    mapdl.set(time=f, kimg=0)
    real = mapdl.get_array(entity="NODE", item1="PRES")
    mapdl.set(time=f, kimg=1)
    imag = mapdl.get_array(entity="NODE", item1="PRES")
    return f, mapdl.mesh.nnum, real + 1j * imag


def match_order(target_ids, source_ids, values):
    """Reorder values (in source_ids order) to line up with target_ids."""
    pos = {n: i for i, n in enumerate(source_ids)}
    return values[[pos[n] for n in target_ids]]


def to_db(pa):
    return 20 * np.log10(np.clip(np.abs(pa), 1e-12, None) / P_REF)


def listener_sweep(node, n_sets):
    """Pressure amplitude at one node across all frequencies (batched)."""
    mapdl.dim("PAMPL", "ARRAY", n_sets)
    with mapdl.non_interactive:
        for i in range(1, n_sets + 1):
            mapdl.set(lstep=1, sbstep=i, kimg=3)
            mapdl.run(f"*GET,PAMPL({i}),NODE,{node},PRES")
    return np.asarray(mapdl.parameters["PAMPL"]).ravel()


# ==== LISTENER SWEEP =========================================================
freqs = solved_freqs
listener_pressure = listener_sweep(listener_node, len(freqs))
listener_spl = to_db(listener_pressure)
lap("listener sweep")

# ==== RESONANCE PEAKS / DIPS =================================================
peak_idx, _ = find_peaks(listener_spl, prominence=3)
if len(peak_idx) == 0:
    peak_idx, _ = find_peaks(listener_spl)
peak_idx = peak_idx[np.argsort(listener_spl[peak_idx])[-NUM_PEAKS:]]
plot_freqs = sorted(freqs[peak_idx])

dip_idx, _ = find_peaks(-listener_spl, prominence=3)
dip_idx = dip_idx[np.argsort(listener_spl[dip_idx])[:NUM_DIPS]]
dip_freqs = sorted(freqs[dip_idx])

# tag each peak with its nearest analytical mode, if close enough
modes = rigid_room_modes(LX, LY, LZ, C0, FREQ_MIN, FREQ_MAX)
matched_modes = [nearest for pf in plot_freqs
                 for nearest in [min(modes, key=lambda fm: abs(fm - pf))]
                 if abs(nearest - pf) <= MODE_MATCH_TOL]

print(f"[sweep] peaks (Hz): {[round(float(f), 1) for f in plot_freqs]}")
print(f"[sweep] dips (Hz): {[round(float(f), 1) for f in dip_freqs]}")

# ==== LISTENER SPL PLOT ======================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, listener_spl, color="tab:orange")
for fm in matched_modes:
    ax.axvline(fm, color="gray", alpha=0.4, lw=0.9, ls="--")
for fd in dip_freqs:
    ax.axvline(fd, color="tab:red", alpha=0.4, lw=0.9, ls=":")

tick_freqs = sorted(set(round(float(f)) for f in list(plot_freqs) + list(dip_freqs)))
ax.set_xticks(tick_freqs)
ax.set_xticklabels([str(f) for f in tick_freqs], rotation=45)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("SPL, unweighted (dB)")
ax.set_title("Listener SPL response")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "listener_sweep.png"), dpi=150)
lap("listener sweep plot")

# ==== RESONANCE FIELD MAPS ===================================================


def plot_3d(f, field, tag):
    # unclipped mesh only shows its exterior (the walls) from outside
    grid = mapdl.mesh.grid.copy()
    grid.point_data["SPL (dB)"] = to_db(field)

    pl = pv.Plotter()
    pl.add_mesh(grid, scalars="SPL (dB)", cmap="jet", show_edges=False)
    pl.add_text(f"SPL field at {f:.1f} Hz ({tag})", font_size=12)
    pl.camera_position = "iso"
    pl.show()

    pl_save = pv.Plotter(off_screen=True)
    pl_save.add_mesh(grid, scalars="SPL (dB)", cmap="jet",
                     show_edges=False)
    pl_save.add_text(f"SPL field at {f:.1f} Hz ({tag})", font_size=12)
    pl_save.camera_position = "iso"
    pl_save.show(screenshot=os.path.join(
        DIR_3D_PRESSURE, f"spl_3d_{f:.0f}Hz_{tag}.png"))


def plot_plane(f, field, plane_z, tag):
    mask = np.abs(mapdl.mesh.nodes[:, 2] - plane_z) < 0.05
    xy = mapdl.mesh.nodes[mask]
    spl = to_db(field[mask])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tpc = ax.tricontourf(xy[:, 0], xy[:, 1], spl, levels=20, cmap="jet")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"SPL field at z={plane_z:.2f} m, f={f:.1f} Hz ({tag})")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="SPL (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(
        DIR_3D_PLANE, f"spl_plane_{f:.0f}Hz_z{plane_z:.2f}m_{tag}.png"), dpi=150)


mapdl.allsel()
for target_freq in plot_freqs:
    f, nnum, pres = nodal_pressure(target_freq)
    field = match_order(mapdl.mesh.nnum, nnum, pres)
    plot_3d(f, field, "peak")
    for plane_z in PLANE_HEIGHTS_Z:
        plot_plane(f, field, plane_z, "peak")
lap("peak field plots")

for target_freq in dip_freqs:
    f, nnum, pres = nodal_pressure(target_freq)
    field = match_order(mapdl.mesh.nnum, nnum, pres)
    plot_3d(f, field, "dip")
    for plane_z in PLANE_HEIGHTS_Z:
        plot_plane(f, field, plane_z, "dip")
lap("dip field plots")

# ==== DONE ===================================================================
mapdl.exit()
lap("mapdl exit")
print(f"[timer] TOTAL: {time.perf_counter() - _T0:.2f} s")