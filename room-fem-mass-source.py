"""Harmonic FEM model of a rectangular room driven by a monopole speaker."""

import time
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import subprocess

pv.OFF_SCREEN = True

from ansys.mapdl.core import launch_mapdl

# ==== OUTPUT FOLDERS ========================================================

_job_id = os.environ.get("SLURM_JOB_ID", "local")

OUTPUT_DIR = os.path.join("runs", _job_id)
DIR_3D_PRESSURE = os.path.join(OUTPUT_DIR, "3d_pressure")
DIR_3D_PLANE = os.path.join(OUTPUT_DIR, "3d_plane")
DIR_3D_DATA = os.path.join(OUTPUT_DIR, "3d_data")
DIR_MODEL = os.path.join(OUTPUT_DIR, "model")

os.makedirs(DIR_3D_PRESSURE, exist_ok=True)
os.makedirs(DIR_3D_PLANE, exist_ok=True)
os.makedirs(DIR_3D_DATA, exist_ok=True)
os.makedirs(DIR_MODEL, exist_ok=True)

JOBNAME = "room_fem"

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

# source is a point excitation
SRC_X = _cfg["source"]["x_m"]
SRC_Y = _cfg["source"]["y_m"]
SRC_Z = _cfg["source"]["z_m"]
SRC_MASS_MAGNITUDE = _cfg["source"]["mass_source_kg_s"]

LISTENER_XYZ = (_cfg["listener"]["x_m"], _cfg["listener"]["y_m"],
                _cfg["listener"]["z_m"])
LISTENER_X, LISTENER_Y, LISTENER_Z = LISTENER_XYZ

C0 = _cfg["acoustics"]["speed_of_sound_m_s"]
RHO_AIR = _cfg["acoustics"]["air_density_kg_m3"]

ALPHA_WALL = _cfg["acoustics"]["wall_absorption_coefficient"]

FREQ_MIN = _cfg["analysis"]["freq_min_hz"]
FREQ_MAX = _cfg["analysis"]["freq_max_hz"]
N_SUBSTEPS = _cfg["analysis"]["num_substeps"]
ELEMS_PER_WAVELENGTH = _cfg["analysis"]["elements_per_wavelength"]

PLANE_HEIGHTS_Z = _cfg["plotting"]["plane_heights_z_m"]

# Wavelength-based target element size.
ESIZE = C0 / (FREQ_MAX * ELEMS_PER_WAVELENGTH)
HP_ELEM_SIZE = ESIZE / 2    # target edge length right at the hard point

P_REF = 20e-6               # 0 dB SPL reference pressure (Pa)


def rigid_room_modes(lx, ly, lz, c, fmin, fmax, nmax=8):
    """Closed-form rigid-wall eigenfrequencies -- the theoretical modes to
    check the real room's response against."""
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


# ==== LAUNCH MAPDL ===========================================================


_mapdl_run_dir = os.path.abspath(DIR_MODEL)

nproc = int(os.environ.get("SLURM_NTASKS", 1))

mapdl = launch_mapdl(
    nproc=nproc,
    timeout=80,
    run_location=_mapdl_run_dir,
    jobname=JOBNAME,
)

_n_ansys = subprocess.run(["pgrep", "-c", "-f", "ansys.e"],
                          capture_output=True, text=True).stdout.strip() or "?"
print(f"[run] job {_job_id} on {os.uname().nodename}: {nproc} core(s), "
      f"{_n_ansys} MAPDL process(es)\n"
      f"[run] room {LX} x {LY} x {LZ} m, {FREQ_MIN}-{FREQ_MAX} Hz in "
      f"{N_SUBSTEPS} steps, element size {ESIZE:.3f} m "
      f"({ELEMS_PER_WAVELENGTH}/wavelength), alpha={ALPHA_WALL}", flush=True)

lap("launch_mapdl")

# ==== ROOM GEOMETRY ==========================================================

mapdl.clear()
mapdl.filname(JOBNAME)
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

# Geometric slices at source, probe, and every plotting height

SLICE_TOL = 1e-6
_raw_heights = [SRC_Z, LISTENER_Z] + list(PLANE_HEIGHTS_Z)
_interior_heights = [z for z in _raw_heights if SLICE_TOL < z < LZ - SLICE_TOL]
slice_heights = []
for z in sorted(_interior_heights):
    if not slice_heights or abs(z - slice_heights[-1]) > SLICE_TOL:
        slice_heights.append(z)

if slice_heights:
    print(f"[geometry] slicing the room at z = {slice_heights} m "
          f"(source + listener hard points + plotting planes)")

_wp_z = 0.0
for z in slice_heights:
    mapdl.allsel()
    mapdl.wpoffs(0, 0, z - _wp_z)
    mapdl.vsbw("ALL", "", "DELETE")
    _wp_z = z
mapdl.wpoffs(0, 0, -_wp_z)   # restore the working plane to the global origin
mapdl.allsel()

if slice_heights:
    mapdl.vglue("ALL")
    mapdl.allsel()
    n_vols = int(mapdl.get_value(entity="volu", entnum=0, item1="count"))
    print(f"[geometry] {n_vols} volume(s) after slicing at {slice_heights} m "
          f"(expected {len(slice_heights) + 1})")
    if n_vols != len(slice_heights) + 1:
        raise RuntimeError(
            f"expected {len(slice_heights) + 1} volumes after slicing at "
            f"{slice_heights} m, found {n_vols} -- one or more of the "
            "requested slice heights likely didn't produce a real cut; check "
            "each height individually with ASEL,S,LOC,Z,<height> in APDL.")


def make_hard_point(label, x, y, z):
    """Keypoint exactly at (x, y, z) at height z."""
    mapdl.allsel()
    mapdl.asel("S", "LOC", "Z", z)
    n_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))
    if n_areas != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 horizontal area at z={z} m, found "
            f"{n_areas} -- check PLANE_HEIGHTS_Z, SRC_Z and LISTENER_Z for "
            "near-duplicate heights or a height outside the room.")
    area = int(mapdl.get_value(entity="area", entnum=0, item1="num", it1num="max"))
    mapdl.hptcreate("AREA", area, "", "COORD", x, y, z)
    kp = int(mapdl.get_value(entity="kp", entnum=0, item1="num", it1num="max"))

    loc = [mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num=c)
           for c in ("x", "y", "z")]
    if max(abs(a - b) for a, b in zip(loc, (x, y, z))) > 1e-9:
        raise RuntimeError(
            f"{label} hard point KP={kp} is at ({loc[0]}, {loc[1]}, {loc[2]}) m, "
            f"not ({x}, {y}, {z}) m -- it did not land on the slice.")
    print(f"[geometry] {label} hard point KP={kp} confirmed at "
          f"({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f}) m, on the z={z} m plane ")
    mapdl.allsel()
    return kp


src_kp = make_hard_point("source", SRC_X, SRC_Y, SRC_Z)
listener_kp = make_hard_point("listener", LISTENER_X, LISTENER_Y, LISTENER_Z)

lap("geometry")

# ==== MESH ===================================================================

mapdl.mshape(1, "3D")
mapdl.mshkey(0)

mapdl.esize(ESIZE)
# mapdl.kesize(src_kp, HP_ELEM_SIZE)
# mapdl.kesize(listener_kp, HP_ELEM_SIZE)

mapdl.allsel()

mapdl.type(1)
mapdl.mat(1)
mapdl.vmesh("all")

mapdl.allsel()

mesh_grid = mapdl.mesh.grid.copy()
print(f"[mesh] {mesh_grid.n_cells} elements, {mesh_grid.n_points} nodes")
mesh_grid.save(os.path.join(DIR_3D_DATA, "mesh.vtu"))
 
lap("mesh")

# ==== RESOLVE KEYPOINTS ======================================================


def node_at_kp(label, kp):
    """Single node on keypoint kp and its (x, y, z)."""
    mapdl.allsel()
    mapdl.ksel("S", "KP", vmin=kp)
    mapdl.nslk()
    n_nodes = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
    if n_nodes != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 node at hard point KP={kp}, found "
            f"{n_nodes} -- the hard point likely wasn't meshed conformally.")
    node = int(mapdl.get_value(entity="node", entnum=0, item1="num", it1num="min"))
    mapdl.allsel()

    mapdl.nsel("S", "NODE", vmin=node)
    mapdl.esln("S", 0)
    n_elems = int(mapdl.get_value(entity="elem", entnum=0, item1="count"))
    mapdl.allsel()
    print(f"[nodes] {label} node {node} has {n_elems} FLUID221 element(s) attached")
    if n_elems == 0:
        raise RuntimeError(
            f"{label} node {node} has no elements attached -- it exists in the "
            "database but isn't part of the mesh, so loads/results on it are "
            "meaningless (e.g. the slice area wasn't fully consumed by VMESH).")

    xyz = tuple(mapdl.get_value(entity="node", entnum=node, item1="loc", it1num=c)
                for c in ("x", "y", "z"))
    print(f"[nodes] {label} hard point KP={kp} -> node {node} at "
          f"({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) m")
    return node, xyz


src_node, (_sx, _sy, _sz) = node_at_kp("source", src_kp)
listener_node, (_lx, _ly, _lz) = node_at_kp("listener", listener_kp)

# ==== WALL ABSORPTION ========================================================

mapdl.allsel()
if ALPHA_WALL > 0:
    mapdl.nsel("S", "EXT")                     
    n_wall_nodes = int(mapdl.get_value("NODE", 0, "COUNT"))
    if n_wall_nodes == 0:
        raise RuntimeError("no exterior nodes found -- ATTN not applied")
    mapdl.sf("ALL", "ATTN", ALPHA_WALL)
    print(f"[abs] alpha={ALPHA_WALL} applied to {n_wall_nodes} wall node(s)")
mapdl.allsel()

# ==== BOUNDARY CONDITIONS ====================================================

# Frequency dependent point mass source

table_freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_SUBSTEPS + 1)
N_TABLE = len(table_freqs)
table_values = SRC_MASS_MAGNITUDE / (2 * np.pi * table_freqs)

SRC_TABLE = "src_mass_tab"
mapdl.load_table(SRC_TABLE, np.column_stack([table_freqs, table_values]),
                 var1="FREQ")
print(f"[bc] frequency-dependent source table '{SRC_TABLE}': {N_TABLE} "
      f"points, {table_values.min():.3e} to {table_values.max():.3e} kg/s "
      f"over {FREQ_MIN:.1f}-{FREQ_MAX:.1f} Hz")
 
mapdl.bf(src_node, "MASS", f"%{SRC_TABLE}%", 0.0)
print(f"[bc] source keypoint {src_kp} -> node {src_node} at "
      f"({SRC_X}, {SRC_Y}, {SRC_Z}) m, frequency-scaled mass source "
      f"(base {SRC_MASS_MAGNITUDE} kg/s at 1 Hz) applied")
  
mapdl.allsel()

mapdl.save(fname=JOBNAME, ext="db")    

lap("boundary conditions")

# ==== HARMONIC SOLVE =========================================================

mapdl.run("/SOLU")
mapdl.run("DDOPTION,FREQ,1") 
mapdl.antype(3)
mapdl.harfrq(freqb=FREQ_MIN, freqe=FREQ_MAX)
mapdl.autots("off")
mapdl.nsubst(N_SUBSTEPS)
mapdl.kbc(1)
mapdl.outres("erase")
mapdl.outres("all", "none")
mapdl.outres("nsol", "all")
mapdl.outpr("ALL", "NONE")        
mapdl.run("SOLVE", mute=True)
mapdl.finish()                      
lap("harmonic solve")

mapdl.post1()
solved_freqs = np.unique(mapdl.post_processing.time_values)

# ==== POST-PROCESSING HELPERS ================================================


def nodal_pressure(target_freq):
    
    f = solved_freqs[np.argmin(np.abs(solved_freqs - target_freq))]
    mapdl.allsel()
    mapdl.set(time=f, kimg=0)
    real = mapdl.post_processing.nodal_pressure()
    mapdl.set(time=f, kimg=1)
    imag = mapdl.post_processing.nodal_pressure()
    return f, real + 1j * imag


def to_db(pa):
    p_rms = np.abs(pa) / np.sqrt(2)
    return 20 * np.log10(np.clip(p_rms, 1e-12, None) / P_REF)


def pressure_sweep(node, n_sets):
    """Complex pressure at the listener node for every solved frequency."""
    mapdl.finish()
    mapdl.post26()
    mapdl.nsol(2, node, "PRES")
    mapdl.vget("PR_RE", 2, kcplx=0)
    mapdl.vget("PR_IM", 2, kcplx=1)
    pres = (np.array(mapdl.parameters["PR_RE"]).ravel()
            + 1j * np.array(mapdl.parameters["PR_IM"]).ravel())
    mapdl.finish()
    mapdl.post1()
    if len(pres) != n_sets:
        raise RuntimeError(
            f"pressure_sweep returned {len(pres)} values, expected {n_sets}.")
    return pres

# ==== PRESSURE SWEEPS ========================================================

freqs = solved_freqs

listener_pressure = pressure_sweep(listener_node, len(freqs))
listener_spl = to_db(listener_pressure)

np.savetxt(os.path.join(OUTPUT_DIR, "listener_sweep.csv"),
           np.column_stack([freqs, listener_pressure.real,
                            listener_pressure.imag, listener_spl]),
           delimiter=",", fmt="%.10g",
           header="freq_hz,p_re_pa,p_im_pa,spl_db", comments="")
lap("pressure sweep")

# ==== THEORETICAL MODES ======================================================

modes = rigid_room_modes(LX, LY, LZ, C0, FREQ_MIN, FREQ_MAX)

N_BANDS = 4                    # number of frequency bands
PLANES_PER_BAND = 2            # how many modes to plot per band
band_edges = np.linspace(FREQ_MIN, FREQ_MAX, N_BANDS + 1)

plot_freqs = []
for lo, hi in zip(band_edges[:-1], band_edges[1:]):
    band_modes = [m for m in modes if lo <= m < hi]
    if not band_modes:
        continue
    band_spl = [listener_spl[np.argmin(np.abs(freqs - m))] for m in band_modes]
    order = np.argsort(band_spl)[::-1][:PLANES_PER_BAND]
    plot_freqs.extend(band_modes[i] for i in order)

plot_freqs = sorted(plot_freqs)
_plot_freqs_str = ", ".join(f"{f:.1f}" for f in plot_freqs)
print(f"[sweep] modes selected for plots (Hz), "
      f"{PLANES_PER_BAND}/band): [{_plot_freqs_str}]")

# ==== LISTENER SPL PLOT ======================================================

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, listener_spl, color="tab:orange")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("SPL, unweighted (dB)")
ax.set_title(f"Listener SPL response at ({_lx:.2f}, {_ly:.2f}, {_lz:.2f}) m")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "listener_sweep.png"), dpi=150)
plt.close(fig)
lap("listener sweep plot")

# ==== RESONANCE FIELD MAPS ===================================================

mapdl.allsel()
plot_grid = mesh_grid
pts_all = np.asarray(plot_grid.points)
if pts_all.shape[0] != mapdl.mesh.n_node:
    raise RuntimeError(
        f"grid has {pts_all.shape[0]} points but the mesh has "
        f"{mapdl.mesh.n_node} nodes -- pressure arrays won't line up with "
        "the grid points.")


def plot_3d(f, field, tag, vmin, vmax):
    grid = plot_grid.copy()
    grid.point_data["SPL (dB)"] = to_db(field)

    grid.save(os.path.join(DIR_3D_DATA, f"spl_3d_{f:.1f}Hz_{tag}.vtu"))

    pl_save = pv.Plotter(off_screen=True)
    pl_save.add_mesh(grid, scalars="SPL (dB)", cmap="jet",
                     show_edges=False, clim=[vmin, vmax],
                     scalar_bar_args={"fmt": "%.0f"})
    pl_save.add_text(f"SPL field at {f:.1f} Hz ({tag})", font_size=12)
    pl_save.camera_position = "iso"
    pl_save.show(screenshot=os.path.join(
        DIR_3D_PRESSURE, f"spl_3d_{f:.1f}Hz_{tag}.png"))


def plot_plane(f, field, plane_z, tag, vmin, vmax):
    
    mask = np.isclose(pts_all[:, 2], plane_z, atol=1e-6)
    if not mask.any():
        raise RuntimeError(
            f"no nodes found at z={plane_z:.3f} m -- this height wasn't "
            "included as a geometry slice plane; check PLANE_HEIGHTS_Z "
            "against the slice_heights list built in ROOM GEOMETRY.")
    if field.shape[0] != pts_all.shape[0]:
        raise RuntimeError(
            f"field has {field.shape[0]} values but the grid has "
            f"{pts_all.shape[0]} points -- ordering can't be trusted.")

    pts_x, pts_y = pts_all[mask, 0], pts_all[mask, 1]
    spl = np.clip(to_db(field[mask]), vmin, vmax)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    levels = np.linspace(vmin, vmax, 101)

    tpc = ax.tricontourf(pts_x, pts_y, spl, levels=levels, cmap="jet")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"SPL field at z={plane_z:.3f} m, f={f:.1f} Hz ({tag})")
    ax.set_aspect("equal")
    fig.colorbar(tpc, ax=ax, label="SPL (dB)", format="%.0f")
    fig.tight_layout()
    fig.savefig(os.path.join(
        DIR_3D_PLANE, f"spl_plane_{f:.1f}Hz_z{plane_z:.3f}m_{tag}.png"), dpi=150)
    plt.close(fig)


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

KEEP = {f"{JOBNAME}.db", f"{JOBNAME}.rst"}
KEEP_EXT = {".out", ".err", ".log"}
_removed = 0
for _f in glob.glob(os.path.join(_mapdl_run_dir, "*")):
    _name = os.path.basename(_f)
    if (os.path.isfile(_f) and _name not in KEEP
            and os.path.splitext(_name)[1].lower() not in KEEP_EXT):
        os.remove(_f)
        _removed += 1
print(f"[cleanup] removed {_removed} MAPDL scratch file(s); kept "
      f"{JOBNAME}.db, {JOBNAME}.rst and MAPDL logs in {DIR_MODEL}")
print(f"[done] results in {OUTPUT_DIR}")
print(f"[timer] TOTAL: {time.perf_counter() - _T0:.2f} s")
