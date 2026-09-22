"""Harmonic FEM model of a rectangular room driven by a monopole speaker."""

import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import griddata

pv.OFF_SCREEN = True

from ansys.mapdl.core import launch_mapdl

# ==== OUTPUT FOLDERS ========================================================

_job_id = os.environ.get("SLURM_JOB_ID", "local")

OUTPUT_DIR = f"plots_{_job_id}"
DIR_3D_PRESSURE = os.path.join(OUTPUT_DIR, "3d_pressure")
DIR_3D_PLANE = os.path.join(OUTPUT_DIR, "3d_plane")
DIR_3D_DATA = os.path.join(OUTPUT_DIR, "3d_data")
DIR_MODEL = os.path.join(OUTPUT_DIR, "model")

os.makedirs(DIR_3D_PRESSURE, exist_ok=True)
os.makedirs(DIR_3D_PLANE, exist_ok=True)
os.makedirs(DIR_3D_DATA, exist_ok=True)
os.makedirs(DIR_MODEL, exist_ok=True)

JOBNAME = "room_fem"     # used for the .db export

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
SMART_SIZE = 6
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


# ==== ROOM GEOMETRY ==========================================================

_run_base = os.environ.get("MAPDL_RUN_BASE", os.getcwd())
_mapdl_log_dir = os.path.join(_run_base, "mapdl_run", f"job_{_job_id}")
os.makedirs(_mapdl_log_dir, exist_ok=True)

mapdl = launch_mapdl(
    timeout=120,
    run_location=_mapdl_log_dir,
)
lap("launch_mapdl")
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
    


mapdl.asel("S", "LOC", "Z", SRC_Z)
n_slice_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))

if slice_heights:
    print(f"[geometry] found {n_slice_areas} slice area(s) at z={SRC_Z} m")
    if n_slice_areas != 1:
        raise RuntimeError(
            f"expected exactly 1 interior area at z=SRC_Z ({SRC_Z} m), found "
            f"{n_slice_areas} -- the slicing may not have produced the "
            "expected geometry; check PLANE_HEIGHTS_Z and SRC_Z for "
            "near-duplicate heights.")

slice_area = int(mapdl.get_value(entity="area", entnum=0, item1="num", it1num="max"))
mapdl.hptcreate("AREA", slice_area, "", "COORD", SRC_X, SRC_Y, SRC_Z)
src_kp = int(mapdl.get_value(entity="kp", entnum=0, item1="num", it1num="max"))

# Confirm the hard point landed exactly on the slice plane

hp_x = mapdl.get_value(entity="kp", entnum=src_kp, item1="loc", it1num="x")
hp_y = mapdl.get_value(entity="kp", entnum=src_kp, item1="loc", it1num="y")
hp_z = mapdl.get_value(entity="kp", entnum=src_kp, item1="loc", it1num="z")
if abs(hp_z - SRC_Z) > 1e-9 or abs(hp_x - SRC_X) > 1e-9 or abs(hp_y - SRC_Y) > 1e-9:
    raise RuntimeError(
        f"hard point KP={src_kp} is at ({hp_x}, {hp_y}, {hp_z}) m, not "
        f"({SRC_X}, {SRC_Y}, {SRC_Z}) m -- it did not land on the slice.")
print(f"[geometry] source hard point KP={src_kp} confirmed at "
      f"({hp_x:.4f}, {hp_y:.4f}, {hp_z:.4f}) m, on the z={SRC_Z} m plane")

mapdl.allsel()
mapdl.asel("S", "LOC", "Z", LISTENER_Z)
n_listener_slice_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))

if slice_heights:
    print(f"[geometry] found {n_listener_slice_areas} slice area(s) at "
      f"z={LISTENER_Z} m (listener plane)")
    if n_listener_slice_areas != 1:
        raise RuntimeError(
        f"expected exactly 1 interior area at z=LISTENER_Z ({LISTENER_Z} "
        f"m), found {n_listener_slice_areas} -- check PLANE_HEIGHTS_Z, "
        "SRC_Z, and LISTENER_Z for near-duplicate heights.")


listener_slice_area = int(mapdl.get_value(entity="area", entnum=0, item1="num", it1num="max"))
mapdl.hptcreate("AREA", listener_slice_area, "", "COORD",
                LISTENER_X, LISTENER_Y, LISTENER_Z)
listener_kp = int(mapdl.get_value(entity="kp", entnum=0, item1="num", it1num="max"))

lhp_x = mapdl.get_value(entity="kp", entnum=listener_kp, item1="loc", it1num="x")
lhp_y = mapdl.get_value(entity="kp", entnum=listener_kp, item1="loc", it1num="y")
lhp_z = mapdl.get_value(entity="kp", entnum=listener_kp, item1="loc", it1num="z")
if (abs(lhp_z - LISTENER_Z) > 1e-9 or abs(lhp_x - LISTENER_X) > 1e-9
        or abs(lhp_y - LISTENER_Y) > 1e-9):
    raise RuntimeError(
        f"listener hard point KP={listener_kp} is at ({lhp_x}, {lhp_y}, "
        f"{lhp_z}) m, not ({LISTENER_X}, {LISTENER_Y}, {LISTENER_Z}) m "
        "-- it did not land on the slice.")

print(f"[geometry] listener hard point KP={listener_kp} confirmed at "
      f"({lhp_x:.4f}, {lhp_y:.4f}, {lhp_z:.4f}) m, on the "
      f"z={LISTENER_Z} m plane")



mapdl.allsel()

lap("geometry")

# ==== MESH ===================================================================

mapdl.mshape(1, "3D")
mapdl.mshkey(0)

mapdl.esize(ESIZE)  
#mapdl.smrtsize(SMART_SIZE) 
#mapdl.kesize(src_kp, HP_ELEM_SIZE)
mapdl.kesize(listener_kp, HP_ELEM_SIZE)

mapdl.allsel()

mapdl.type(1)
mapdl.mat(1)
mapdl.vmesh("all")

print(f"[mesh] {mapdl.mesh.n_elem} elements, {mapdl.mesh.n_node} nodes ")

mapdl.allsel()

# Mesh for ParaView.
mesh_grid = mapdl.mesh.grid.copy()
mesh_grid.save(os.path.join(DIR_3D_DATA, "mesh.vtu"))

lap("mesh")

# =====RESOLVE KEYPOINTS=====

# Source

mapdl.ksel("S", "KP", vmin=src_kp)
mapdl.nslk()
n_src_nodes = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
if n_src_nodes != 1:
    raise RuntimeError(
        f"expected exactly 1 node at hard point KP={src_kp}, found "
        f"{n_src_nodes} -- the hard point likely wasn't meshed "
        "conformally; check mesh.n_node before/after meshing.")
src_node = int(mapdl.get_value(entity="node", entnum=0, item1="num", it1num="min"))
mapdl.allsel()


mapdl.nsel("S", "NODE", vmin=src_node)
mapdl.esln("S", 0)
n_attached_elems = int(mapdl.get_value(entity="elem", entnum=0, item1="count"))
print(f"[nodes] source node {src_node} has {n_attached_elems} FLUID221 element(s) attached")
if n_attached_elems == 0:
    raise RuntimeError(
        f"node {src_node} has no elements attached -- it exists in the "
        "database but isn't actually part of the mesh, so BF loads on it "
        "have no effect on the solve. This points at the hard point "
        "ending up isolated (e.g. the slice area wasn't fully consumed "
        "by VMESH, or the two split volumes didn't mesh as one "
        "conformal set).")
mapdl.allsel()

_sx = mapdl.get_value(entity="node", entnum=src_node, item1="loc", it1num="x")
_sy = mapdl.get_value(entity="node", entnum=src_node, item1="loc", it1num="y")
_sz = mapdl.get_value(entity="node", entnum=src_node, item1="loc", it1num="z")
print(f"[nodes] source hard point KP={src_kp} -> node {src_node} at "
      f"({_sx:.4f}, {_sy:.4f}, {_sz:.4f}) m")

#Listener

mapdl.ksel("S", "KP", vmin=listener_kp)
mapdl.nslk()
n_listener_nodes = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
if n_listener_nodes != 1:
    raise RuntimeError(
        f"expected exactly 1 node at listener hard point KP={listener_kp}, "
        f"found {n_listener_nodes} -- the hard point likely wasn't meshed "
        "conformally; check mesh.n_node before/after meshing.")
listener_node = int(mapdl.get_value(entity="node", entnum=0, item1="num", it1num="min"))
mapdl.allsel()

mapdl.nsel("S", "NODE", vmin=listener_node)
mapdl.esln("S", 0)
n_listener_elems = int(mapdl.get_value(entity="elem", entnum=0, item1="count"))
print(f"[nodes] listener node {listener_node} has {n_listener_elems} FLUID221 "
      f"element(s) attached")
if n_listener_elems == 0:
    raise RuntimeError(
        f"listener node {listener_node} has no elements attached -- it "
        "exists in the database but isn't actually part of the mesh.")
mapdl.allsel()

_lx = mapdl.get_value(entity="node", entnum=listener_node, item1="loc", it1num="x")
_ly = mapdl.get_value(entity="node", entnum=listener_node, item1="loc", it1num="y")
_lz = mapdl.get_value(entity="node", entnum=listener_node, item1="loc", it1num="z")
print(f"[nodes] listener hard point KP={listener_kp} -> node {listener_node} at "
      f"({_lx:.4f}, {_ly:.4f}, {_lz:.4f}) m")

# ==== WALL ABSORPTION ========================================================

mapdl.allsel()
if ALPHA_WALL > 0:
    mapdl.asel("S", "LOC", "X", 0)
    mapdl.asel("A", "LOC", "X", LX)
    mapdl.asel("A", "LOC", "Y", 0)
    mapdl.asel("A", "LOC", "Y", LY)
    mapdl.asel("A", "LOC", "Z", 0)
    mapdl.asel("A", "LOC", "Z", LZ)
    n_wall_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))
    print(f"[abs] {n_wall_areas} exterior wall area(s) selected for "
          f"absorption (alpha={ALPHA_WALL})")
    if n_wall_areas == 0:
        raise RuntimeError(
            "no exterior wall areas found at the 6 bounding planes -- "
            "wall absorption BC was not applied.")

    mapdl.nsla("S", 1)   # include nodes shared with unselected areas/edges
    n_wall_nodes = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
    mapdl.sf("ALL", "ATTN", ALPHA_WALL)
    print(f"[abs] absorption coefficient alpha={ALPHA_WALL} applied to "
          f"{n_wall_nodes} wall node(s)")
else:
    print("[abs] ALPHA_WALL <= 0 -- walls left fully rigid "
          "(no absorption BC applied)")
    
for label, node in (("source", src_node), ("listener", listener_node)):
    mapdl.nsel("S", "NODE", vmin=node)
    mapdl.sf("ALL", "ATTN", 0)
    print(f"[abs] {label} node {node} exempted from wall absorption "
          f"(ATTN reset to 0)")
mapdl.allsel()    

# ==== BOUNDARY CONDITIONS ====================================================

# Frequency dependent point mass source

N_TABLE = 50
table_freqs = np.linspace(FREQ_MIN, FREQ_MAX, N_TABLE)
table_values = SRC_MASS_MAGNITUDE / table_freqs

SRC_TABLE = "src_mass_tab"
mapdl.dim(SRC_TABLE, "TABLE", N_TABLE, 1, 1, "FREQ")
for i, (f_i, v_i) in enumerate(zip(table_freqs, table_values), start=1):
    mapdl.run(f"{SRC_TABLE}({i},0) = {f_i}")
    mapdl.run(f"{SRC_TABLE}({i},1) = {v_i}")
print(f"[bc] frequency-dependent source table '{SRC_TABLE}': {N_TABLE} "
      f"points, {table_values.min():.3e} to {table_values.max():.3e} kg/s "
      f"over {FREQ_MIN:.1f}-{FREQ_MAX:.1f} Hz")

mapdl.bf(src_node, "MASS", f"%{SRC_TABLE}%", 0.0)
print(f"[bc] source keypoint {src_kp} -> node {src_node} at "
      f"({SRC_X}, {SRC_Y}, {SRC_Z}) m, frequency-scaled mass source "
      f"(base {SRC_MASS_MAGNITUDE} kg/s at 1 Hz) applied")


print(mapdl.bflist(src_node, "all"))

mapdl.allsel()

mapdl.save(fname=JOBNAME, ext="db")
mapdl.download(f"{JOBNAME}.db", target_dir=DIR_MODEL)

lap("boundary conditions")

# ==== HARMONIC SOLVE =========================================================

mapdl.run("/SOLU")
mapdl.antype(3)
mapdl.harfrq(freqb=FREQ_MIN, freqe=FREQ_MAX)
mapdl.autots("off")
mapdl.nsubst(N_SUBSTEPS)
mapdl.kbc(0)
mapdl.outres("erase")
mapdl.outres("all", "all")  
mapdl.outres("nsol", "all")
mapdl.solve()
mapdl.finish()
mapdl.download(f"{JOBNAME}.rst", target_dir=DIR_MODEL)
lap("harmonic solve")

mapdl.post1()
solved_freqs = np.unique(mapdl.post_processing.time_values)

# ==== POST-PROCESSING HELPERS ================================================

def nodal_pressure(target_freq):

    """Return the complex nodal pressure at the target frequency."""
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


def listener_sweep(node, n_sets):
    """Complex pressure at a listener node."""
    pres = np.zeros(n_sets, dtype=complex)

    for i in range(1, n_sets + 1):
        mapdl.set(lstep=1, sbstep=i, kimg=0)
        p_re = mapdl.get_value(entity="node", entnum=node, item1="pres")

        mapdl.set(lstep=1, sbstep=i, kimg=1)
        p_im = mapdl.get_value(entity="node", entnum=node, item1="pres")

        pres[i - 1] = p_re + 1j * p_im

    return pres

def listener_velocity_sweep(node, n_sets):
    """Complex particle velocity at a listener node."""
    mapdl.allsel()
    mapdl.nsel("S", "NODE", vmin=node)
    mapdl.esln("S", 0)
    attached_elems = mapdl.mesh.enum.tolist()

    vel = np.zeros((n_sets, 4), dtype=complex)

    for i in range(1, n_sets + 1):
        comps = {}
        for part, kimg in (("re", 0), ("im", 1)):
            mapdl.set(lstep=1, sbstep=i, kimg=kimg)
            mapdl.etable("pgx", "PG", "X")
            mapdl.etable("pgy", "PG", "Y")
            mapdl.etable("pgz", "PG", "Z")
            comps[part] = {
                c: np.mean([
                    mapdl.get_value(entity="elem", entnum=e, item1="etab",
                                     it1num=f"pg{c.lower()}")
                    for e in attached_elems
                ])
                for c in ("x", "y", "z")
            }

        vx = comps["re"]["x"] + 1j * comps["im"]["x"]
        vy = comps["re"]["y"] + 1j * comps["im"]["y"]
        vz = comps["re"]["z"] + 1j * comps["im"]["z"]
        vel[i - 1] = [vx, vy, vz, np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)]

    mapdl.allsel()   # restore full selection once, after the sweep is done
    return vel

# ==== LISTENER SWEEP ============================================================

freqs = solved_freqs
listener_pressure = listener_sweep(listener_node, len(freqs))
listener_spl = to_db(listener_pressure)
listener_velocity = listener_velocity_sweep(listener_node, len(freqs))
listener_vmag_rms = np.abs(listener_velocity[:, 3]) / np.sqrt(2)   # RMS |v|, m/s

vmag = np.sqrt(np.abs(listener_velocity[:, 0])**2
                         + np.abs(listener_velocity[:, 1])**2
                         + np.abs(listener_velocity[:, 2])**2) / np.sqrt(2)

lap("listener sweep")

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
print(f"[sweep] modes selected for field plots (Hz, band-stratified, "
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


# ==== LISTENER VELOCITY PLOT =================================================

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(freqs, listener_vmag_rms, color="tab:blue")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Particle velocity, RMS (m/s)")
ax.set_title(f"Listener particle velocity magnitude at ({_lx:.2f}, {_ly:.2f}, {_lz:.2f}) m")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "listener_velocity_sweep.png"), dpi=150)
plt.close(fig)
lap("listener velocity plot")

# ==== LISTENER SPL + VELOCITY COMBINED PLOT ==================================

fig, ax1 = plt.subplots(figsize=(8, 4.5))

ax1.plot(freqs, listener_spl, color="tab:orange", label="SPL")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("SPL, unweighted (dB)", color="tab:orange")
ax1.tick_params(axis="y", labelcolor="tab:orange")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(freqs, listener_vmag_rms, color="tab:blue", label="Velocity")
ax2.set_ylabel("Particle velocity, RMS (m/s)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

ax1.set_title(f"Listener SPL and particle velocity at ({_lx:.2f}, {_ly:.2f}, {_lz:.2f}) m")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "listener_spl_velocity_combined.png"), dpi=150)
plt.close(fig)
lap("listener combined plot")

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


def plot_plane(f, plane_z, tag, vmin, vmax):
    
    mapdl.allsel()
    mapdl.nsel("S", "LOC", "Z", plane_z)
    n_pts = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
    if n_pts == 0:
        raise RuntimeError(
            f"no nodes found at z={plane_z:.3f} m -- this height wasn't "
            "included as a geometry slice plane; check PLANE_HEIGHTS_Z "
            "against the slice_heights list built in ROOM GEOMETRY.")

    coords = mapdl.mesh.nodes
    if coords.shape[0] != n_pts:
        raise RuntimeError(
            f"mapdl.mesh.nodes returned {coords.shape[0]} nodes but "
            f"{n_pts} are selected at z={plane_z} m -- selection state "
            "and mesh.nodes are out of sync; re-check ordering assumptions.")
    pts_x, pts_y = coords[:, 0], coords[:, 1]

    mapdl.set(time=f, kimg=0)
    p_re = mapdl.post_processing.nodal_pressure()
    mapdl.set(time=f, kimg=1)
    p_im = mapdl.post_processing.nodal_pressure()
    if p_re.shape[0] != n_pts or p_im.shape[0] != n_pts:
        raise RuntimeError(
            f"post_processing.nodal_pressure() returned "
            f"{p_re.shape[0]}/{p_im.shape[0]} values but {n_pts} nodes "
            f"are selected at z={plane_z} m -- selection state mismatch.")
    spl = np.clip(to_db(p_re + 1j * p_im), vmin, vmax)

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
        DIR_3D_PLANE, f"spl_plane_{f:.0f}Hz_z{plane_z:.3f}m_{tag}.png"), dpi=150)
    plt.close(fig)
    mapdl.allsel()


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
        plot_plane(f, plane_z, "peak", global_min, global_max)
lap("peak field plots")

# ==== DONE ===================================================================
mapdl.exit()
lap("mapdl exit")
import shutil
shutil.rmtree(_mapdl_log_dir, ignore_errors=True)
print(f"[cleanup] removed MAPDL scratch dir {_mapdl_log_dir}; "
      f"kept {JOBNAME}.db and {JOBNAME}.rst in {DIR_MODEL}")
print(f"[timer] TOTAL: {time.perf_counter() - _T0:.2f} s")
