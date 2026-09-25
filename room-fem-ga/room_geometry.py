import json
import os
import numpy as np
from ansys.mapdl.core import launch_mapdl

P_REF = 20e-6          
JP_REF = 1e-12          

N_TABLE = 50        
MONOPOLE_SIGNS = np.array([-1.0, +1.0, +1.0, -1.0])   
DIPOLE_PAIRS = [(0, 1), (2, 3)]


# ==== CONFIG =================================================================

def load_config(path="room_config.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"Config file '{path}' not found. Create one alongside this "
            f"script (see room_config.json) or point at an existing file.")


def rigid_room_modes(lx, ly, lz, c, fmin, fmax, nmax=8):
   
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


def to_db(pa, p_ref=P_REF):
    p_rms = np.abs(pa) / np.sqrt(2)
    return 20 * np.log10(np.clip(p_rms, 1e-12, None) / p_ref)


# ==== MODEL ===================================================================

class RoomModel:
    def __init__(self, cfg, output_dir, jobname="room_acoustics_control",
                 mapdl_run_base=None, timeout=120):
        self.cfg = cfg
        self.output_dir = output_dir
        self.jobname = jobname

        self.LX = cfg["room"]["length_x_m"]
        self.LY = cfg["room"]["length_y_m"]
        self.LZ = cfg["room"]["height_z_m"]

        self.SRC_X = cfg["source"]["x_m"]
        self.SRC_Y = cfg["source"]["y_m"]
        self.SRC_Z = cfg["source"]["z_m"]
        self.SRC_MASS_MAGNITUDE = cfg["source"]["mass_source_kg_s"]

        self.LISTENER_X = cfg["listener"]["x_m"]
        self.LISTENER_Y = cfg["listener"]["y_m"]
        self.LISTENER_Z = cfg["listener"]["z_m"]

        self.C0 = cfg["acoustics"]["speed_of_sound_m_s"]
        self.RHO_AIR = cfg["acoustics"]["air_density_kg_m3"]
        self.ALPHA_WALL = cfg["acoustics"]["wall_absorption_coefficient"]

        self.FREQ_MIN = cfg["analysis"]["freq_min_hz"]
        self.FREQ_MAX = cfg["analysis"]["freq_max_hz"]
        self.N_SUBSTEPS = cfg["analysis"]["num_substeps"]
        self.ELEMS_PER_WAVELENGTH = cfg["analysis"]["elements_per_wavelength"]

        self.PLANE_HEIGHTS_Z = cfg["plotting"]["plane_heights_z_m"]

        self.ESIZE = self.C0 / (self.FREQ_MAX * self.ELEMS_PER_WAVELENGTH)

        # ---- control source (longitudinal quadrupole) ----
        CTRL = cfg["control_source"]
        self.CTRL_X, self.CTRL_Y, self.CTRL_Z = (
            CTRL["center_x_m"], CTRL["center_y_m"], CTRL["z_m"])
        axis = np.array([CTRL["axis_x"], CTRL["axis_y"]], dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.D_SPACING = CTRL["monopole_spacing_m"]
        self.CTRL_HP_ELEM_SIZE = CTRL.get("hard_point_elem_size_m") or (
            min(self.ESIZE, self.D_SPACING) / 2.0)
        self.MASS_REF = self.SRC_MASS_MAGNITUDE  

        offsets = (np.arange(4) - 1.5) * self.D_SPACING
        self.CTRL_MONOPOLE_XY = np.array([
            [self.CTRL_X + off * self.axis[0], self.CTRL_Y + off * self.axis[1]]
            for off in offsets
        ])
        margin = max(1e-6, 0.02 * self.CTRL_HP_ELEM_SIZE)
        bad = [(i, x, y) for i, (x, y) in enumerate(self.CTRL_MONOPOLE_XY)
               if not (margin <= x <= self.LX - margin and margin <= y <= self.LY - margin)]
        if bad:
            lines = "\n".join(f"    monopole {i}: (x={x:.4f}, y={y:.4f}) m" for i, x, y in bad)
            raise SystemExit(
                "[config error] control_source places one or more monopoles "
                f"outside the room (0..{self.LX:.4f} m x 0..{self.LY:.4f} m):\n{lines}\n"
                "The 4 monopoles span center +/- 1.5*monopole_spacing_m along "
                "(axis_x, axis_y). Move control_source.center_x_m/center_y_m, "
                "or shrink monopole_spacing_m, then re-run.")

        # ---- Jp evaluation grid ----
        JP = cfg["jp_grid"]
        self.JP_PLANE_Z = JP["plane_z_m"]
        self.JP_N_X = JP["n_points_x"]
        self.JP_N_Y = JP["n_points_y"]
        self.JP_EXCLUSION_RADIUS_M = JP.get("exclusion_radius_m", 0.10)
        self.ROOM_VOLUME_M3 = self.LX * self.LY * self.LZ
        
        margin = 1e-4
        jp_xs = np.linspace(margin, self.LX - margin, self.JP_N_X)
        jp_ys = np.linspace(margin, self.LY - margin, self.JP_N_Y)
        if JP.get("interior_only", True) and self.JP_N_X > 2 and self.JP_N_Y > 2:
            jp_xs = jp_xs[1:-1]
            jp_ys = jp_ys[1:-1]
        XX, YY = np.meshgrid(jp_xs, jp_ys, indexing="xy")
        candidate_xy = np.column_stack([XX.ravel(), YY.ravel()])

        # Drop any grid point too close to the primary source or any of the control monopoles
        exclude_centers = [(self.SRC_X, self.SRC_Y)] + [tuple(p) for p in self.CTRL_MONOPOLE_XY]
        keep = np.ones(candidate_xy.shape[0], dtype=bool)
        for cx, cy in exclude_centers:
            keep &= (np.hypot(candidate_xy[:, 0] - cx, candidate_xy[:, 1] - cy)
                      > self.JP_EXCLUSION_RADIUS_M)
        n_excluded = int((~keep).sum())
        self.jp_grid_xy = candidate_xy[keep]
        print(f"[jp] grid: {candidate_xy.shape[0]} interior points (paper-consistent "
              f"22x12 edge grid with the boundary ring dropped, {self.JP_N_X}x{self.JP_N_Y} "
              "before interior-trim)")
        if n_excluded:
            print(f"[jp] excluded {n_excluded}/{candidate_xy.shape[0]} Jp grid point(s) "
                  f"within {self.JP_EXCLUSION_RADIUS_M:.3f} m of the primary source or a "
                  f"control monopole -- {self.jp_grid_xy.shape[0]} points remain")

        self.table_freqs = np.linspace(self.FREQ_MIN, self.FREQ_MAX, N_TABLE)

        print(f"[control source] longitudinal quadrupole: 4 monopoles at (x,y) = "
              f"{[tuple(np.round(p, 4)) for p in self.CTRL_MONOPOLE_XY]}, "
              f"z={self.CTRL_Z} m, signs={MONOPOLE_SIGNS.tolist()}, "
              f"dipole pairs={DIPOLE_PAIRS}")

        # ---- launch mapdl and build ----
        run_base = mapdl_run_base or os.environ.get("MAPDL_RUN_BASE", os.getcwd())
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        self._mapdl_log_dir = os.path.join(run_base, "mapdl_run", f"job_{job_id}")
        os.makedirs(self._mapdl_log_dir, exist_ok=True)

        self.mapdl = launch_mapdl(timeout=timeout, run_location=self._mapdl_log_dir)
        self.mapdl.clear()
        self.mapdl.filname(self.jobname)
        self.mapdl.prep7()
        self.mapdl.units("SI")
        self.mapdl.mp("DENS", 1, self.RHO_AIR)
        self.mapdl.mp("SONC", 1, self.C0)
        self.mapdl.et(1, "FLUID221", kop2=1)

        self._build_geometry_and_hardpoints()
        self._mesh()
        self._resolve_nodes()
        self._apply_wall_absorption()
        self._build_freq_tables()

    # -- geometry -------------------------------------------------------

    def _build_geometry_and_hardpoints(self):
        mapdl = self.mapdl
        LX, LY, LZ = self.LX, self.LY, self.LZ

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
        mapdl.va(front_wall, back_wall, floor, ceiling, left_wall, right_wall)

        SLICE_TOL = 1e-6
        raw_heights = ([self.SRC_Z, self.LISTENER_Z, self.CTRL_Z]
                        + list(self.PLANE_HEIGHTS_Z))
        interior = [z for z in raw_heights if SLICE_TOL < z < LZ - SLICE_TOL]
        slice_heights = []
        for z in sorted(interior):
            if not slice_heights or abs(z - slice_heights[-1]) > SLICE_TOL:
                slice_heights.append(z)
        self.slice_heights = slice_heights
        if slice_heights:
            print(f"[geometry] slicing the room at z = {slice_heights} m "
                  "(source + listener + control-source hard points + plotting planes)")

        wp_z = 0.0
        for z in slice_heights:
            mapdl.allsel()
            mapdl.wpoffs(0, 0, z - wp_z)
            mapdl.vsbw("ALL", "", "DELETE")
            wp_z = z
        mapdl.wpoffs(0, 0, -wp_z)
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
                    f"{slice_heights} m, found {n_vols} -- check each height "
                    "individually with ASEL,S,LOC,Z,<height> in APDL.")

        self.src_kp = self._make_hardpoint_on_zplane(
            self.SRC_X, self.SRC_Y, self.SRC_Z, "source", require_unique=bool(slice_heights))
        self.listener_kp = self._make_hardpoint_on_zplane(
            self.LISTENER_X, self.LISTENER_Y, self.LISTENER_Z, "listener",
            require_unique=bool(slice_heights))

        mapdl.allsel()
        mapdl.asel("S", "LOC", "Z", self.CTRL_Z)
        n_ctrl_slice_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))
        print(f"[geometry] found {n_ctrl_slice_areas} slice area(s) at "
              f"z={self.CTRL_Z} m (control-source plane)")
        if n_ctrl_slice_areas != 1:
            raise RuntimeError(
                f"expected exactly 1 interior area at z=CTRL_Z ({self.CTRL_Z} m), "
                f"found {n_ctrl_slice_areas} -- check control_source.z_m against "
                "source/listener/plotting heights for near-duplicate values.")
        ctrl_slice_area = int(mapdl.get_value(entity="area", entnum=0, item1="num", it1num="max"))

        self.ctrl_kps = []
        for i, (mx, my) in enumerate(self.CTRL_MONOPOLE_XY):
            mapdl.hptcreate("AREA", ctrl_slice_area, "", "COORD", mx, my, self.CTRL_Z)
            kp = int(mapdl.get_value(entity="kp", entnum=0, item1="num", it1num="max"))
            hx = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="x")
            hy = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="y")
            hz = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="z")
            if abs(hz - self.CTRL_Z) > 1e-9 or abs(hx - mx) > 1e-9 or abs(hy - my) > 1e-9:
                raise RuntimeError(
                    f"control monopole {i} hard point KP={kp} landed at "
                    f"({hx},{hy},{hz}), not ({mx},{my},{self.CTRL_Z}).")
            self.ctrl_kps.append(kp)
            print(f"[geometry] control monopole {i} hard point KP={kp} confirmed at "
                  f"({hx:.4f},{hy:.4f},{hz:.4f}) m (sign={MONOPOLE_SIGNS[i]:+.0f})")
        mapdl.allsel()

    def _make_hardpoint_on_zplane(self, x, y, z, label, require_unique):
        mapdl = self.mapdl
        mapdl.allsel()
        mapdl.asel("S", "LOC", "Z", z)
        n_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))
        if require_unique:
            print(f"[geometry] found {n_areas} slice area(s) at z={z} m ({label} plane)")
            if n_areas != 1:
                raise RuntimeError(
                    f"expected exactly 1 interior area at z={z} m ({label}), "
                    f"found {n_areas} -- check for near-duplicate heights.")
        area = int(mapdl.get_value(entity="area", entnum=0, item1="num", it1num="max"))
        mapdl.hptcreate("AREA", area, "", "COORD", x, y, z)
        kp = int(mapdl.get_value(entity="kp", entnum=0, item1="num", it1num="max"))
        hx = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="x")
        hy = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="y")
        hz = mapdl.get_value(entity="kp", entnum=kp, item1="loc", it1num="z")
        if abs(hz - z) > 1e-9 or abs(hx - x) > 1e-9 or abs(hy - y) > 1e-9:
            raise RuntimeError(
                f"{label} hard point KP={kp} is at ({hx}, {hy}, {hz}) m, not "
                f"({x}, {y}, {z}) m -- it did not land on the slice.")
        print(f"[geometry] {label} hard point KP={kp} confirmed at "
              f"({hx:.4f}, {hy:.4f}, {hz:.4f}) m")
        mapdl.allsel()
        return kp

    # -- mesh -------------------------------------------------------------

    def _mesh(self):
        mapdl = self.mapdl
        mapdl.mshape(1, "3D")
        mapdl.mshkey(0)
        mapdl.esize(self.ESIZE)

        # Local refinement between the control-source monopoles
        if self.ESIZE > self.D_SPACING:
            print(f"[mesh] global element size ({self.ESIZE:.4f} m) is larger than "
                  f"the control-source monopole spacing ({self.D_SPACING:.4f} m) -- "
                  f"relying on local refinement (kesize={self.CTRL_HP_ELEM_SIZE:.4f} m) "
                  "at the 4 control hard points to resolve it.")
        for kp in self.ctrl_kps:
            mapdl.kesize(kp, self.CTRL_HP_ELEM_SIZE)
        mapdl.allsel()
        mapdl.type(1)
        mapdl.mat(1)
        mapdl.vmesh("all")
        print(f"[mesh] {mapdl.mesh.n_elem} elements, {mapdl.mesh.n_node} nodes")
        mapdl.allsel()

    def _kp_to_single_node(self, kp, label):
        mapdl = self.mapdl
        mapdl.allsel()
        mapdl.ksel("S", "KP", vmin=kp)
        mapdl.nslk()
        n = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
        if n != 1:
            raise RuntimeError(
                f"expected exactly 1 node at hard point KP={kp} ({label}), found {n} "
                "-- the hard point likely wasn't meshed conformally.")
        node = int(mapdl.get_value(entity="node", entnum=0, item1="num", it1num="min"))
        mapdl.allsel()

        mapdl.nsel("S", "NODE", vmin=node)
        mapdl.esln("S", 0)
        n_elems = int(mapdl.get_value(entity="elem", entnum=0, item1="count"))
        print(f"[nodes] {label} node {node} has {n_elems} FLUID221 element(s) attached")
        if n_elems == 0:
            raise RuntimeError(
                f"{label} node {node} has no elements attached -- it exists in "
                "the database but isn't part of the mesh, so BF loads on it "
                "would have no effect on the solve.")
        mapdl.allsel()
        return node

    def _resolve_nodes(self):
        self.src_node = self._kp_to_single_node(self.src_kp, "source")
        self.listener_node = self._kp_to_single_node(self.listener_kp, "listener")
        self.ctrl_nodes = [self._kp_to_single_node(kp, f"ctrl_monopole_{i}")
                            for i, kp in enumerate(self.ctrl_kps)]
        print(f"[nodes] control monopoles -> nodes {self.ctrl_nodes}")

    # -- wall absorption ----------------------------------------------------

    def _apply_wall_absorption(self):
        mapdl = self.mapdl
        mapdl.allsel()
        if self.ALPHA_WALL > 0:
            mapdl.asel("S", "LOC", "X", 0)
            mapdl.asel("A", "LOC", "X", self.LX)
            mapdl.asel("A", "LOC", "Y", 0)
            mapdl.asel("A", "LOC", "Y", self.LY)
            mapdl.asel("A", "LOC", "Z", 0)
            mapdl.asel("A", "LOC", "Z", self.LZ)
            n_wall_areas = int(mapdl.get_value(entity="area", entnum=0, item1="count"))
            print(f"[abs] {n_wall_areas} exterior wall area(s) selected for "
                  f"absorption (alpha={self.ALPHA_WALL})")
            if n_wall_areas == 0:
                raise RuntimeError("no exterior wall areas found -- absorption BC not applied.")
            mapdl.nsla("S", 1)
            n_wall_nodes = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
            mapdl.sf("ALL", "ATTN", self.ALPHA_WALL)
            print(f"[abs] absorption coefficient alpha={self.ALPHA_WALL} applied to "
                  f"{n_wall_nodes} wall node(s)")
        else:
            print("[abs] ALPHA_WALL <= 0 -- walls left fully rigid")

        # Exclude source nodes and listener node from wall absorption
        exempt = [("source", self.src_node), ("listener", self.listener_node)]
        exempt += [(f"ctrl_monopole_{i}", n) for i, n in enumerate(self.ctrl_nodes)]
        for label, node in exempt:
            mapdl.nsel("S", "NODE", vmin=node)
            mapdl.sf("ALL", "ATTN", 0)
            print(f"[abs] {label} node {node} exempted from wall absorption (ATTN reset to 0)")
        mapdl.allsel()

    # -- frequency-dependent source tables ------------------------------

    def _build_freq_tables(self):
        mapdl = self.mapdl

        self.SRC_TABLE = "src_mass_tab"
        primary_values = self.SRC_MASS_MAGNITUDE / self.table_freqs
        mapdl.dim(self.SRC_TABLE, "TABLE", N_TABLE, 1, 1, "FREQ")
        for i, (f_i, v_i) in enumerate(zip(self.table_freqs, primary_values), start=1):
            mapdl.run(f"{self.SRC_TABLE}({i},0) = {f_i}")
            mapdl.run(f"{self.SRC_TABLE}({i},1) = {v_i}")
        print(f"[bc] frequency-dependent source table '{self.SRC_TABLE}': {N_TABLE} "
              f"points, {primary_values.min():.3e} to {primary_values.max():.3e} kg/s "
              f"over {self.FREQ_MIN:.1f}-{self.FREQ_MAX:.1f} Hz")

        self.ctrl_tab_names = []
        for i in range(4):
            name = f"ctrl_mass_tab_{i}"
            values = MONOPOLE_SIGNS[i] * self.MASS_REF / self.table_freqs
            mapdl.dim(name, "TABLE", N_TABLE, 1, 1, "FREQ")
            for k, (f_k, v_k) in enumerate(zip(self.table_freqs, values), start=1):
                mapdl.run(f"{name}({k},0) = {f_k}")
                mapdl.run(f"{name}({k},1) = {v_k}")
            self.ctrl_tab_names.append(name)
        print(f"[bc] control monopole tables {self.ctrl_tab_names} created "
              f"(unit reference magnitude {self.MASS_REF} kg/s at 1 Hz, signs "
              f"{MONOPOLE_SIGNS.tolist()}) -- not yet applied")

    # -- BF helpers -------------------------------------------------------

    def clear_all_bf(self):
        self.mapdl.finish()
        self.mapdl.prep7()
        self.mapdl.allsel()
        self.mapdl.bfdele("ALL", "ALL")

    def apply_primary_bf(self):
        self.mapdl.bf(self.src_node, "MASS", f"%{self.SRC_TABLE}%", 0.0)

    def apply_dipole_bf(self, dipole_index):
        for m in DIPOLE_PAIRS[dipole_index]:
            self.mapdl.bf(self.ctrl_nodes[m], "MASS", f"%{self.ctrl_tab_names[m]}%", 0.0)

    def apply_dipole_bf_single_freq(self, dipole_index, freq_hz, level_db=0.0, polarity=0):
        """Used for testing a single frequency"""
        scale = 10 ** (level_db / 20.0)
        sign = 1.0 if polarity == 0 else -1.0
        for m in DIPOLE_PAIRS[dipole_index]:
            value = MONOPOLE_SIGNS[m] * self.MASS_REF / freq_hz * scale * sign
            self.mapdl.bf(self.ctrl_nodes[m], "MASS", value, 0.0)

    def apply_primary_bf_single_freq(self, freq_hz):
        value = self.SRC_MASS_MAGNITUDE / freq_hz
        self.mapdl.bf(self.src_node, "MASS", value, 0.0)

    # -- harmonic solve -----------------------------------------------------

    def harmonic_solve(self, light_outres=False):
        mapdl = self.mapdl
        mapdl.run("/SOLU")
        mapdl.antype(3)
        mapdl.harfrq(freqb=self.FREQ_MIN, freqe=self.FREQ_MAX)
        mapdl.autots("off")
        mapdl.nsubst(self.N_SUBSTEPS)
        mapdl.kbc(0)
        mapdl.outres("erase")
        if light_outres:
            mapdl.outres("all", "none")
            mapdl.outres("nsol", "all")
        else:
            mapdl.outres("all", "all")
            mapdl.outres("nsol", "all")
        mapdl.solve()
        mapdl.finish()
        mapdl.post1()
        return np.unique(mapdl.post_processing.time_values)

    def harmonic_solve_single_freq(self, freq_hz, light_outres=False):
      
        mapdl = self.mapdl
        mapdl.run("/SOLU")
        mapdl.antype(3)
        mapdl.harfrq(freqb=freq_hz, freqe=freq_hz)
        mapdl.autots("off")
        mapdl.nsubst(1)
        mapdl.kbc(0)
        mapdl.outres("erase")
        if light_outres:
            mapdl.outres("all", "none")
            mapdl.outres("nsol", "all")
        else:
            mapdl.outres("all", "all")
            mapdl.outres("nsol", "all")
        mapdl.solve()
        mapdl.finish()
        mapdl.post1()
        return np.unique(mapdl.post_processing.time_values)

    # -- post-processing helpers -------------------------------------------

    def nodal_pressure_at(self, solved_freqs, target_freq):
        mapdl = self.mapdl
        f = solved_freqs[np.argmin(np.abs(solved_freqs - target_freq))]
        mapdl.allsel()
        mapdl.set(time=f, kimg=0)
        real = mapdl.post_processing.nodal_pressure()
        mapdl.set(time=f, kimg=1)
        imag = mapdl.post_processing.nodal_pressure()
        return f, real + 1j * imag

    def node_pressure_sweep(self, node, n_sets):
        mapdl = self.mapdl
        pres = np.zeros(n_sets, dtype=complex)
        for i in range(1, n_sets + 1):
            mapdl.set(lstep=1, sbstep=i, kimg=0)
            p_re = mapdl.get_value(entity="node", entnum=node, item1="pres")
            mapdl.set(lstep=1, sbstep=i, kimg=1)
            p_im = mapdl.get_value(entity="node", entnum=node, item1="pres")
            pres[i - 1] = p_re + 1j * p_im
        return pres

    def node_velocity_sweep(self, node, n_sets):
       
        mapdl = self.mapdl
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
                    ]) for c in ("x", "y", "z")
                }
            vx = comps["re"]["x"] + 1j * comps["im"]["x"]
            vy = comps["re"]["y"] + 1j * comps["im"]["y"]
            vz = comps["re"]["z"] + 1j * comps["im"]["z"]
            vel[i - 1] = [vx, vy, vz, np.sqrt(np.abs(vx) ** 2 + np.abs(vy) ** 2 + np.abs(vz) ** 2)]
        mapdl.allsel()
        return vel

    def plane_grid_pressure_sweep(self, plane_z, grid_xy, n_sets, label):
        """Complex pressure at the given XY grid on a horizontal plane at Z=plane_z, for each of n_sets"""
        from scipy.interpolate import griddata
        mapdl = self.mapdl
        mapdl.allsel()
        mapdl.nsel("S", "LOC", "Z", plane_z)
        n_pts = int(mapdl.get_value(entity="node", entnum=0, item1="count"))
        if n_pts == 0:
            raise RuntimeError(f"[{label}] no nodes found at z={plane_z} m for the Jp grid.")
        coords = mapdl.mesh.nodes
        if coords.shape[0] != n_pts:
            raise RuntimeError(f"[{label}] mesh.nodes/{n_pts}-selection mismatch at z={plane_z} m.")
        plane_xy = coords[:, :2]

        n_grid = grid_xy.shape[0]
        p_grid = np.zeros((n_sets, n_grid), dtype=complex)

        for i in range(1, n_sets + 1):
            mapdl.set(lstep=1, sbstep=i, kimg=0)
            p_re_nodes = mapdl.post_processing.nodal_pressure()
            mapdl.set(lstep=1, sbstep=i, kimg=1)
            p_im_nodes = mapdl.post_processing.nodal_pressure()
            if p_re_nodes.shape[0] != n_pts or p_im_nodes.shape[0] != n_pts:
                raise RuntimeError(f"[{label}] selection lost partway through the sweep.")
            p_re_i = griddata(plane_xy, p_re_nodes, grid_xy, method="linear")
            p_im_i = griddata(plane_xy, p_im_nodes, grid_xy, method="linear")
            p_grid[i - 1, :] = p_re_i + 1j * p_im_i

        mapdl.allsel()
        return grid_xy, p_grid

    def jp_from_grid(self, p_grid):
        n_pts = p_grid.shape[-1]
        return (self.ROOM_VOLUME_M3 / (4.0 * self.RHO_AIR * self.C0 ** 2 * n_pts)) * np.sum(
            np.abs(p_grid) ** 2, axis=-1)

    def jp_db(self, jp):
        return 10 * np.log10(np.clip(jp, 1e-30, None) / JP_REF)

    # -- teardown -----------------------------------------------------------

    def exit(self):
        self.mapdl.exit()
        import shutil
        shutil.rmtree(self._mapdl_log_dir, ignore_errors=True)