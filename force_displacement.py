# -*- coding: utf-8 -*-
"""
One force–displacement curve, with:
- optional units kept DOWN (vertical DOF frozen at y = -h0)
- displacement control on one unit (default: center)

How to tune the simulation (see __main__ at the bottom):
- constrained units (tabs / units kept DOWN) are defined here:
    constrained_down_indices = ...
- k_theta and w_beam are defined here:
    k_theta = ...
    w_beam  = ...
- filters (point removal + smoothing) are defined here:
    trim_steps   = ...
    smooth_window = ...
    smooth_poly   = ...

Author: Eléonore Duval
"""
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import lax

from scipy.signal import savgol_filter


def run_one_force_displacement(
    # geometry / material
    N,                      # number of units
    l_strut,                # strut length [m]
    d,                      # distance between beam bases [m]
    w_tab,                  # tab width parameter -> d_app = d - w_tab [m]
    E,                      # Young's modulus [Pa]
    l_beam,                 # beam length [m]
    w_beam,                 # beam width [m]
    t_out,                  # out-of-plane thickness [m]
    rho,                    # material density [kg/m^3]
    S_strut,                # [m^2] measured in CAD
    l_top,                  # pull-tab length [m]
    w_strut,                # strut area [m^2], for axial stiffness

    # springs / damping
    k_theta,                # torsional spring stiffness [N*m/rad]
    c_beam,                 # damping on beam DOF [N*s/m]
    c_top,                  # damping on top DOF [N*s/m]

    # displacement control on one unit
    controlled_unit,        # None -> center unit
    k_ctrl,                 # penalty stiffness for displacement control [N/m]
    T_total,                # total ramp time [s]
    dt,                     # time step [s]

    # initial configuration
    init_left_half_down,       # if True: left half DOWN, right half UP initially
    constrained_down_indices,  # units kept DOWN (vertical DOF frozen)

    # contact penalty parameters (torsion stiffening)
    theta_c_deg,
    contact_gain,
    contact_power,

    # filters / point removal
    trim_steps=4,
    smooth_window=51,
    smooth_poly=3,
):
    """
    Returns
    -------
    v_top_mm : np.ndarray
        imposed displacement of the controlled unit [mm]
    F_top_N : np.ndarray
        controller force (reaction) [N] vs v_top_mm
    """

    # derived  properties
    I_beam = (w_beam**3) * t_out / 12.0

    # masses computed
    m_top = rho * l_top * t_out * w_tab + rho * S_strut * t_out
    m_beam = (0.236 * w_beam * l_beam + S_strut) * t_out * rho

    # geometry helpers
    d_app = d - w_tab
    h0 = jnp.sqrt(l_strut**2 - (d_app/2)**2)

    angle_l0 = jnp.arccos((d_app/2) / l_strut)
    angle_r0 = -angle_l0

    # spring stiffnesses
    k_beam = 3.0 * E * I_beam / (l_beam**3)
    k_strut = E * w_strut * t_out / l_strut

    # controlled unit index
    if controlled_unit is None:
        controlled_unit = N // 2
    j_ctrl = int(controlled_unit)
    if j_ctrl < 0 or j_ctrl >= N:
        raise ValueError(f"controlled_unit must be in [0, {N-1}].")

    # time discretization
    t_series = jnp.arange(0.0, T_total + dt, dt)

    # displacement target (absolute y) for the controlled unit
    # We ramp from DOWN (-h0) to UP (+h0) linearly over T_total.
    def y_target(t):
        t_clamped = jnp.clip(t, 0.0, T_total)
        return -h0 + (2.0 * h0) * (t_clamped / T_total)

    # constrained mask for "kept down" units
    # True means: freeze vertical DOF y (vel_y = 0 and acc_y = 0).
    if constrained_down_indices is None:
        constrained_mask = jnp.zeros((N,), dtype=bool)
    else:
        constrained_set = set(int(ii) for ii in constrained_down_indices)
        constrained_mask = jnp.array([i in constrained_set for i in range(N)], dtype=bool)

    # initial conditions
    # Beam nodes x positions (N+1 nodes)
    x0_beams = jnp.linspace(0.0, N * d_app, N + 1)

    # Top nodes initial x positions (one per unit cell)
    x0_tops = jnp.arange(N) * d_app + d_app / 2

    # Default initial y: all UP
    y0_tops = h0 * jnp.ones((N,))

    # Optionally set left half DOWN at t=0
    if init_left_half_down:
        y0_tops = y0_tops.at[:(N // 2)].set(-h0)

    # Ensure constrained units start DOWN
    if constrained_down_indices is not None:
        for idx in constrained_down_indices:
            idx = int(idx)
            if 0 <= idx < N:
                y0_tops = y0_tops.at[idx].set(-h0)

    # Initial top positions (N,2)
    pos_t0 = jnp.stack([x0_tops, y0_tops], axis=1)

    # Initial velocities
    vel_b0 = jnp.zeros((N + 1,))
    vel_t0 = jnp.zeros((N, 2))

    # Force vertical velocity to zero for constrained units
    vel_t0 = vel_t0.at[constrained_mask, 1].set(0.0)

    # Pack into state vector y:
    # y = [pos_beams (N+1),
    #      pos_tops  (2N),
    #      vel_beams (N+1),
    #      vel_tops  (2N)]
    y0 = jnp.concatenate([x0_beams, pos_t0.reshape(-1), vel_b0, vel_t0.reshape(-1)])

    # Contact threshold in radians
    theta_c = jnp.deg2rad(theta_c_deg)

    # ODE
    def deriv(y, t):
        i1 = N + 1
        i2 = i1 + 2 * N
        i3 = i2 + (N + 1)

        pos_b = y[0:i1]                        # (N+1,) beam x positions
        pos_t = y[i1:i2].reshape((N, 2))       # (N,2) top (x,y)
        vel_b = y[i2:i3]                       # (N+1,) beam x velocities
        vel_t = y[i3:].reshape((N, 2))         # (N,2) top velocities

        # Force
        F_b = jnp.zeros_like(pos_b)            # x forces on beams
        F_t = jnp.zeros_like(pos_t)            # (x,y) forces on tops

        # Ground springs
        initial_b = jnp.linspace(0.0, N * d_app, N + 1)
        F_b = F_b - k_beam * (pos_b - initial_b)

        # Loop over cells
        for i in range(N):
            # Beam nodes at y=0
            pl = jnp.array([pos_b[i],   0.0])
            pr = jnp.array([pos_b[i+1], 0.0])
            # Top node
            pt = pos_t[i]

            # axial struts
            dl = pt - pl
            distl = jnp.linalg.norm(dl) + 1e-6
            fl = -k_strut * (distl - l_strut) * (dl / distl)

            dr = pt - pr
            distr = jnp.linalg.norm(dr) + 1e-6
            fr = -k_strut * (distr - l_strut) * (dr / distr)

            # Apply to top and beams (x reaction)
            F_t = F_t.at[i].add(fl + fr)
            F_b = F_b.at[i].add(-fl[0])
            F_b = F_b.at[i+1].add(-fr[0])

            # torsion
            anglel = jnp.arctan2(dl[0], dl[1])
            angler = jnp.arctan2(dr[0], dr[1])

            # Perpendicular unit directions
            tl_vec = jnp.array([dl[1], -dl[0]]) / distl
            tr_vec = jnp.array([dr[1], -dr[0]]) / distr

            # linear torsion spring
            taul = -k_theta * (anglel - angle_l0)
            taur = -k_theta * (angler - angle_r0)

            # Convert torque to force along perpendicular direction
            F_t = F_t.at[i].add((taul / l_strut) * tl_vec)
            F_b = F_b.at[i].add(-(taul / l_strut) * tl_vec[0])
            F_t = F_t.at[i].add((taur / l_strut) * tr_vec)
            F_b = F_b.at[i+1].add(-(taur / l_strut) * tr_vec[0])

            # contact penalty /// way faster without. Can be commented wihtout much effect. 
            dev_l = jnp.abs(anglel - angle_l0)
            dev_r = jnp.abs(angler - angle_r0)
            k_eff_l = jnp.where(
                dev_l <= theta_c,
                k_theta,
                k_theta + (contact_gain * k_theta) * (dev_l - theta_c) ** contact_power
            )
            k_eff_r = jnp.where(
                dev_r <= theta_c,
                k_theta,
                k_theta + (contact_gain * k_theta) * (dev_r - theta_c) ** contact_power
            )
            tau_add_l = -(k_eff_l - k_theta) * (anglel - angle_l0)
            tau_add_r = -(k_eff_r - k_theta) * (angler - angle_r0)
            # Convert penalty torque to forces
            Fc_l = (tau_add_l / l_strut) * tl_vec
            Fc_r = (tau_add_r / l_strut) * tr_vec
            F_t = F_t.at[i].add(Fc_l + Fc_r)
            F_b = F_b.at[i].add(-Fc_l[0])
            F_b = F_b.at[i+1].add(-Fc_r[0])

        # Damping
        F_b = F_b - c_beam * vel_b
        F_t = F_t - c_top * vel_t

        # Clamp end beams (no external force / no motion at boundaries)
        F_b = F_b.at[0].set(0.0).at[-1].set(0.0)

        # Displacement control on controlled unit vertical DOF (unless constrained)
        y_act = pos_t[j_ctrl, 1]
        y_tar = y_target(t)
        F_ctrl = -k_ctrl * (y_act - y_tar)
        F_ctrl_applied = lax.cond(
            constrained_mask[j_ctrl],
            lambda _: 0.0,         # constrained -> no controller
            lambda val: val,       # else -> apply
            operand=F_ctrl
        )
        F_t = F_t.at[j_ctrl, 1].add(F_ctrl_applied)

        # Accelerations
        a_b = F_b / m_beam
        a_t = F_t / m_top

        # Enforce beam boundary accelerations too -> to 0
        a_b = a_b.at[0].set(0.0).at[-1].set(0.0)

        # Freeze vertical DOF for constrained units:
        vel_t = vel_t.at[constrained_mask, 1].set(0.0)
        a_t   = a_t.at[constrained_mask, 1].set(0.0)

        # Return dy/dt = [velocities, accelerations]
        return jnp.concatenate([vel_b, vel_t.reshape(-1), a_b, a_t.reshape(-1)])

    # integrate
    sol = odeint(deriv, y0, t_series)

    # extract force-displacement curve
    pos_t_array = sol[:, (N+1):(N+1+2*N)].reshape((-1, N, 2))

    # actual controlled y(t) and target y(t)
    y_actual = np.array(pos_t_array[:, j_ctrl, 1])
    y_targets = np.array([float(y_target(float(ti))) for ti in np.array(t_series)])

    # controller reaction force
    F_ctrls = -k_ctrl * (y_actual - y_targets)

    # Convert to imposed displacement v_top (mm)
    h0_float = float(h0)
    v_top_mm = (y_targets - h0_float) * 1e3
    F_top_N = F_ctrls

    # -------------------- filters + point removal --------------------
    # 1) remove non-finite points (safety)
    mask = np.isfinite(v_top_mm) & np.isfinite(F_top_N)
    v_top_mm = v_top_mm[mask]
    F_top_N = F_top_N[mask]

    # 2) remove first points (transient)
    if trim_steps is not None and trim_steps > 0 and len(v_top_mm) > trim_steps:
        v_top_mm = v_top_mm[trim_steps:]
        F_top_N = F_top_N[trim_steps:]

    # 3) smooth (Savitzky–Golay)
    # window must be odd and <= len(signal)
    if smooth_window is not None and smooth_window > 2 and len(F_top_N) >= smooth_window:
        if smooth_window % 2 == 0:
            smooth_window = smooth_window + 1
        if smooth_window <= len(F_top_N):
            F_top_N = savgol_filter(F_top_N, smooth_window, smooth_poly)

    return v_top_mm, F_top_N, float(m_beam), float(m_top), float(I_beam)


# to run
if __name__ == "__main__":

    # set of values
    N = 9
    k_theta = 1e-3
    w_beam = 2.8e-3
    l_beam = 15e-3
    t_out = 1e-2
    w_tab = 6e-3
    S_strut = 27.55e-6

    constrained_down_indices = list(range(0, N//2))  # keep left half down
    # if all units down
    # constrained_down_indices = list(range(N))
    # constrained_down_indices.remove(N//2)
    # if all units up
    # constrained_down_indices = None

    # filters / point removal (publication-friendly defaults)
    trim_steps = 4
    smooth_window = 51
    smooth_poly = 3

    # aditionnal parameter set
    geom = dict(
        N=N,
        l_strut=11e-3,
        d=18.4e-3,
        w_tab=w_tab,
        l_beam=l_beam,
        w_beam=w_beam,
        t_out=t_out,
        S_strut=S_strut,
        l_top=1.5e-2,
        w_strut=3e-3,
    )
    material = dict(E=1.3e6, rho=1200.0)
    model = dict(k_theta=k_theta, c_beam=10.0, c_top=10.0)
    control = dict(controlled_unit=None, k_ctrl=1e6, T_total=10.0, dt=0.01)
    init = dict(init_left_half_down=True, constrained_down_indices=constrained_down_indices)
    contact = dict(theta_c_deg=108.0, contact_gain=1e3, contact_power=2.0)
    filt = dict(trim_steps=trim_steps, smooth_window=smooth_window, smooth_poly=smooth_poly)

    v_mm, F_N, m_beam, m_top, I_beam = run_one_force_displacement(
        **geom, **material, **model, **control, **init, **contact, **filt
    )

    plt.figure(figsize=(6, 4))
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.5)
    plt.plot(v_mm, F_N, "k-", linewidth=2)
    plt.xlabel(r"$v_{\mathrm{top}}$ [mm]")
    plt.ylabel(r"$F_{\mathrm{top}}$ [N]")

    # show indices of frozen units in the title
    if constrained_down_indices:
        frozen_str = ", ".join(str(i) for i in constrained_down_indices)
    else:
        frozen_str = "none"

    plt.title(
        fr"$k_{{\theta}}={k_theta:.2e}$  N.m/rad, $w_{{beam}}={w_beam*1e3:.2f}$ mm"
        + fr", down = [{frozen_str}]"
    )
    plt.tight_layout()
    plt.show()