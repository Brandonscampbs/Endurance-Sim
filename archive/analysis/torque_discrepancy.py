"""Torque discrepancy analysis: sim input vs real motor delivery.

Compares LVCU Torque Req (what ReplayStrategy feeds to the sim) against
Torque Feedback (what the motor actually delivered) and Torque Command
(what the inverter commanded internally).

Goal: quantify how much extra torque the sim applies by using the
request value instead of the actual delivered value.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from fsae_sim.data.loader import load_cleaned_csv


def main():
    _, df = load_cleaned_csv("Real-Car-Data-And-Stats/CleanedEndurance.csv")

    # ----------------------------------------------------------------
    # 1. Filter to moving samples (LFspeed > 5 km/h)
    # ----------------------------------------------------------------
    moving = df[df["LFspeed"] > 5.0].copy()
    print(f"Total samples: {len(df)}")
    print(f"Moving samples (LFspeed > 5 km/h): {len(moving)}")
    print()

    lvcu_req = moving["LVCU Torque Req"].values
    tf_raw   = moving["Torque Feedback"].values
    tc       = moving["Torque Command"].values
    rpm      = moving["Motor RPM"].values
    time     = moving["Time"].values

    # ----------------------------------------------------------------
    # 2. Detect and filter CAN error / ramp-up artifacts in Torque Feedback
    #    Values > 150 Nm are physically impossible (motor peak 230 Nm but
    #    inverter limits to 85 Nm, LVCU to 150 Nm). The ramp pattern
    #    (multiples of ~54) looks like CAN initialization.
    # ----------------------------------------------------------------
    FEEDBACK_SANITY_LIMIT = 150.0
    bad_mask = tf_raw > FEEDBACK_SANITY_LIMIT
    n_bad = np.sum(bad_mask)
    print(f"Torque Feedback samples > {FEEDBACK_SANITY_LIMIT} Nm (CAN errors): {n_bad}")
    if n_bad > 0:
        print(f"  Values: {tf_raw[bad_mask][:10]}...")
        print(f"  These will be excluded from analysis.")
    print()

    # Clean mask: exclude CAN errors
    clean = ~bad_mask
    lvcu_req_c = lvcu_req[clean]
    tf_c       = tf_raw[clean]
    tc_c       = tc[clean]
    rpm_c      = rpm[clean]
    time_c     = time[clean]
    n_clean    = len(lvcu_req_c)
    print(f"Clean moving samples for analysis: {n_clean}")
    print()

    # ----------------------------------------------------------------
    # 3. Side-by-side statistics
    # ----------------------------------------------------------------
    print("=" * 70)
    print("COLUMN STATISTICS (clean moving samples)")
    print("=" * 70)
    for name, arr in [("LVCU Torque Req", lvcu_req_c),
                      ("Torque Feedback", tf_c),
                      ("Torque Command", tc_c),
                      ("Motor RPM", rpm_c)]:
        print(f"  {name:20s}: mean={np.mean(arr):7.2f}  "
              f"median={np.median(arr):7.2f}  "
              f"max={np.max(arr):7.2f}  "
              f"std={np.std(arr):7.2f}")
    print()

    # ----------------------------------------------------------------
    # 4. Difference: LVCU Torque Req vs Torque Feedback
    # ----------------------------------------------------------------
    diff = lvcu_req_c - tf_c  # positive = sim over-torques

    print("=" * 70)
    print("LVCU Torque Req - Torque Feedback (positive = sim over-torques)")
    print("=" * 70)
    print(f"  Mean difference:   {np.mean(diff):+.3f} Nm")
    print(f"  Median difference: {np.median(diff):+.3f} Nm")
    print(f"  Max difference:    {np.max(diff):+.3f} Nm")
    print(f"  Min difference:    {np.min(diff):+.3f} Nm")
    print(f"  Std deviation:     {np.std(diff):.3f} Nm")
    print()

    # Only look at powered samples (request > 0)
    powered = lvcu_req_c > 0
    diff_powered = diff[powered]
    print(f"  Powered samples (Req > 0): {np.sum(powered)}")
    print(f"  Mean diff (powered only):  {np.mean(diff_powered):+.3f} Nm")
    print(f"  Median diff (powered):     {np.median(diff_powered):+.3f} Nm")
    print()

    # ----------------------------------------------------------------
    # 5. Fraction where motor couldn't deliver what was requested
    # ----------------------------------------------------------------
    under_delivery = tf_c < lvcu_req_c
    over_delivery  = tf_c > lvcu_req_c
    exact_match    = tf_c == lvcu_req_c

    print("=" * 70)
    print("DELIVERY ANALYSIS")
    print("=" * 70)
    print(f"  Feedback < Request (under-delivery): {np.sum(under_delivery):6d} "
          f"({100*np.mean(under_delivery):.1f}%)")
    print(f"  Feedback > Request (over-delivery):  {np.sum(over_delivery):6d} "
          f"({100*np.mean(over_delivery):.1f}%)")
    print(f"  Feedback == Request (exact match):   {np.sum(exact_match):6d} "
          f"({100*np.mean(exact_match):.1f}%)")
    print()

    # Among powered samples only
    under_pow = (tf_c[powered] < lvcu_req_c[powered])
    print(f"  Under-delivery (powered only):       {np.sum(under_pow):6d} "
          f"({100*np.mean(under_pow):.1f}%)")
    print()

    # ----------------------------------------------------------------
    # 6. RPM-binned analysis: where does the gap occur?
    # ----------------------------------------------------------------
    print("=" * 70)
    print("RPM-BINNED TORQUE GAP (LVCU Req - Feedback)")
    print("=" * 70)
    rpm_bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000),
                (2000, 2400), (2400, 2600), (2600, 2800), (2800, 2950)]
    print(f"  {'RPM range':>15s}  {'Count':>6s}  {'Mean gap':>9s}  "
          f"{'Median gap':>10s}  {'Max gap':>8s}  {'Pct under':>9s}")
    for lo, hi in rpm_bins:
        mask = (rpm_c >= lo) & (rpm_c < hi) & powered
        if np.sum(mask) == 0:
            continue
        d = diff[mask]
        under_pct = 100 * np.mean(tf_c[mask] < lvcu_req_c[mask])
        print(f"  {lo:5d}-{hi:4d} RPM  {np.sum(mask):6d}  "
              f"{np.mean(d):+9.2f}  {np.median(d):+10.2f}  "
              f"{np.max(d):+8.2f}  {under_pct:8.1f}%")
    print()

    # ----------------------------------------------------------------
    # 7. Total excess torque over the entire endurance
    # ----------------------------------------------------------------
    # The sim uses LVCU Torque Req (clamped to 85), the motor delivered
    # Torque Feedback. Compute cumulative torque-time integral difference.
    print("=" * 70)
    print("TOTAL EXCESS TORQUE (sim vs reality)")
    print("=" * 70)

    # Sim uses min(LVCU Torque Req, 85) * torque_delivery_factor(RPM)
    # Let's compute what the sim feeds vs what the motor actually did.
    SIM_CLAMP = 85.0
    sim_input = np.minimum(lvcu_req_c, SIM_CLAMP)

    # Apply sim's torque_delivery_factor
    FW_ONSET = 2800.0
    FW_SLOPE = 0.75 / 250.0
    delivery_factor = np.ones_like(rpm_c)
    above_onset = rpm_c > FW_ONSET
    delivery_factor[above_onset] = np.maximum(
        0.0, 1.0 - (rpm_c[above_onset] - FW_ONSET) * FW_SLOPE)
    sim_effective_torque = sim_input * delivery_factor

    # Time intervals (for integration)
    dt = np.diff(time_c, prepend=time_c[0])
    dt = np.maximum(dt, 0)  # no negative intervals

    # Torque-time integrals (Nm*s)
    integral_sim = np.sum(sim_effective_torque * dt)
    integral_real = np.sum(tf_c * dt)
    integral_req = np.sum(lvcu_req_c * dt)
    integral_excess = integral_sim - integral_real

    print(f"  Sim effective torque integral:  {integral_sim:12.1f} Nm*s")
    print(f"  Real Torque Feedback integral:  {integral_real:12.1f} Nm*s")
    print(f"  Raw LVCU Torque Req integral:   {integral_req:12.1f} Nm*s")
    print(f"  Excess (sim - real):            {integral_excess:+12.1f} Nm*s")
    print(f"  Excess as % of real:            {100*integral_excess/integral_real:+.2f}%")
    print()

    # Convert to approximate energy using average RPM
    avg_rpm_powered = np.mean(rpm_c[powered])
    avg_omega = avg_rpm_powered * np.pi / 30.0
    excess_energy_j = integral_excess * avg_omega  # very rough (T*omega ~ P, integrate P*dt ~ E)
    # More accurate: integrate torque*omega*dt at each sample
    excess_power = (sim_effective_torque - tf_c) * rpm_c * np.pi / 30.0
    excess_energy_accurate_j = np.sum(excess_power * dt)
    excess_energy_kwh = excess_energy_accurate_j / 3.6e6

    print(f"  Excess energy (accurate):       {excess_energy_accurate_j/1000:+.1f} kJ  "
          f"({excess_energy_kwh*1000:+.1f} Wh)")
    print()

    # ----------------------------------------------------------------
    # 8. The 85 Nm clamp analysis
    # ----------------------------------------------------------------
    print("=" * 70)
    print("85 Nm CLAMP ANALYSIS")
    print("=" * 70)
    above_85 = lvcu_req_c > 85.0
    n_above = np.sum(above_85)
    print(f"  Samples with LVCU Torque Req > 85 Nm: {n_above} "
          f"({100*n_above/n_clean:.2f}%)")
    if n_above > 0:
        print(f"  LVCU Torque Req at those points:")
        print(f"    mean={np.mean(lvcu_req_c[above_85]):.1f}, "
              f"max={np.max(lvcu_req_c[above_85]):.1f}")
        print(f"  Torque Feedback at those points:")
        print(f"    mean={np.mean(tf_c[above_85]):.1f}, "
              f"max={np.max(tf_c[above_85]):.1f}")
        print(f"  Torque Command at those points:")
        print(f"    mean={np.mean(tc_c[above_85]):.1f}, "
              f"max={np.max(tc_c[above_85]):.1f}")
        print(f"  Motor RPM at those points:")
        print(f"    mean={np.mean(rpm_c[above_85]):.0f}, "
              f"min={np.min(rpm_c[above_85]):.0f}, "
              f"max={np.max(rpm_c[above_85]):.0f}")
        print()
        # What the sim does: clamp to 85, then apply delivery factor
        sim_at_85 = np.minimum(lvcu_req_c[above_85], SIM_CLAMP) * delivery_factor[above_85]
        print(f"  Sim effective torque at those points: mean={np.mean(sim_at_85):.1f}")
        print(f"  Real Torque Feedback at those points: mean={np.mean(tf_c[above_85]):.1f}")
        print(f"  Gap (sim - real):                     mean={np.mean(sim_at_85 - tf_c[above_85]):+.1f}")
    print()

    # ----------------------------------------------------------------
    # 9. Torque Command vs Torque Feedback (inverter perspective)
    # ----------------------------------------------------------------
    print("=" * 70)
    print("TORQUE COMMAND vs TORQUE FEEDBACK (inverter internal)")
    print("=" * 70)
    tc_diff = tc_c - tf_c
    print(f"  Mean (Cmd - Fbk):    {np.mean(tc_diff):+.3f} Nm")
    print(f"  Median (Cmd - Fbk):  {np.median(tc_diff):+.3f} Nm")
    print(f"  These should be close if the inverter control loop is working.")
    print()

    # ----------------------------------------------------------------
    # 10. LVCU Torque Req vs Torque Command (LVCU vs inverter)
    # ----------------------------------------------------------------
    print("=" * 70)
    print("LVCU TORQUE REQ vs TORQUE COMMAND (request vs inverter interpretation)")
    print("=" * 70)
    req_cmd_diff = lvcu_req_c - tc_c
    print(f"  Mean (Req - Cmd):    {np.mean(req_cmd_diff):+.3f} Nm")
    print(f"  Median (Req - Cmd):  {np.median(req_cmd_diff):+.3f} Nm")
    print(f"  Samples where Cmd < Req: {np.sum(tc_c < lvcu_req_c)} "
          f"({100*np.mean(tc_c < lvcu_req_c):.1f}%)")
    print(f"  -> The inverter ALREADY reduces the request before commanding the motor")
    print()

    # ----------------------------------------------------------------
    # 11. Summary: What should the sim use?
    # ----------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  The sim feeds LVCU Torque Req (clamped to 85 Nm) as motor torque.")
    print(f"  The real motor delivered Torque Feedback, which is LESS on average.")
    print(f"")
    print(f"  Mean sim effective torque: {np.mean(sim_effective_torque):.2f} Nm")
    print(f"  Mean real Torque Feedback: {np.mean(tf_c):.2f} Nm")
    print(f"  Mean Torque Command:       {np.mean(tc_c):.2f} Nm")
    print(f"  Sim over-torques by:       {np.mean(sim_effective_torque) - np.mean(tf_c):+.2f} Nm "
          f"({100*(np.mean(sim_effective_torque) - np.mean(tf_c))/max(np.mean(tf_c), 0.01):+.1f}%)")
    print()

    # Time impact estimate
    # Extra force = extra torque * gear_ratio * gearbox_eff / tire_radius
    GEAR = 3.6363
    GB_EFF = 0.97
    TIRE_R = 0.2042
    MASS = 288.0
    extra_torque_mean = np.mean(sim_effective_torque) - np.mean(tf_c)
    extra_force = extra_torque_mean * GEAR * GB_EFF / TIRE_R
    extra_accel = extra_force / MASS
    total_time = time_c[-1] - time_c[0]
    powered_time = np.sum(dt[powered])
    print(f"  Extra tractive force:  {extra_force:+.1f} N")
    print(f"  Extra acceleration:    {extra_accel:+.4f} m/s^2")
    print(f"  Powered time:          {powered_time:.0f} s")
    print(f"  Total driving time:    {total_time:.0f} s")
    print(f"  (Detailed time impact requires full sim, but this excess")
    print(f"   acceleration applied continuously would compound significantly)")


if __name__ == "__main__":
    main()
