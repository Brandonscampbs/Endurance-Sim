"""Deep dive into the resistance model to diagnose 11.4% time error.

Computes each resistance component at typical FSAE speeds, estimates
their contribution to the lap time error, and performs sensitivity analysis.
"""

import sys
import math
import os

# Ensure project root is on the path
PROJECT_ROOT = r"C:\Users\brand\Development-BC"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
os.chdir(PROJECT_ROOT)

import numpy as np
import yaml

# =====================================================================
# 1. Load configuration
# =====================================================================
print("=" * 80)
print("RESISTANCE MODEL DEEP DIVE")
print("Diagnosing 11.4% time error (~8.4 s/lap over ~74 s nominal)")
print("=" * 80)

with open(os.path.join(PROJECT_ROOT, "configs", "ct16ev.yaml")) as f:
    config = yaml.safe_load(f)

vp = config["vehicle"]
mass_kg = vp["mass_kg"]
frontal_area = vp["frontal_area_m2"]
drag_coeff = vp["drag_coefficient"]
rolling_res = vp["rolling_resistance"]
downforce_coeff = vp["downforce_coefficient"]
wheelbase = vp["wheelbase_m"]

print("\n--- Configuration Parameters ---")
print(f"  Mass (with driver):       {mass_kg} kg")
print(f"  Frontal area:             {frontal_area} m^2")
print(f"  Drag coefficient (Cd):    {drag_coeff}")
print(f"  CdA (Cd * A):             {drag_coeff * frontal_area:.3f} m^2")
print(f"  Rolling resistance coeff: {rolling_res}")
print(f"  Downforce coeff (ClA):    {downforce_coeff} m^2")
print(f"  Wheelbase:                {wheelbase} m")
print(f"  Rotor inertia:            {vp.get('rotor_inertia_kg_m2', 0.06)} kg*m^2")
print(f"  Wheel inertia:            {vp.get('wheel_inertia_kg_m2', 0.3)} kg*m^2")

# Check: CdA encoding
# The YAML has drag_coefficient = 1.502 and frontal_area = 1.0
# So the code computes: 0.5 * rho * drag_coeff * frontal_area * v^2
#                      = 0.5 * 1.225 * 1.502 * 1.0 * v^2
# CdA_effective = 1.502 * 1.0 = 1.502 m^2
# DSS says CdA = 1.50 m^2 (from 431 N at 80 kph)
# Let's verify: F = 0.5 * 1.225 * 1.502 * (80/3.6)^2 = ?
v80 = 80.0 / 3.6
f_drag_80 = 0.5 * 1.225 * 1.502 * 1.0 * v80**2
print(f"\n--- CdA Verification ---")
print(f"  Drag at 80 kph:  {f_drag_80:.1f} N (DSS says 431 N)")
print(f"  CdA effective:   {drag_coeff * frontal_area:.3f} m^2 (DSS: 1.50)")

# Similarly verify downforce
f_down_80 = 0.5 * 1.225 * downforce_coeff * v80**2
print(f"  Downforce at 80: {f_down_80:.1f} N (DSS says 625 N)")

# =====================================================================
# 2. Compute resistance forces at typical speeds
# =====================================================================
print("\n" + "=" * 80)
print("2. RESISTANCE FORCES AT TYPICAL FSAE SPEEDS")
print("=" * 80)

rho = 1.225  # air density
g = 9.81

speeds_kmh = [10, 20, 30, 40, 50, 60]
speeds_ms = [v / 3.6 for v in speeds_kmh]

# Typical FSAE curvatures (1/radius)
# Tight hairpin: r ~ 5m, kappa = 0.2 /m
# Medium corner: r ~ 15m, kappa = 0.067 /m
# Fast sweeper:  r ~ 30m, kappa = 0.033 /m
# Straight:      kappa = 0
curvatures = {
    "Straight (r=inf)": 0.0,
    "Fast sweeper (r=30m)": 1.0/30.0,
    "Medium corner (r=15m)": 1.0/15.0,
    "Tight hairpin (r=5m)": 1.0/5.0,
}

def aero_drag(v_ms):
    """F_drag = 0.5 * rho * CdA * v^2"""
    return 0.5 * rho * drag_coeff * frontal_area * v_ms**2

def downforce(v_ms):
    """F_down = 0.5 * rho * ClA * v^2"""
    return 0.5 * rho * downforce_coeff * v_ms**2

def rolling_resistance_force(v_ms):
    """F_rr = (m*g + downforce) * Crr"""
    normal = mass_kg * g + downforce(v_ms)
    return normal * rolling_res

def cornering_drag_analytical(v_ms, curvature):
    """Analytical cornering drag: F_lat^2 / C_alpha_total"""
    if abs(curvature) < 1e-6 or v_ms < 0.5:
        return 0.0
    f_lat = mass_kg * v_ms**2 * abs(curvature)
    mu_peak = 1.5
    alpha_peak = 0.15  # rad
    c_alpha_total = mass_kg * g * mu_peak / alpha_peak
    return f_lat**2 / c_alpha_total

print(f"\n{'Speed':>8} | {'Aero Drag':>10} | {'Roll Res':>10} | {'Corn Drag':>12} {'(r=15m)':>8} | {'Total':>10}")
print(f"{'(km/h)':>8} | {'(N)':>10} | {'(N)':>10} | {'(N)':>12} {'':>8} | {'(N)':>10}")
print("-" * 75)

for v_kmh, v_ms in zip(speeds_kmh, speeds_ms):
    fd = aero_drag(v_ms)
    frr = rolling_resistance_force(v_ms)
    fc = cornering_drag_analytical(v_ms, 1.0/15.0)
    total = fd + frr + fc
    print(f"{v_kmh:8.0f} | {fd:10.2f} | {frr:10.2f} | {fc:12.2f} {'':>8} | {total:10.2f}")

# Full table with all curvatures
print("\n\n--- Full Cornering Drag Table (N) ---")
header = f"{'Speed':>8} |"
for name in curvatures:
    header += f" {name:>22} |"
print(header)
print("-" * len(header))

for v_kmh, v_ms in zip(speeds_kmh, speeds_ms):
    row = f"{v_kmh:8.0f} |"
    for name, kappa in curvatures.items():
        fc = cornering_drag_analytical(v_ms, kappa)
        row += f" {fc:22.2f} |"
    print(row)

# =====================================================================
# 3. Load track and analyze curvature distribution
# =====================================================================
print("\n" + "=" * 80)
print("3. TRACK CURVATURE ANALYSIS")
print("=" * 80)

track = None
curvatures_arr = None
curved_frac = None

try:
    from fsae_sim.track.track import Track
    from fsae_sim.data.loader import load_aim_csv, load_cleaned_csv

    # Try cleaned CSV first, then raw AiM CSV
    cleaned_csv = os.path.join(PROJECT_ROOT, "Real-Car-Data-And-Stats", "CleanedEndurance.csv")
    aim_csv = os.path.join(PROJECT_ROOT, "Real-Car-Data-And-Stats", "2025 Endurance Data.csv")

    if os.path.exists(cleaned_csv):
        print(f"  Loading cleaned CSV: {cleaned_csv}")
        _, df_telem = load_cleaned_csv(cleaned_csv)
        track = Track.from_telemetry(df=df_telem)
    elif os.path.exists(aim_csv):
        track = Track.from_telemetry(aim_csv)
    else:
        track = None

    if track is not None:
        print(f"  Track: {track.name}")
        print(f"  Segments: {track.num_segments}")
        print(f"  Lap distance: {track.total_distance_m:.1f} m")

        curvatures_arr = np.array([s.curvature for s in track.segments])
        abs_curv = np.abs(curvatures_arr)
        lengths = np.array([s.length_m for s in track.segments])

        # Classify segments
        straight_mask = abs_curv < 0.005  # radius > 200m
        gentle_mask = (abs_curv >= 0.005) & (abs_curv < 0.02)  # r 50-200m
        medium_mask = (abs_curv >= 0.02) & (abs_curv < 0.05)   # r 20-50m
        tight_mask = (abs_curv >= 0.05) & (abs_curv < 0.1)     # r 10-20m
        hairpin_mask = abs_curv >= 0.1                          # r < 10m

        total_len = lengths.sum()
        print(f"\n  Curvature Distribution:")
        print(f"    Straight (r>200m):     {lengths[straight_mask].sum():7.1f} m ({100*lengths[straight_mask].sum()/total_len:.1f}%)")
        print(f"    Gentle (r=50-200m):    {lengths[gentle_mask].sum():7.1f} m ({100*lengths[gentle_mask].sum()/total_len:.1f}%)")
        print(f"    Medium (r=20-50m):     {lengths[medium_mask].sum():7.1f} m ({100*lengths[medium_mask].sum()/total_len:.1f}%)")
        print(f"    Tight (r=10-20m):      {lengths[tight_mask].sum():7.1f} m ({100*lengths[tight_mask].sum()/total_len:.1f}%)")
        print(f"    Hairpin (r<10m):       {lengths[hairpin_mask].sum():7.1f} m ({100*lengths[hairpin_mask].sum()/total_len:.1f}%)")

        print(f"\n  Curvature Statistics:")
        print(f"    Mean |curvature|:      {np.mean(abs_curv):.4f} 1/m (mean radius {1.0/max(np.mean(abs_curv), 1e-6):.1f} m)")
        print(f"    Median |curvature|:    {np.median(abs_curv):.4f} 1/m")
        print(f"    Max |curvature|:       {np.max(abs_curv):.4f} 1/m (min radius {1.0/max(np.max(abs_curv), 1e-6):.1f} m)")
        print(f"    75th percentile:       {np.percentile(abs_curv, 75):.4f} 1/m")
        print(f"    90th percentile:       {np.percentile(abs_curv, 90):.4f} 1/m")

        # Distance-weighted mean curvature
        weighted_mean_curv = np.average(abs_curv, weights=lengths)
        print(f"    Dist-weighted mean:    {weighted_mean_curv:.4f} 1/m (equiv radius {1.0/max(weighted_mean_curv, 1e-6):.1f} m)")

        # Fraction of track that is curved
        curved_frac = lengths[~straight_mask].sum() / total_len
        print(f"    Fraction curved:       {100*curved_frac:.1f}%")

    else:
        print("  [WARN] No telemetry CSV found, skipping track analysis")
        track = None
        curvatures_arr = None
        curved_frac = None
except Exception as e:
    print(f"  [ERROR] Track loading failed: {e}")
    import traceback
    traceback.print_exc()
    track = None
    curvatures_arr = None
    curved_frac = None

# =====================================================================
# 4. Estimate cornering drag contribution over a full lap
# =====================================================================
print("\n" + "=" * 80)
print("4. CORNERING DRAG CONTRIBUTION OVER A FULL LAP")
print("=" * 80)

if track is not None:
    # Assume average speeds by segment type for a rough estimate
    # (real sim has varying speeds, but this gives the order of magnitude)
    avg_speed_estimates = {
        "straight": 50.0 / 3.6,   # ~14 m/s = 50 km/h
        "gentle":   40.0 / 3.6,   # ~11 m/s
        "medium":   30.0 / 3.6,   # ~8.3 m/s
        "tight":    20.0 / 3.6,   # ~5.6 m/s
        "hairpin":  15.0 / 3.6,   # ~4.2 m/s
    }

    total_aero_work = 0.0
    total_rr_work = 0.0
    total_cd_work = 0.0
    total_time_estimate = 0.0

    for seg in track.segments:
        kappa = abs(seg.curvature)
        if kappa < 0.005:
            v_est = avg_speed_estimates["straight"]
        elif kappa < 0.02:
            v_est = avg_speed_estimates["gentle"]
        elif kappa < 0.05:
            v_est = avg_speed_estimates["medium"]
        elif kappa < 0.1:
            v_est = avg_speed_estimates["tight"]
        else:
            v_est = avg_speed_estimates["hairpin"]

        fd = aero_drag(v_est)
        frr = rolling_resistance_force(v_est)
        fc = cornering_drag_analytical(v_est, seg.curvature)

        total_aero_work += fd * seg.length_m
        total_rr_work += frr * seg.length_m
        total_cd_work += fc * seg.length_m
        total_time_estimate += seg.length_m / max(v_est, 0.5)

    total_resistance_work = total_aero_work + total_rr_work + total_cd_work

    print(f"  Estimated lap time:     {total_time_estimate:.1f} s")
    print(f"\n  Energy (Work) per lap:")
    print(f"    Aero drag:            {total_aero_work/1000:.2f} kJ  ({100*total_aero_work/total_resistance_work:.1f}%)")
    print(f"    Rolling resistance:   {total_rr_work/1000:.2f} kJ  ({100*total_rr_work/total_resistance_work:.1f}%)")
    print(f"    Cornering drag:       {total_cd_work/1000:.2f} kJ  ({100*total_cd_work/total_resistance_work:.1f}%)")
    print(f"    TOTAL resistance:     {total_resistance_work/1000:.2f} kJ")

# =====================================================================
# 5. Per-segment cornering drag with Pacejka model
# =====================================================================
print("\n" + "=" * 80)
print("5. PACEJKA CORNERING DRAG vs ANALYTICAL (at typical conditions)")
print("=" * 80)

try:
    from fsae_sim.vehicle.tire_model import PacejkaTireModel
    from fsae_sim.vehicle.load_transfer import LoadTransferModel
    from fsae_sim.vehicle.vehicle import VehicleParams, SuspensionConfig

    tir_path = os.path.join(PROJECT_ROOT, config["tire"]["tir_file"])
    tire = PacejkaTireModel(tir_path)

    # Apply grip scale as the sim does
    grip_scale = config["tire"]["grip_scale"]
    tire.apply_grip_scale(grip_scale)
    print(f"  Grip scale applied: {grip_scale}")
    print(f"  LMUY after scaling: {tire.scaling.get('LMUY', 1.0):.4f}")

    vparams = VehicleParams(**config["vehicle"])
    susp_cfg = SuspensionConfig(**config["suspension"])
    lt = LoadTransferModel(vparams, susp_cfg)

    # Compare analytical vs Pacejka cornering drag
    from fsae_sim.vehicle.dynamics import VehicleDynamics
    from fsae_sim.vehicle.cornering_solver import CorneringSolver
    from fsae_sim.vehicle.powertrain import PowertrainConfig

    cs = CorneringSolver(
        tire, lt, mass_kg,
        math.radians(config["tire"]["static_camber_front_deg"]),
        math.radians(config["tire"]["static_camber_rear_deg"]),
        config["suspension"]["roll_camber_front_deg_per_deg"],
        config["suspension"]["roll_camber_rear_deg_per_deg"],
    )
    pt_cfg = PowertrainConfig(**config["powertrain"])

    dynamics_pacejka = VehicleDynamics(vparams, tire, lt, cs, powertrain_config=pt_cfg)
    dynamics_legacy = VehicleDynamics(vparams, powertrain_config=pt_cfg)  # no tire model

    print(f"\n  Effective mass (Pacejka): {dynamics_pacejka.m_effective:.2f} kg (vs bare {mass_kg} kg)")
    print(f"  Rotational inertia adds: {dynamics_pacejka.m_effective - mass_kg:.2f} kg effective")

    print(f"\n  {'Speed':>8} | {'Curv':>8} | {'Analytical':>12} | {'Pacejka':>12} | {'Ratio':>8}")
    print(f"  {'(km/h)':>8} | {'(1/m)':>8} | {'(N)':>12} | {'(N)':>12} | {'P/A':>8}")
    print("  " + "-" * 65)

    test_cases = [
        (20, 0.033),
        (20, 0.067),
        (20, 0.2),
        (30, 0.033),
        (30, 0.067),
        (40, 0.033),
        (40, 0.067),
        (50, 0.033),
    ]

    for v_kmh, kappa in test_cases:
        v_ms = v_kmh / 3.6
        cd_analytical = cornering_drag_analytical(v_ms, kappa)
        cd_pacejka = dynamics_pacejka.cornering_drag(v_ms, kappa)
        ratio = cd_pacejka / cd_analytical if cd_analytical > 0 else float('nan')
        print(f"  {v_kmh:8.0f} | {kappa:8.3f} | {cd_analytical:12.2f} | {cd_pacejka:12.2f} | {ratio:8.2f}")

    # Now compute full-lap cornering drag with Pacejka
    if track is not None:
        total_cd_pacejka_work = 0.0
        total_cd_analytical_work = 0.0
        total_aero_work_full = 0.0
        total_rr_work_full = 0.0

        for seg in track.segments:
            kappa = abs(seg.curvature)
            if kappa < 0.005:
                v_est = avg_speed_estimates["straight"]
            elif kappa < 0.02:
                v_est = avg_speed_estimates["gentle"]
            elif kappa < 0.05:
                v_est = avg_speed_estimates["medium"]
            elif kappa < 0.1:
                v_est = avg_speed_estimates["tight"]
            else:
                v_est = avg_speed_estimates["hairpin"]

            cd_p = dynamics_pacejka.cornering_drag(v_est, seg.curvature)
            cd_a = cornering_drag_analytical(v_est, seg.curvature)
            fd = aero_drag(v_est)
            frr = rolling_resistance_force(v_est)

            total_cd_pacejka_work += cd_p * seg.length_m
            total_cd_analytical_work += cd_a * seg.length_m
            total_aero_work_full += fd * seg.length_m
            total_rr_work_full += frr * seg.length_m

        total_pacejka_work = total_aero_work_full + total_rr_work_full + total_cd_pacejka_work
        total_analytical_work = total_aero_work_full + total_rr_work_full + total_cd_analytical_work

        print(f"\n  Full-lap energy comparison:")
        print(f"    Cornering drag (Pacejka):    {total_cd_pacejka_work/1000:.2f} kJ")
        print(f"    Cornering drag (Analytical): {total_cd_analytical_work/1000:.2f} kJ")
        print(f"    Ratio (Pacejka/Analytical):  {total_cd_pacejka_work/total_cd_analytical_work:.3f}")
        print(f"    Total resistance (Pacejka):  {total_pacejka_work/1000:.2f} kJ")
        print(f"    Total resistance (Analytical):{total_analytical_work/1000:.2f} kJ")

except Exception as e:
    print(f"  [ERROR] Pacejka comparison failed: {e}")
    import traceback
    traceback.print_exc()

# =====================================================================
# 6. Sensitivity analysis: effect of reducing each resistance by 20%
# =====================================================================
print("\n" + "=" * 80)
print("6. SENSITIVITY ANALYSIS: 20% REDUCTION IN EACH COMPONENT")
print("=" * 80)

if track is not None:
    # Reference: compute baseline lap time with force balance
    # Use energy method: total_time = integral(ds/v)
    # Extra resistance -> lower speed -> higher time
    # delta_time ~ (1/m) * integral(delta_F * ds / v^2) approx
    # Simpler: delta_v/v ~ delta_F / (2 * total_F) for small changes
    # => delta_time/time ~ -delta_v/v ~ delta_F / (2 * F_total)

    # More rigorous: P = F*v, so F = P/v. If we reduce F by delta_F,
    # the excess power goes to acceleration: delta_P = delta_F * v
    # Over distance ds, extra speed: delta_v ~ (delta_F/m) * (ds/v)
    # Total time saved: sum(delta_v * ds / v^2)

    # Let's do it properly: for each segment, F=ma, so if we remove delta_F,
    # a_extra = delta_F / m, giving delta_time over each segment
    # delta_time = -0.5 * (delta_F/m) * (L/v^2) approximately

    components = {
        "Aero drag": lambda v, k: aero_drag(v),
        "Rolling resistance": lambda v, k: rolling_resistance_force(v),
        "Cornering drag": lambda v, k: cornering_drag_analytical(v, k),
    }

    print(f"\n  {'Component':<25} | {'Avg Force':>10} | {'Work/lap':>10} | {'dTime 20%':>10} | {'dTime 100%':>10}")
    print(f"  {'':25} | {'(N)':>10} | {'(kJ)':>10} | {'(s)':>10} | {'(s)':>10}")
    print("  " + "-" * 75)

    total_resistance_avg = 0.0
    for name, f_func in components.items():
        total_force = 0.0
        total_work = 0.0
        delta_time_20pct = 0.0
        delta_time_100pct = 0.0
        n_segs = 0

        for seg in track.segments:
            kappa = abs(seg.curvature)
            if kappa < 0.005:
                v_est = avg_speed_estimates["straight"]
            elif kappa < 0.02:
                v_est = avg_speed_estimates["gentle"]
            elif kappa < 0.05:
                v_est = avg_speed_estimates["medium"]
            elif kappa < 0.1:
                v_est = avg_speed_estimates["tight"]
            else:
                v_est = avg_speed_estimates["hairpin"]

            f = f_func(v_est, seg.curvature)
            total_force += f
            total_work += f * seg.length_m
            n_segs += 1

            # Time saved by reducing this force by 20%
            delta_f = 0.20 * f
            # a_extra = delta_f / m_effective
            # Assuming constant speed through segment (quasi-static):
            # seg_time = L / v
            # With extra acceleration: v_new ~ v + a_extra * (L/v) / 2
            # seg_time_new ~ L / v_new ~ L / (v + delta_v) ~ (L/v)(1 - delta_v/v)
            # delta_time = -(L/v) * delta_v / v = -(L/v) * (delta_f * L / (2*m*v)) / v
            #            = -delta_f * L^2 / (2 * m * v^3)
            m_eff = dynamics_pacejka.m_effective if 'dynamics_pacejka' in dir() else mass_kg + 10
            dt = delta_f * seg.length_m**2 / (2.0 * m_eff * v_est**3)
            delta_time_20pct += dt
            delta_time_100pct += dt * 5.0  # 100% removal = 5x the 20% effect

        avg_force = total_force / max(n_segs, 1)
        total_resistance_avg += avg_force
        print(f"  {name:<25} | {avg_force:10.2f} | {total_work/1000:10.2f} | {delta_time_20pct:10.3f} | {delta_time_100pct:10.3f}")

    print(f"\n  Total resistance avg:  {total_resistance_avg:.2f} N")
    print(f"\n  Target: 8.4 s/lap faster needed to fix 11.4% error")

# =====================================================================
# 7. Downforce effect on rolling resistance
# =====================================================================
print("\n" + "=" * 80)
print("7. DOWNFORCE EFFECT ON ROLLING RESISTANCE")
print("=" * 80)

print(f"\n  Base RR (no aero): {mass_kg * g * rolling_res:.2f} N")
for v_kmh in [10, 20, 30, 40, 50, 60]:
    v_ms = v_kmh / 3.6
    df = downforce(v_ms)
    rr_base = mass_kg * g * rolling_res
    rr_with_df = (mass_kg * g + df) * rolling_res
    extra = rr_with_df - rr_base
    pct_increase = 100 * extra / rr_base
    print(f"  {v_kmh:3.0f} km/h: downforce = {df:6.1f} N, RR = {rr_with_df:.2f} N (+{extra:.2f} N = +{pct_increase:.1f}%)")

# Speed where downforce doubles rolling resistance
# mass*g = 0.5*rho*ClA*v^2 => v = sqrt(2*m*g / (rho*ClA))
v_double = math.sqrt(2 * mass_kg * g / (rho * downforce_coeff))
print(f"\n  Speed where downforce equals car weight: {v_double:.1f} m/s = {v_double*3.6:.1f} km/h")
print(f"  (At this speed, rolling resistance is doubled by downforce)")

# =====================================================================
# 8. Rolling resistance coefficient check
# =====================================================================
print("\n" + "=" * 80)
print("8. ROLLING RESISTANCE COEFFICIENT CHECK")
print("=" * 80)

print(f"\n  Config value: Crr = {rolling_res}")
print(f"  Typical ranges for Hoosier LC0 on asphalt:")
print(f"    Low estimate:  0.015  (fresh smooth asphalt)")
print(f"    Mid estimate:  0.020  (typical FSAE autocross surface)")
print(f"    High estimate: 0.025  (rough/hot asphalt)")
print(f"    Very high:     0.030  (very rough surface)")
print(f"\n  Config Crr = {rolling_res} is at the {'LOW' if rolling_res < 0.018 else 'MID' if rolling_res < 0.022 else 'HIGH'} end of the range")

if rolling_res == 0.015:
    print(f"  NOTE: 0.015 is the absolute minimum for racing slicks.")
    print(f"  Real FSAE endurance surfaces are rougher. Consider 0.020-0.025.")
    rr_sensitivity = mass_kg * g * (0.020 - 0.015)
    print(f"  Increasing to 0.020 would ADD {rr_sensitivity:.1f} N constant resistance")
    print(f"  ... which would make the car SLOWER, not faster.")
    print(f"  Since we're already too slow, RR is not the culprit unless it's too HIGH.")

# =====================================================================
# 9. What force error produces 8.4 s/lap?
# =====================================================================
print("\n" + "=" * 80)
print("9. FORCE ERROR NEEDED TO EXPLAIN 8.4 s/lap")
print("=" * 80)

if track is not None:
    lap_dist = track.total_distance_m
    target_lap_time_real = 74.0  # approximate from telemetry
    target_lap_time_sim = target_lap_time_real * 1.114  # 11.4% slower
    delta_time_target = target_lap_time_sim - target_lap_time_real
    avg_speed = lap_dist / target_lap_time_real

    print(f"  Lap distance:          {lap_dist:.1f} m")
    print(f"  Real lap time:         {target_lap_time_real:.1f} s")
    print(f"  Sim lap time:          {target_lap_time_sim:.1f} s")
    print(f"  Time error per lap:    {delta_time_target:.1f} s")
    print(f"  Average speed (real):  {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)")

    # delta_time/time ~ delta_F / (m * a_avg + F_resist)
    # More direct: average deceleration from excess resistance
    # delta_v ~ delta_F/m * t, over lap with avg speed v_avg
    # delta_time ~ lap_dist * delta_v / v_avg^2
    # => delta_F ~ delta_time * m * v_avg^2 / (lap_dist * t_lap)

    # Actually: if the car is slower by delta_v_avg on average,
    # delta_time = lap_dist / (v_avg - delta_v_avg) - lap_dist / v_avg
    #            ~ lap_dist * delta_v_avg / v_avg^2  (for small delta_v)
    # delta_v_avg from constant force: delta_v_avg ~ delta_F * t / (2*m)
    # Not quite right for quasi-static... better approach:
    # Power balance: P_drive = P_resist. Extra resistance = delta_F.
    # New avg speed: v_new s.t. P_drive = (F_resist + delta_F) * v_new
    # P_drive = F_resist * v_avg (original balance)
    # => F_resist * v_avg = (F_resist + delta_F) * v_new
    # => v_new = v_avg * F_resist / (F_resist + delta_F)
    # => delta_time/time = (F_resist + delta_F) / F_resist - 1 = delta_F / F_resist
    # => delta_F = delta_time/time * F_resist

    # Estimate average total resistance
    total_resist_avg = 0.0
    for seg in track.segments:
        kappa = abs(seg.curvature)
        if kappa < 0.005:
            v_est = avg_speed_estimates["straight"]
        elif kappa < 0.02:
            v_est = avg_speed_estimates["gentle"]
        elif kappa < 0.05:
            v_est = avg_speed_estimates["medium"]
        elif kappa < 0.1:
            v_est = avg_speed_estimates["tight"]
        else:
            v_est = avg_speed_estimates["hairpin"]

        fr = aero_drag(v_est) + rolling_resistance_force(v_est) + cornering_drag_analytical(v_est, seg.curvature)
        total_resist_avg += fr

    avg_resistance = total_resist_avg / track.num_segments

    frac_error = delta_time_target / target_lap_time_real  # 0.114
    delta_F_needed = frac_error * avg_resistance

    print(f"\n  Average total resistance: {avg_resistance:.1f} N")
    print(f"  delta_F needed for 11.4% error: {delta_F_needed:.1f} N")
    print(f"  That's {100*delta_F_needed/avg_resistance:.1f}% of total resistance")

    # Alternative: direct F=ma approach
    # If car accelerates 11.4% less on average:
    m_eff = dynamics_pacejka.m_effective if 'dynamics_pacejka' in dir() else mass_kg + 10
    delta_a = 2 * avg_speed * delta_time_target / (target_lap_time_real**2)
    delta_F_fma = m_eff * delta_a
    print(f"\n  Alternative (F=ma approach):")
    print(f"    delta_a needed:  {delta_a:.4f} m/s^2")
    print(f"    delta_F needed:  {delta_F_fma:.1f} N")

# =====================================================================
# 10. Cornering drag dominance analysis
# =====================================================================
print("\n" + "=" * 80)
print("10. CORNERING DRAG DOMINANCE ANALYSIS")
print("=" * 80)

if track is not None:
    # For each segment, compute the ratio of cornering drag to total resistance
    high_cd_segments = 0
    dominant_cd_segments = 0

    print(f"\n  Top 20 segments by cornering drag force:")
    print(f"  {'Seg':>5} | {'Dist':>7} | {'Curv':>8} | {'Radius':>8} | {'V_est':>7} | {'Corn Drag':>10} | {'Aero':>7} | {'RR':>7} | {'CD/Total':>8}")
    print("  " + "-" * 90)

    cd_forces = []
    for seg in track.segments:
        kappa = abs(seg.curvature)
        if kappa < 0.005:
            v_est = avg_speed_estimates["straight"]
        elif kappa < 0.02:
            v_est = avg_speed_estimates["gentle"]
        elif kappa < 0.05:
            v_est = avg_speed_estimates["medium"]
        elif kappa < 0.1:
            v_est = avg_speed_estimates["tight"]
        else:
            v_est = avg_speed_estimates["hairpin"]

        fc = cornering_drag_analytical(v_est, seg.curvature)
        fd = aero_drag(v_est)
        frr = rolling_resistance_force(v_est)
        total = fd + frr + fc
        ratio = fc / total if total > 0 else 0

        if ratio > 0.3:
            high_cd_segments += 1
        if ratio > 0.5:
            dominant_cd_segments += 1

        cd_forces.append((seg.index, seg.distance_start_m, seg.curvature, v_est, fc, fd, frr, ratio))

    # Sort by cornering drag force
    cd_forces.sort(key=lambda x: x[4], reverse=True)
    for i, (idx, dist, curv, v_est, fc, fd, frr, ratio) in enumerate(cd_forces[:20]):
        radius = 1.0 / abs(curv) if abs(curv) > 1e-6 else float('inf')
        print(f"  {idx:5d} | {dist:7.1f} | {curv:8.4f} | {radius:8.1f} | {v_est*3.6:7.1f} | {fc:10.2f} | {fd:7.2f} | {frr:7.2f} | {ratio:8.1%}")

    print(f"\n  Segments where cornering drag > 30% of total: {high_cd_segments} / {track.num_segments} ({100*high_cd_segments/track.num_segments:.1f}%)")
    print(f"  Segments where cornering drag > 50% of total: {dominant_cd_segments} / {track.num_segments} ({100*dominant_cd_segments/track.num_segments:.1f}%)")

# =====================================================================
# 11. Analytical cornering drag formula audit
# =====================================================================
print("\n" + "=" * 80)
print("11. ANALYTICAL CORNERING DRAG FORMULA AUDIT")
print("=" * 80)

print(f"\n  Formula: F_cd = F_lat^2 / C_alpha_total")
print(f"  Where:")
print(f"    F_lat = m * v^2 * kappa  (centripetal force needed)")
print(f"    C_alpha_total = m * g * mu_peak / alpha_peak")
print(f"                  = {mass_kg} * {g} * 1.5 / 0.15")
mu_peak = 1.5
alpha_peak = 0.15
c_alpha_total = mass_kg * g * mu_peak / alpha_peak
print(f"                  = {c_alpha_total:.1f} N/rad")
print(f"\n  Key assumptions:")
print(f"    mu_peak = 1.5 (peak tire friction)")
print(f"    alpha_peak = 0.15 rad = {math.degrees(0.15):.1f} deg (peak slip angle)")

# Check: with grip_scale = 0.4697, the effective mu_peak should be lower
effective_mu = 1.5 * grip_scale if 'grip_scale' in dir() else 1.5
print(f"\n  BUT: grip_scale = {grip_scale}")
print(f"    Effective mu = {effective_mu:.3f}")
print(f"    C_alpha with real grip = {mass_kg * g * effective_mu / alpha_peak:.1f} N/rad")
print(f"    -> The analytical model uses mu_peak=1.5, but the Pacejka model")
print(f"       uses grip_scale=0.4697. These are DIFFERENT models.")
print(f"    -> Analytical: higher C_alpha -> lower cornering drag (optimistic)")
print(f"    -> Pacejka: lower grip -> tires saturate sooner -> higher cornering drag")

# How much does the analytical underestimate?
c_alpha_analytical = c_alpha_total
c_alpha_scaled = mass_kg * g * effective_mu / alpha_peak
print(f"\n  C_alpha ratio (analytical/scaled): {c_alpha_analytical / c_alpha_scaled:.2f}")
print(f"  Cornering drag ratio (scaled/analytical): {c_alpha_analytical / c_alpha_scaled:.2f}")
print(f"  -> Analytical underestimates cornering drag by {c_alpha_analytical / c_alpha_scaled:.1f}x if grip is actually {effective_mu:.2f}")

# =====================================================================
# 12. Effective mass analysis
# =====================================================================
print("\n" + "=" * 80)
print("12. EFFECTIVE MASS (ROTATIONAL INERTIA) ANALYSIS")
print("=" * 80)

rotor_inertia = vp.get("rotor_inertia_kg_m2", 0.06)
wheel_inertia = vp.get("wheel_inertia_kg_m2", 0.3)
gear_ratio = config["powertrain"]["gear_ratio"]
eta = config["powertrain"]["drivetrain_efficiency"]
tire_radius = 0.228

j_eff = rotor_inertia * gear_ratio**2 * eta + 4 * wheel_inertia
m_eff = mass_kg + j_eff / tire_radius**2

print(f"  Rotor inertia:     {rotor_inertia} kg*m^2")
print(f"  Wheel inertia:     {wheel_inertia} kg*m^2 (per wheel)")
print(f"  Gear ratio:        {gear_ratio}")
print(f"  Drivetrain eff:    {eta}")
print(f"  Tire radius:       {tire_radius} m")
print(f"  J_eff:             {j_eff:.4f} kg*m^2")
print(f"  m_effective:        {m_eff:.2f} kg  (bare: {mass_kg} kg, added: {m_eff - mass_kg:.2f} kg)")
print(f"  Mass increase:     {100*(m_eff - mass_kg)/mass_kg:.1f}%")
print(f"\n  Component breakdown:")
print(f"    Motor rotor:     {rotor_inertia * gear_ratio**2 * eta / tire_radius**2:.2f} kg (through gear ratio)")
print(f"    4 wheels:        {4 * wheel_inertia / tire_radius**2:.2f} kg")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 80)
print("SUMMARY: RESISTANCE MODEL DIAGNOSIS")
print("=" * 80)

print(f"""
The sim is 11.4% too slow. Here's where the resistance budget stands:

1. AERODYNAMIC DRAG (CdA = {drag_coeff * frontal_area:.3f} m^2)
   - DSS verified: 431 N at 80 kph matches config
   - At FSAE speeds (20-50 km/h), drag is {aero_drag(35/3.6):.0f}-{aero_drag(50/3.6):.0f} N
   - Relatively small at low FSAE speeds — NOT the main suspect

2. ROLLING RESISTANCE (Crr = {rolling_res})
   - Crr = {rolling_res} is at the low end of the range (0.015-0.025)
   - If anything, real Crr might be HIGHER, making the car even slower
   - Constant ~{mass_kg * g * rolling_res:.0f} N base + downforce contribution
   - NOT the culprit (already conservative)

3. CORNERING DRAG — THE PRIMARY SUSPECT
   - FSAE tracks are predominantly curved (autocross-style layout)
""")

if track is not None and curved_frac is not None:
    print(f"   - Track is {100*curved_frac:.0f}% curved")
    print(f"   - Cornering drag accounts for {100*total_cd_work/total_resistance_work:.0f}% of total resistance work")

print(f"""   - At tight corners (r<10m), cornering drag can be 10-50x aero drag
   - The analytical model uses mu_peak=1.5 but Pacejka uses grip_scale={grip_scale}
   - Pacejka cornering drag is likely {c_alpha_analytical / c_alpha_scaled:.1f}x the analytical estimate

4. DOWNFORCE-AUGMENTED ROLLING RESISTANCE
   - At 30 km/h: +{100*(rolling_resistance_force(30/3.6) / (mass_kg*g*rolling_res) - 1):.1f}% RR increase
   - At 50 km/h: +{100*(rolling_resistance_force(50/3.6) / (mass_kg*g*rolling_res) - 1):.1f}% RR increase
   - Not huge at FSAE speeds but adds up

5. EFFECTIVE MASS ({m_eff:.0f} kg vs {mass_kg:.0f} kg bare)
   - Rotational inertia adds {m_eff - mass_kg:.0f} kg effective mass ({100*(m_eff-mass_kg)/mass_kg:.1f}%)
   - This slows acceleration but doesn't affect steady-state speed
   - Wheel inertia of {wheel_inertia} kg*m^2 per wheel is a rough estimate
""")

if track is not None and 'delta_F_needed' in dir() and 'avg_resistance' in dir():
    print(f"""6. FORCE ERROR NEEDED: ~{delta_F_needed:.0f} N average excess resistance
   - This is {100*delta_F_needed/avg_resistance:.0f}% of the average resistance
   - Cornering drag is the most likely source of this excess
""")

print("=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
1. INVESTIGATE CORNERING DRAG: The Pacejka path uses grip_scale=0.4697 which
   dramatically changes the cornering drag vs the analytical fallback. Need to
   verify which path the sim actually takes and whether the force is realistic.

2. CHECK EFFECTIVE MASS: wheel_inertia=0.3 kg*m^2 is a rough estimate.
   Measure or calculate from wheel mass and geometry.

3. VALIDATE AGAINST TELEMETRY SEGMENT-BY-SEGMENT: Compare sim resistance
   forces to real deceleration during coasting segments in telemetry. This
   is the definitive test — coast deceleration = total_resistance / m.

4. DO NOT lower CdA or Crr below physical values to make the sim faster.
   The error is more likely in cornering drag or in the drive force model.
""")
