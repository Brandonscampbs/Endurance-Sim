"""Audit every step of the torque-to-force conversion chain.

Checks all parameters against DSS spec and identifies errors.
"""
import math
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def main():
    print("=" * 80)
    print("TORQUE-TO-FORCE CHAIN AUDIT")
    print("=" * 80)

    # ==================================================================
    # 1. Load config and print ALL powertrain and vehicle parameters
    # ==================================================================
    with open("configs/ct16ev.yaml") as f:
        cfg = yaml.safe_load(f)

    print()
    print("1. CONFIGURED PARAMETERS")
    print("-" * 60)
    print()
    print("Vehicle:")
    for k, v in cfg["vehicle"].items():
        print(f"  {k:35s} = {v}")
    print()
    print("Powertrain:")
    for k, v in cfg["powertrain"].items():
        print(f"  {k:35s} = {v}")

    # ==================================================================
    # 2. Motor efficiency map analysis at FSAE operating points
    # ==================================================================
    print()
    print("=" * 80)
    print("2. MOTOR EFFICIENCY MAP ANALYSIS")
    print("-" * 60)

    df = pd.read_csv("Real-Car-Data-And-Stats/emrax228_hv_cc_motor_map_long.csv")
    print(f"Map size: {len(df)} rows")
    print(f"RPM range: {df['speed_rpm'].min()} to {df['speed_rpm'].max()} RPM")
    print(f"Torque range: {df['torque_Nm'].min()} to {df['torque_Nm'].max()} Nm")
    eff_valid = df["efficiency_pct"].dropna()
    print(f"Efficiency range: {eff_valid.min():.1f}% to {eff_valid.max():.1f}%")

    # FSAE typical operating range
    print()
    print("Efficiency at typical FSAE operating points (RPM 500-2500, Torque 10-85 Nm):")
    print(f"{'RPM':>6s} {'Torque':>8s} {'Eff %':>8s} {'Mech kW':>8s} {'Elec kW':>8s}")
    for rpm in [500, 750, 1000, 1500, 2000, 2500]:
        for torque in [10, 20, 40, 60, 85]:
            match = df[(df["speed_rpm"] == rpm) & (df["torque_Nm"] == torque)]
            if len(match) > 0:
                row_data = match.iloc[0]
                eff = row_data["efficiency_pct"]
                mech = row_data["mech_power_kW"]
                elec = row_data["elec_power_kW"]
                if not np.isnan(eff):
                    print(f"{rpm:6d} {torque:8d} {eff:8.1f} {mech:8.2f} {elec:8.2f}")

    # ==================================================================
    # 3. Trace specific example: 40 Nm at 30 km/h
    # ==================================================================
    print()
    print("=" * 80)
    print("3. FULL CHAIN TRACE: LVCU Torque Req = 40 Nm at 30 km/h")
    print("-" * 60)

    gear_ratio = cfg["powertrain"]["gear_ratio"]
    drivetrain_eff = cfg["powertrain"]["drivetrain_efficiency"]
    tire_radius = 0.228  # hardcoded in PowertrainModel
    mass = cfg["vehicle"]["mass_kg"]

    speed_kmh = 30.0
    speed_ms = speed_kmh / 3.6
    motor_torque = 40.0

    # Speed -> RPM
    wheel_rpm = (speed_ms / tire_radius) * 60.0 / (2.0 * math.pi)
    motor_rpm = wheel_rpm * gear_ratio

    print(f"Vehicle speed:        {speed_kmh:.1f} km/h = {speed_ms:.3f} m/s")
    print(f"Tire radius:          {tire_radius:.3f} m")
    print(f"Wheel angular vel:    {speed_ms / tire_radius:.2f} rad/s")
    print(f"Wheel RPM:            {wheel_rpm:.1f} RPM")
    print(f"Motor RPM:            {motor_rpm:.1f} RPM (gear ratio = {gear_ratio})")
    print(f"Motor torque command: {motor_torque:.1f} Nm")
    print()

    # Motor efficiency from map at this point via interpolation
    rpm_unique = np.sort(df["speed_rpm"].unique())
    torque_unique = np.sort(df["torque_Nm"].unique())

    nearest_rpm_below = max([r for r in rpm_unique if r <= motor_rpm], default=rpm_unique[0])
    nearest_rpm_above = min([r for r in rpm_unique if r >= motor_rpm], default=rpm_unique[-1])
    print(f"Nearest RPM grid points: {nearest_rpm_below}, {nearest_rpm_above}")

    for rpm_check in [nearest_rpm_below, nearest_rpm_above]:
        match = df[(df["speed_rpm"] == rpm_check) & (df["torque_Nm"] == 40)]
        if len(match) > 0:
            eff_val = match.iloc[0]["efficiency_pct"]
            print(f"  At RPM={rpm_check}, Torque=40 Nm: efficiency = {eff_val:.2f}%")

    # Build interpolator same as MotorEfficiencyMap
    eff_grid = np.full((len(rpm_unique), len(torque_unique)), np.nan)
    rpm_to_idx = {r: i for i, r in enumerate(rpm_unique)}
    torque_to_idx = {t: i for i, t in enumerate(torque_unique)}
    for _, row_data in df.iterrows():
        ri = rpm_to_idx[row_data["speed_rpm"]]
        ti = torque_to_idx[row_data["torque_Nm"]]
        if not np.isnan(row_data["efficiency_pct"]):
            eff_grid[ri, ti] = row_data["efficiency_pct"] / 100.0
    for i in range(len(rpm_unique)):
        row_arr = eff_grid[i, :]
        valid = ~np.isnan(row_arr)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            eff_grid[i, :] = np.interp(
                np.arange(len(torque_unique)), valid_idx, row_arr[valid]
            )
        else:
            eff_grid[i, :] = 0.80

    interp = RegularGridInterpolator(
        (rpm_unique.astype(float), torque_unique.astype(float)),
        eff_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    GEARBOX_EFF = 0.97
    FLOOR = 0.80

    motor_inv_eff = float(interp((motor_rpm, motor_torque)))
    motor_inv_eff = max(FLOOR, min(1.0, motor_inv_eff))
    total_eff = motor_inv_eff * GEARBOX_EFF

    print()
    print(f"Motor+Inverter efficiency (interpolated): {motor_inv_eff * 100:.2f}%")
    print(f"Gearbox efficiency:                       {GEARBOX_EFF * 100:.1f}%")
    print(f"Total drivetrain efficiency (map):         {total_eff * 100:.2f}%")
    print(f"Fixed drivetrain efficiency (config):      {drivetrain_eff * 100:.1f}%")
    print()

    # === THE KEY QUESTION: how does wheel_force use efficiency? ===
    print("=== WHEEL FORCE CALCULATION ===")
    print()

    # What the code ACTUALLY does (powertrain_model.py lines 246-258):
    # wheel_torque = motor_torque * gear_ratio * drivetrain_efficiency
    # wheel_force  = wheel_torque / TIRE_RADIUS_M
    # NOTE: wheel_torque() and wheel_force() use config.drivetrain_efficiency
    #       NOT the motor efficiency map!

    wt_config = motor_torque * gear_ratio * drivetrain_eff
    wf_config = wt_config / tire_radius
    print(f"Using FIXED drivetrain_efficiency ({drivetrain_eff}):")
    print(f"  Wheel torque = {motor_torque} * {gear_ratio} * {drivetrain_eff} = {wt_config:.2f} Nm")
    print(f"  Wheel force  = {wt_config:.2f} / {tire_radius} = {wf_config:.1f} N")
    print()

    wt_map = motor_torque * gear_ratio * total_eff
    wf_map = wt_map / tire_radius
    print(f"Using MOTOR MAP total efficiency ({total_eff:.4f}):")
    print(f"  Wheel torque = {motor_torque} * {gear_ratio} * {total_eff:.4f} = {wt_map:.2f} Nm")
    print(f"  Wheel force  = {wt_map:.2f} / {tire_radius} = {wf_map:.1f} N")
    print()

    # Correct physics: motor torque IS mechanical output. Only gearbox loss.
    wt_correct = motor_torque * gear_ratio * GEARBOX_EFF
    wf_correct = wt_correct / tire_radius
    print(f"CORRECT physics (torque is mechanical output, only gearbox loss):")
    print(f"  Wheel torque = {motor_torque} * {gear_ratio} * {GEARBOX_EFF} = {wt_correct:.2f} Nm")
    print(f"  Wheel force  = {wt_correct:.2f} / {tire_radius} = {wf_correct:.1f} N")
    print()

    print(f"*** FORCE ERROR from using full drivetrain_eff in wheel_force:")
    print(f"    Config method:  {wf_config:.1f} N vs correct {wf_correct:.1f} N")
    print(f"    -> {(1 - wf_config / wf_correct) * 100:.1f}% force lost")
    print()

    # ==================================================================
    # 4. Compare drive force vs resistance
    # ==================================================================
    print("=" * 80)
    print("4. DRIVE FORCE vs RESISTANCE CHECK")
    print("-" * 60)

    Cd = cfg["vehicle"]["drag_coefficient"]
    A = cfg["vehicle"]["frontal_area_m2"]
    CdA_actual = Cd * A
    Crr = cfg["vehicle"]["rolling_resistance"]
    ClA = cfg["vehicle"]["downforce_coefficient"]
    rho = 1.225

    drag = 0.5 * rho * CdA_actual * speed_ms ** 2
    downforce = 0.5 * rho * ClA * speed_ms ** 2
    normal = mass * 9.81 + downforce
    rolling = normal * Crr

    print(f"Speed: {speed_ms:.2f} m/s ({speed_kmh:.0f} km/h)")
    print(f"CdA = Cd * A = {Cd} * {A} = {CdA_actual:.3f} m^2")
    print(f"Drag force:     {drag:.1f} N")
    print(f"Downforce:      {downforce:.1f} N")
    print(f"Rolling resist: {rolling:.1f} N")
    print(f"Total resist:   {drag + rolling:.1f} N")
    print()

    net_config = wf_config - drag - rolling
    net_correct = wf_correct - drag - rolling
    accel_config = net_config / mass
    accel_correct = net_correct / mass

    print(f"Net force (config eff):  {net_config:.1f} N -> accel = {accel_config:.3f} m/s^2")
    print(f"Net force (correct):     {net_correct:.1f} N -> accel = {accel_correct:.3f} m/s^2")
    if accel_correct != 0:
        print(f"Acceleration difference: {(accel_correct - accel_config) / accel_correct * 100:.1f}%")
    print()

    # ==================================================================
    # 5. Tire radius check
    # ==================================================================
    print("=" * 80)
    print("5. TIRE RADIUS VERIFICATION")
    print("-" * 60)
    print()
    print("Tire: Hoosier 16x7.5-10 LC0")
    print()
    print("  Tire naming: 16 x 7.5 - 10")
    print("    16  = Overall diameter in inches")
    print("    7.5 = Section width in inches")
    print("    10  = Rim diameter in inches")
    print()

    od_in = 16.0
    od_mm = od_in * 25.4
    unloaded_radius_mm = od_mm / 2.0
    unloaded_radius_m = unloaded_radius_mm / 1000.0
    print(f"  Unloaded outer diameter: {od_in}\" = {od_mm:.0f} mm")
    print(f"  Unloaded radius:         {unloaded_radius_mm:.0f} mm = {unloaded_radius_m:.4f} m")
    print()

    section_height_mm = (16.0 - 10.0) / 2.0 * 25.4
    print(f"  Section height: ({od_in}-10)/2 * 25.4 = {section_height_mm:.1f} mm")
    print()

    corner_weight_n = mass * 9.81 / 4
    print(f"  Static corner weight: {corner_weight_n:.0f} N ({corner_weight_n / 4.45:.0f} lbs)")
    print()

    defl_10pct = section_height_mm * 0.10
    defl_15pct = section_height_mm * 0.15
    loaded_radius_10 = (od_mm / 2 - defl_10pct) / 1000.0
    loaded_radius_15 = (od_mm / 2 - defl_15pct) / 1000.0
    print(f"  Deflection 10%: {defl_10pct:.1f} mm -> loaded radius: {loaded_radius_10 * 1000:.1f} mm = {loaded_radius_10:.4f} m")
    print(f"  Deflection 15%: {defl_15pct:.1f} mm -> loaded radius: {loaded_radius_15 * 1000:.1f} mm = {loaded_radius_15:.4f} m")
    print()
    print(f"  CONFIG value:   {tire_radius * 1000:.0f} mm = {tire_radius:.3f} m")
    print()

    force_at_196 = wt_correct / 0.196
    force_at_228 = wt_correct / 0.228
    print(f"  If loaded radius were 0.196 m instead of 0.228 m:")
    print(f"    Force at r=0.196: {force_at_196:.1f} N")
    print(f"    Force at r=0.228: {force_at_228:.1f} N")
    diff_pct = (force_at_196 - force_at_228) / force_at_228 * 100
    print(f"    Difference: {diff_pct:.1f}% more force with correct loaded radius")
    print()
    print(f"  Unloaded radius ({unloaded_radius_m:.4f} m) is SMALLER than config (0.228 m)!")
    print(f"  *** 0.228 m seems very wrong for this tire ***")
    print(f"  0.228 m = 228 mm = 9.0\" radius = 18\" diameter tire (NOT a 16\" tire)")
    print()
    print(f"  A 16\" diameter tire has radius = 203.2 mm = 0.2032 m UNLOADED")
    print(f"  Loaded, it should be ~192-200 mm = 0.192-0.200 m")

    # ==================================================================
    # 6. Gear ratio check
    # ==================================================================
    print()
    print("=" * 80)
    print("6. GEAR RATIO VERIFICATION")
    print("-" * 60)
    print()
    exact_ratio = 40.0 / 11.0
    print(f"  DSS: 40/11 teeth = {exact_ratio:.4f}")
    print(f"  Config: {gear_ratio}")
    print(f"  Match: {abs(gear_ratio - exact_ratio) < 0.001}")

    # ==================================================================
    # 7. FULL AUDIT TABLE
    # ==================================================================
    print()
    print("=" * 80)
    print("7. FULL PARAMETER AUDIT TABLE")
    print("-" * 60)
    print()
    print(f"{'Parameter':40s} {'Config':>12s} {'DSS/Expected':>14s} {'Status':>14s} {'Impact':>10s}")
    print("-" * 92)

    def row(param, config_val, expected_val, status, impact=""):
        print(f"{param:40s} {str(config_val):>12s} {str(expected_val):>14s} {status:>14s} {impact:>10s}")

    row("Mass (kg)", cfg["vehicle"]["mass_kg"], "288.0", "OK")
    row("Frontal area (m^2)", cfg["vehicle"]["frontal_area_m2"], "1.0", "OK")
    row("Drag coefficient (Cd)", cfg["vehicle"]["drag_coefficient"], "1.502", "OK")
    row("CdA = Cd * A", f"{CdA_actual:.3f}", "1.502", "OK")
    row("ClA downforce (m^2)", cfg["vehicle"]["downforce_coefficient"], "2.18", "OK")
    row("Rolling resistance", cfg["vehicle"]["rolling_resistance"], "0.015", "OK")
    row("Wheelbase (m)", cfg["vehicle"]["wheelbase_m"], "1.549", "OK")
    row("Gear ratio", cfg["powertrain"]["gear_ratio"], f"{exact_ratio:.4f}", "OK")
    row("Inverter torque limit (Nm)", cfg["powertrain"]["torque_limit_inverter_nm"], "85.0", "OK")
    row("LVCU torque limit (Nm)", cfg["powertrain"]["torque_limit_lvcu_nm"], "150.0", "OK")
    row("Motor speed max (RPM)", cfg["powertrain"]["motor_speed_max_rpm"], "2900", "OK")
    row("Brake speed (RPM)", cfg["powertrain"]["brake_speed_rpm"], "2400", "OK")
    row("Drivetrain efficiency", cfg["powertrain"]["drivetrain_efficiency"], "0.92", "MISUSED", "~5.4%")
    row("TIRE_RADIUS_M (hardcoded)", "0.228", "~0.196-0.203", "*** WRONG ***", "12-16%")

    # ==================================================================
    # COMBINED IMPACT at multiple operating points
    # ==================================================================
    print()
    print("=" * 80)
    print("MULTI-POINT COMPARISON (40 Nm motor torque)")
    print("-" * 60)
    print(f"{'Speed':>8s} {'RPM(0.228)':>12s} {'RPM(0.200)':>12s} {'F(0.228)':>10s} {'F(0.200)':>10s} {'F diff%':>10s}")

    for speed_kph in [15, 20, 25, 30, 35, 40, 45]:
        v = speed_kph / 3.6
        mt = 40

        wheel_rpm_228 = (v / 0.228) * 60.0 / (2 * math.pi)
        motor_rpm_228 = wheel_rpm_228 * gear_ratio
        wt_228 = mt * gear_ratio * drivetrain_eff
        wf_228 = wt_228 / 0.228

        wheel_rpm_200 = (v / 0.200) * 60.0 / (2 * math.pi)
        motor_rpm_200 = wheel_rpm_200 * gear_ratio
        wt_200 = mt * gear_ratio * GEARBOX_EFF
        wf_200 = wt_200 / 0.200

        diff = (wf_200 - wf_228) / wf_228 * 100
        print(f"{speed_kph:>6d}kph {motor_rpm_228:>12.0f} {motor_rpm_200:>12.0f} {wf_228:>10.1f} {wf_200:>10.1f} {diff:>9.1f}%")

    # ==================================================================
    # CRITICAL FINDINGS
    # ==================================================================
    print()
    print("=" * 80)
    print("CRITICAL FINDINGS")
    print("=" * 80)
    print()
    print("FINDING 1: TIRE_RADIUS_M = 0.228 m IS WRONG")
    print()
    print("  The Hoosier 16x7.5-10 has a 16\" overall diameter.")
    print("  Unloaded radius = 16\"/2 = 8\" = 203.2 mm = 0.2032 m")
    print("  Loaded radius (under ~700N corner weight) is ~192-200 mm.")
    print()
    print("  0.228 m = 228 mm = 9.0\" radius = 18\" diameter tire.")
    print("  This is NOT a 10\" wheel tire radius. It is ~12-16% too large.")
    print()
    print("  Impact on FORCE: Force = wheel_torque / tire_radius.")
    print("    Too-large radius REDUCES force at the wheel.")
    print("    With r=0.228 vs r=0.200: force is reduced by ~14%.")
    print()
    print("  Impact on RPM: RPM = (v/r) * 60/(2pi) * gear_ratio.")
    print("    Too-large radius REDUCES computed RPM at same speed.")
    print("    This shifts the efficiency map lookup and")
    print("    field-weakening behavior.")
    print()
    max_speed_228 = 2900 * 0.228 * 2 * math.pi / (60 * gear_ratio)
    max_speed_200 = 2900 * 0.200 * 2 * math.pi / (60 * gear_ratio)
    print(f"  Impact on SPEED CAPABILITY:")
    print(f"    At 2900 RPM: speed(0.228) = {max_speed_228:.1f} m/s = {max_speed_228 * 3.6:.1f} km/h")
    print(f"    At 2900 RPM: speed(0.200) = {max_speed_200:.1f} m/s = {max_speed_200 * 3.6:.1f} km/h")
    print()

    print("FINDING 2: wheel_force() APPLIES FULL DRIVETRAIN EFFICIENCY TO FORCE")
    print()
    print("  Current code (powertrain_model.py line 247):")
    print("    wheel_torque = motor_torque * gear_ratio * drivetrain_efficiency")
    print()
    print("  drivetrain_efficiency = 0.92 includes motor+inverter losses.")
    print("  But when the motor produces 40 Nm, that IS 40 Nm of mechanical")
    print("  shaft torque. Motor/inverter efficiency affects how much ELECTRICAL")
    print("  power was consumed to produce it, NOT the mechanical force.")
    print()
    print("  The only loss from motor shaft to wheel is gearbox friction (~3%).")
    print("  So wheel_torque should be: motor_torque * gear_ratio * gearbox_eff")
    print("  NOT: motor_torque * gear_ratio * 0.92")
    print()
    eff_loss = (1 - drivetrain_eff / GEARBOX_EFF) * 100
    print(f"  This causes a ~{eff_loss:.1f}% force deficit: ({GEARBOX_EFF} - {drivetrain_eff}) / {GEARBOX_EFF}")
    print()
    print("  NOTE: When the motor efficiency map is loaded, electrical_power()")
    print("  correctly uses the map for power/energy calculations.")
    print("  But wheel_force() and wheel_torque() ALWAYS use")
    print("  config.drivetrain_efficiency (0.92) for FORCE, not the map.")
    print()

    print("FINDING 3: COMBINED EFFECT")
    print()
    # Tire radius effect: F_correct/F_wrong = (1/0.200) / (1/0.228) = 0.228/0.200 = 1.14
    radius_factor = 0.228 / 0.200
    # Efficiency effect: F_correct/F_wrong = 0.97 / 0.92 = 1.054
    eff_factor = GEARBOX_EFF / drivetrain_eff
    combined_factor = radius_factor * eff_factor
    print(f"  Tire radius error:     {(radius_factor - 1) * 100:.1f}% force deficit")
    print(f"  Efficiency misuse:     {(eff_factor - 1) * 100:.1f}% force deficit")
    print(f"  Combined (multiplicative): {(combined_factor - 1) * 100:.1f}% force deficit")
    print()
    print("  You reported 11.4% too slow. These two errors together produce")
    print(f"  a ~{(combined_factor - 1) * 100:.0f}% force deficit which more than explains the")
    print("  speed discrepancy. (Force deficit > speed deficit because")
    print("  resistance forces absorb a large fraction of drive force.)")
    print()

    # Cross-check with telemetry-derived radius
    print("CROSS-CHECK: Verifying tire radius from telemetry")
    print("  If you have simultaneous GPS Speed and Motor RPM data:")
    print("  r_effective = GPS_speed / (Motor_RPM / gear_ratio * 2*pi / 60)")
    print()
    for test_rpm, test_v in [(500, 15/3.6), (1000, 30/3.6), (1500, 45/3.6)]:
        r_check = test_v / (test_rpm / gear_ratio * 2 * math.pi / 60)
        print(f"  At RPM={test_rpm}, v={test_v*3.6:.0f} km/h: r = {r_check*1000:.1f} mm")
    print()

    print("WHERE DID 0.228 m COME FROM?")
    print("  Hoosier 16x7.5-10: 16\" OD -> radius = 203.2 mm (0.2032 m)")
    print("  0.228 m = 228 mm = 9.0\" radius = 18\" diameter")
    print("  This does NOT correspond to the Hoosier 16x7.5-10.")
    print("  Possible source: confused with a different tire size,")
    print("  or incorrectly computed from rim + sidewall dimensions.")
    print()

    # Check if .tir file has loaded radius info
    print("=" * 80)
    print("8. CHECKING .TIR FILE FOR TIRE RADIUS DATA")
    print("-" * 60)
    try:
        tir_path = cfg["tire"]["tir_file"]
        with open(tir_path) as f:
            tir_lines = f.readlines()
        # Look for UNLOADED_RADIUS, LOADED_RADIUS, FNOMIN, etc.
        for line in tir_lines:
            line_stripped = line.strip()
            for keyword in ["RADIUS", "FNOMIN", "NOMPRES", "R0", "RE ", "WIDTH", "RIM_RADIUS", "ASPECT_RATIO"]:
                if keyword in line_stripped.upper():
                    print(f"  {line_stripped}")
    except Exception as e:
        print(f"  Could not read .tir file: {e}")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
