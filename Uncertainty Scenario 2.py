import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# --------------------------------------
# 🌍 SETTING: Define location and year
lat_wind = 52.42
lon_wind = 4.13
lat = 52.45
lon = 4.6
year = 2023

# --------------------------------------
# 🎲 SCENARIOS: Setup for uncertainty modeling
num_scenarios = 10  # Number of uncertainty scenarios
scenario_probabilities = np.full(num_scenarios, 1 / num_scenarios)  # Equal probability
sigma = 0.05  # Standard deviation for load variability (5%)

# --------------------------------------
# ☀️ SOLAR DATA: Get hourly irradiance for multiple tilt angles
tilt_angles = [0, 15, 30, 44, 60]
irradiance_data = {}

for tilt in tilt_angles:
    result = get_pvgis_hourly(
        latitude=lat,
        longitude=lon,
        start=year,
        end=year,
        raddatabase='PVGIS-SARAH3',  # Satellite-based irradiance
        surface_tilt=tilt,
        surface_azimuth=180,
        components=False,
        outputformat='json',
        usehorizon=True
    )
    df = result[0]
    irradiance_data[tilt] = df['poa_global']  # POA = Plane of array irradiance

# Extract one full 24-hour day: June 21 (summer solstice)
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)
irr_data_24h = {
    tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values
    for tilt in tilt_angles
}
n_hours = len(next(iter(irr_data_24h.values())))  # Should be 24

# --------------------------------------
# ☀️ SCENARIOS: Create irradiance scenarios with bias and noise
sigma_irr = 0.05  # Std dev for irradiance variation
bias_irr = -0.03  # Slight negative bias to simulate uncertainty

irradiance_scenarios = {
    s: {
        a: np.clip(
            irr_data_24h[a] + np.random.normal(
                loc=bias_irr * irr_data_24h[a],  # mean bias
                scale=sigma_irr * irr_data_24h[a]  # relative noise
            ), 0, None  # clip to keep values ≥ 0
        )
        for a in tilt_angles
    }
    for s in range(num_scenarios)
}

# Flatten to 3D dict for Pyomo use
R_S_scen = {
    (s, t, a): irradiance_scenarios[s][a][t]
    for s in range(num_scenarios)
    for t in range(24)
    for a in tilt_angles
}

# --------------------------------------
# 🌬️ WIND DATA: Fetch hourly wind speed
result_wind = get_pvgis_hourly(
    latitude=lat,
    longitude=lon,
    start=year,
    end=year,
    raddatabase='PVGIS-ERA5',
    surface_tilt=30,
    surface_azimuth=180,
    components=True,
    outputformat='json',
    usehorizon=True
)
df_wind = result_wind[0]
wind_data = df_wind['wind_speed']
wind_data_24h = wind_data.loc[june_21:june_21 + pd.Timedelta("1D")].values

# WIND SCENARIOS: apply noise and bias
sigma_wind = 0.05
bias_wind = -0.03
wind_speed_scenarios = [
    np.clip(
        wind_data_24h + np.random.normal(
            loc=bias_wind * wind_data_24h,
            scale=sigma_wind * wind_data_24h
        ), 0, None
    )
    for _ in range(num_scenarios)
]

# WIND POWER SCENARIOS using Vestas V90 curve
total_wind_capacity_mw = 225  # 225 MW installed capacity

# Turbine operational parameters
v_cut_in = 3.0    # m/s: wind starts producing power
v_rated = 12.0    # m/s: turbine produces max power
v_cut_out = 25.0  # m/s: turbine shuts off

# Power curve function
def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

# Apply turbine power curve to wind scenarios
P_wind_avail_scen = {
    (s, t): wind_power_available(wind_speed_scenarios[s][t])
    for s in range(num_scenarios)
    for t in range(24)
}
# -------------------------------
# 🕓 TIME HORIZON: 24-hour day
T = range(24)

# -------------------------------
# ⚡ LOAD PROFILE: Example hourly base load
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# 🌀 LOAD SCENARIOS: Add noise to load profile for each scenario
load_scenarios = [
    load_demand + np.random.normal(0, sigma * load_demand)  # Add 5% noise
    for _ in range(num_scenarios)
]

# 🌍 CARBON INTENSITY PROFILE (kg CO₂ per MWh)
carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])

# 🔄 SPLIT LOAD into shiftable and non-shiftable components
shiftable_load = 0.4 * load_demand       # 40% is flexible
non_shiftable_load = load_demand - shiftable_load  # 60% must be met as-is

# -------------------------------
# 🔧 TECHNICAL PARAMETERS
eta = 0.18             # PV efficiency
A_s = 1860000          # Solar panel area in m²
cost_conv = 80         # Placeholder for conventional gen cost
emission_conv = 0.5    # Placeholder for emissions per unit energy
battery_capacity = 100             # MWh
battery_charge_rate = 25           # Max charge/discharge power in MW
battery_eff = 0.95                 # Round-trip efficiency
ev_capacity = 201.25              # MWh EV battery
ev_charge_rate = 30              # MW max EV charge/discharge
ev_eff = 0.90                    # EV round-trip efficiency
ev_initial_soc = 100            # MWh
ev_min_soc = 40                 # MWh min at any time
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]  # Only home at night/morning

# Pricing and penalty settings
alpha = 10  # (unused?)
hourly_prices = np.array([
    119.70, 110.77, 104.80, 103.92, 103.76, 107.02,
    126.56, 137.67, 144.55, 126.16, 103.02, 91.16,
    89.12, 86.50, 86.64, 94.60, 100.16, 113.62,
    138.17, 167.50, 190.24, 180.66, 155.57, 134.86
])

# -------------------------------
# ⚙️ CREATE PYOMO MODEL
model = pyo.ConcreteModel()

# SETS
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)
model.S = pyo.Set(initialize=range(num_scenarios))

# -------------------------------
# 🔣 PARAMETERS
# Load per scenario per time
model.load_demand_scen = pyo.Param(model.S, model.T, initialize={
    (s, t): load_scenarios[s][t] for s in range(num_scenarios) for t in T
}, mutable=True)

# Irradiance and wind power available per scenario
model.R_S_scen = pyo.Param(model.S, model.T, model.A, initialize=R_S_scen, mutable=True)
model.P_wind_avail_scen = pyo.Param(model.S, model.T, initialize=P_wind_avail_scen, mutable=True)

# -------------------------------
# 🔧 DECISION VARIABLES

# Per-scenario generation and load shifting
model.P_conv_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)     # Conventional generation
model.D_shift_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)    # Shifted load
model.P_solar_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)    # Solar power
model.P_wind_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)     # Wind power

# Angle selection (only one tilt per hour)
model.use_angle = pyo.Var(model.T, model.A, within=pyo.Binary)

# Shared variables (same across all scenarios)
model.P_conv = pyo.Var(model.T, within=pyo.NonNegativeReals)         # Conv. gen reference
model.P_batt_ch = pyo.Var(model.T, within=pyo.NonNegativeReals)      # Battery charging
model.P_batt_dis = pyo.Var(model.T, within=pyo.NonNegativeReals)     # Battery discharging
model.SoC = pyo.Var(model.T, bounds=(0, battery_capacity))           # Battery state of charge

model.P_EV_ch = pyo.Var(model.T, within=pyo.NonNegativeReals)        # EV charging
model.P_EV_dis = pyo.Var(model.T, within=pyo.NonNegativeReals)       # EV discharging
model.SoC_EV = pyo.Var(model.T, bounds=(0, ev_capacity))             # EV battery state of charge

model.D_shift = pyo.Var(model.T, within=pyo.NonNegativeReals)        # Total shifted load
model.Ramp = pyo.Var(model.T, within=pyo.NonNegativeReals)           # Ramp movement
model.P_EV_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)   # EV V2G export
model.beta = pyo.Param(initialize=0.5, mutable=True)                 # Cost-vs-emissions weighting
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)                  # EV mode: 1=charge, 0=discharge


# 🔧 Penalties for flexibility
ramp_penalty = 10
shift_penalty_per_hour = np.array([
    20, 20, 20, 20, 20, 20, 20, 15,  # 00:00–07:00
    5, 2, 0, 0, 0, 0, 1, 3, 5, 8,    # 08:00–17:00
    15, 18, 20, 20, 20, 20           # 18:00–23:00
])

# 🎯 OBJECTIVE FUNCTION: Minimize weighted cost + emissions + penalties
model.obj = pyo.Objective(
    expr=sum(
        scenario_probabilities[s] * (
            model.beta * sum(hourly_prices[t] * model.P_conv_s[s, t] for t in T) +
            (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv_s[s, t] for t in T) +
            ramp_penalty * sum((model.D_shift_s[s, t] - model.D_shift_s[s, t - 1])**2 for t in T if t > 0) +
            sum(shift_penalty_per_hour[t] * model.D_shift_s[s, t] for t in T)
        )
        for s in model.S
    ),
    sense=pyo.minimize
)

# ⚖️ EV V2G limit
model.ev_export_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_to_grid[t] <= m.P_EV_dis[t])

# 🌀 Load shifting balance constraint (shared across scenarios)
model.shift_balance = pyo.Constraint(expr=sum(model.D_shift[t] for t in model.T) == sum(shiftable_load))

# 🔒 Shift limit per hour
max_shiftable_per_hour = 1.5 * np.max(shiftable_load)
model.shift_limit = pyo.Constraint(model.T, rule=lambda m, t: m.D_shift[t] <= max_shiftable_per_hour)

# 🔁 Ramp constraints
model.ramp_up = pyo.Constraint(model.T, rule=lambda m, t: pyo.Constraint.Skip if t == 0 else m.Ramp[t] >= m.D_shift[t] - m.D_shift[t - 1])
model.ramp_down = pyo.Constraint(model.T, rule=lambda m, t: pyo.Constraint.Skip if t == 0 else m.Ramp[t] >= m.D_shift[t - 1] - m.D_shift[t])

# ☀️ Solar output constrained by angle and irradiance
def solar_output_s_rule(model, s, t):
    return model.P_solar_s[s, t] <= (eta * A_s / 1_000_000) * sum(model.R_S_scen[s, t, a] * model.use_angle[t, a] for a in model.A)
model.solar_output_s = pyo.Constraint(model.S, model.T, rule=solar_output_s_rule)

# ☀️ Enforce only one tilt angle per hour
model.one_angle = pyo.Constraint(model.T, rule=lambda m, t: sum(m.use_angle[t, a] for a in m.A) == 1)

# 🌬️ Wind constrained by scenario availability
model.wind_limit_s = pyo.Constraint(model.S, model.T, rule=lambda m, s, t: m.P_wind_s[s, t] <= m.P_wind_avail_scen[s, t])

# ⚡ Power balance per scenario: supply = demand + losses + charging
def power_balance_rule_s(model, s, t):
    return (
        model.P_solar_s[s, t] + model.P_wind_s[s, t] + model.P_conv_s[s, t] +
        model.P_batt_dis[t] + model.P_EV_dis[t]
        ==
        model.load_demand_scen[s, t] - shiftable_load[t] + model.D_shift_s[s, t] +
        model.P_batt_ch[t] + model.P_EV_ch[t]
    )
model.power_balance_s = pyo.Constraint(model.S, model.T, rule=power_balance_rule_s)

# 🔄 Shifting constraints per scenario
model.shift_balance_s = pyo.Constraint(model.S, rule=lambda m, s: sum(m.D_shift_s[s, t] for t in m.T) == sum(shiftable_load))
model.shift_limit_s = pyo.Constraint(model.S, model.T, rule=lambda m, s, t: m.D_shift_s[s, t] <= max_shiftable_per_hour)

# 🔋 Battery charging/discharging limits
model.batt_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_ch[t] <= battery_charge_rate)
model.batt_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_dis[t] <= battery_charge_rate)

# 🔋 Battery state-of-charge dynamics
def soc_batt_rule(model, t):
    if t == 0:
        return model.SoC[t] == 20 + battery_eff * model.P_batt_ch[t] - (1 / battery_eff) * model.P_batt_dis[t]
    return model.SoC[t] == model.SoC[t - 1] + battery_eff * model.P_batt_ch[t - 1] - (1 / battery_eff) * model.P_batt_dis[t - 1]
model.soc_batt = pyo.Constraint(model.T, rule=soc_batt_rule)

# 🚗 EV limits and dynamics
model.ev_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate)
model.ev_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate)
model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

# EV state-of-charge
def soc_ev_rule(model, t):
    if t == 0:
        return model.SoC_EV[t] == ev_initial_soc + ev_eff * model.P_EV_ch[t] - (1 / ev_eff) * model.P_EV_dis[t]
    return model.SoC_EV[t] == model.SoC_EV[t - 1] + ev_eff * model.P_EV_ch[t] - (1 / ev_eff) * model.P_EV_dis[t]
model.soc_ev = pyo.Constraint(model.T, rule=soc_ev_rule)

# Maintain minimum SoC for EV
model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC_EV[t] >= ev_min_soc)
# 🧠 Solve using Gurobi
solver = pyo.SolverFactory('gurobi')
import time
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
print(f"\n⏱️ Model solved in {end_time - start_time:.2f} seconds.")

# 📦 Extract decision variable values
from pyomo.environ import value
P_EV_ch = [value(model.P_EV_ch[t]) for t in T]
SoC = [value(model.SoC[t]) for t in T]
SoC_EV = [value(model.SoC_EV[t]) for t in T]
P_batt_dis = [value(model.P_batt_dis[t]) for t in T]
P_EV_dis = [value(model.P_EV_dis[t]) for t in T]

# 🧮 Expected power (mean across scenarios)
P_solar = [sum(scenario_probabilities[s] * value(model.P_solar_s[s, t]) for s in range(num_scenarios)) for t in T]
P_wind = [sum(scenario_probabilities[s] * value(model.P_wind_s[s, t]) for s in range(num_scenarios)) for t in T]
P_conv_avg = [sum(scenario_probabilities[s] * value(model.P_conv_s[s, t]) for s in range(num_scenarios)) for t in T]

# 🔋 Compute total MWh per source
solar_energy = sum(P_solar)
wind_energy = sum(P_wind)
batt_dis_energy = sum(P_batt_dis)
ev_dis_energy = sum(P_EV_dis)
conv_energy = sum(P_conv_avg)

# 🧾 Print energy summary
print("🔋 Total Power Generated Over 24 Hours (MWh):")
print(f"  ☀️  Solar:           {solar_energy:.2f} MWh")
print(f"  🌬️  Wind:            {wind_energy:.2f} MWh")
print(f"  🔌 Battery Discharge: {batt_dis_energy:.2f} MWh")
print(f"  🚗 EV Discharge:      {ev_dis_energy:.2f} MWh")
print(f"  ⚡ Conventional:      {conv_energy:.2f} MWh (Expected)")

# 💰 Total cost and emissions from conventional generation
total_cost = sum(
    scenario_probabilities[s] * sum(hourly_prices[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)
total_emissions = sum(
    scenario_probabilities[s] * sum(carbon_intensity_kg_per_mwh[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)
total_emissions_tonnes = total_emissions / 1000  # Convert to tonnes

print("\n💰 Total Cost: €{:.2f}".format(total_cost))
print("🌍 Emissions: {:.2f} kgCO₂ | {:.2f} tCO₂".format(total_emissions, total_emissions_tonnes))
# 📈 Generate load statistics across scenarios for uncertainty visualization
load_array = np.array(load_scenarios)  # Shape: (num_scenarios, 24)
load_mean = np.mean(load_array, axis=0)
load_p5 = np.percentile(load_array, 5, axis=0)
load_p95 = np.percentile(load_array, 95, axis=0)

# 📈 Compute solar generation statistics
solar_array = np.array([[value(model.P_solar_s[s, t]) for t in T] for s in range(num_scenarios)])
solar_mean = np.mean(solar_array, axis=0)
solar_p5 = np.percentile(solar_array, 5, axis=0)
solar_p95 = np.percentile(solar_array, 95, axis=0)

# 📈 Compute wind generation statistics
wind_array = np.array([[value(model.P_wind_s[s, t]) for t in T] for s in range(num_scenarios)])
wind_mean = np.mean(wind_array, axis=0)
wind_p5 = np.percentile(wind_array, 5, axis=0)
wind_p95 = np.percentile(wind_array, 95, axis=0)

# 🔍 Plot power generation mix with uncertainty bands
fig, ax = plt.subplots(figsize=(12, 6))

# Plot solar generation
ax.plot(T, solar_mean, label="Solar", color='deepskyblue')
ax.fill_between(T, solar_mean, color='deepskyblue', alpha=0.3)
ax.fill_between(T, solar_p5, solar_p95, color='blue', alpha=0.1, label='Solar 90% Band')

# Plot wind generation
ax.plot(T, wind_mean, label="Wind", color='green')
ax.fill_between(T, wind_mean, color='green', alpha=0.3)
ax.fill_between(T, wind_p5, wind_p95, color='green', alpha=0.1, label='Wind 90% Band')

# Plot conventional generation (average over scenarios)
ax.plot(T, P_conv_avg, label="Conventional", color='darkorange')
ax.fill_between(T, P_conv_avg, color='darkorange', alpha=0.3)

# Plot battery discharge
ax.plot(T, P_batt_dis, label="Battery Discharge", color='purple')
ax.fill_between(T, P_batt_dis, color='purple', alpha=0.3)

# Plot EV discharge
ax.plot(T, P_EV_dis, label="EV Discharge", color='red')
ax.fill_between(T, P_EV_dis, color='red', alpha=0.3)

# Load with uncertainty band
ax.plot(T, load_mean, 'k--', label="Mean Load")
ax.fill_between(T, load_p5, load_p95, color='black', alpha=0.1, label='Load 90% Band')

# Final styling
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=23)
ax.set_xticks(np.arange(0, 24, 1))
ax.set_title("Power Generation (MW) with Load, Solar, Wind Uncertainty")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/all_uncertainty_power_generation.png")
plt.show()

# 🔋 Plot state of charge for battery and EV over time
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Battery and EV SoC
ax.plot(T, SoC, 'magenta', label="Battery SoC")
ax.plot(T, SoC_EV, 'cyan', label="EV SoC")

# Styling
ax.set_ylim(bottom=0)
ax.set_xlim(left=0, right=23)
ax.set_title("State of Charge")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Energy (MWh)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
