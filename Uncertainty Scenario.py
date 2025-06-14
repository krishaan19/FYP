# === Import required libraries ===
import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# === Location and year settings ===
lat_wind = 52.42  # Wind location latitude
lon_wind = 4.13   # Wind location longitude
year = 2023
lat = 52.45       # Solar location latitude
lon = 4.6         # Solar location longitude

# === Fetch solar irradiance data for multiple tilt angles ===
tilt_angles = [0, 15, 30, 44, 60]
irradiance_data = {}
for tilt in tilt_angles:
    result = get_pvgis_hourly(
        latitude=lat,
        longitude=lon,
        start=year,
        end=year,
        raddatabase='PVGIS-SARAH3',
        surface_tilt=tilt,
        surface_azimuth=180,
        components=False,
        outputformat='json',
        usehorizon=True
    )
    df = result[0]
    irradiance_data[tilt] = df['poa_global']  # Extract plane-of-array global irradiance

# === Extract 24-hour irradiance values for June 21 ===
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)
irr_data_24h = {
    tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values
    for tilt in tilt_angles
}

# Dictionary R_S for irradiance per hour and tilt angle
n_hours = len(next(iter(irr_data_24h.values())))
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

# === Fetch hourly wind speed data ===
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

# === Wind turbine power curve and conversion function ===
total_wind_capacity_mw = 225  # Wind farm capacity (MW)
v_cut_in = 3.0    # Cut-in wind speed (m/s)
v_rated = 12.0    # Rated wind speed (m/s)
v_cut_out = 25.0  # Cut-out wind speed (m/s)

def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

# Wind power availability for each hour
P_wind_avail = {t: wind_power_available(wind_data_24h[t]) for t in range(n_hours)}

# === Time horizon ===
T = range(24)

# === Base load profile (MW) ===
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# === Generate 10 load uncertainty scenarios ===
num_scenarios = 10
scenario_probabilities = np.full(num_scenarios, 1 / num_scenarios)  # Equal weights
sigma = 0.05  # 5% standard deviation for perturbation

# Create stochastic scenarios by adding noise
load_scenarios = [
    load_demand + np.random.normal(0, sigma * load_demand)
    for _ in range(num_scenarios)
]

# === Carbon intensity profile (kgCO₂/MWh) ===
carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])

# === Split load into shiftable and non-shiftable ===
shiftable_load = 0.4 * load_demand  
non_shiftable_load = load_demand - shiftable_load

# === System configuration parameters ===
eta = 0.18  # PV efficiency
A_s = 1860000  # Total PV area (m²)
cost_conv = 80  # €/MWh conventional gen
emission_conv = 0.5  # kgCO₂/MWh
battery_capacity = 100
battery_charge_rate = 25
battery_eff = 0.95
ev_capacity = 201.25
ev_charge_rate = 30
ev_eff = 0.90
ev_initial_soc = 100
ev_min_soc = 40
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]  # EV connected overnight
alpha = 10  # Not used explicitly here
hourly_prices = np.array([
    119.70, 110.77, 104.80, 103.92, 103.76, 107.02,
    126.56, 137.67, 144.55, 126.16, 103.02, 91.16,
    89.12, 86.50, 86.64, 94.60, 100.16, 113.62,
    138.17, 167.50, 190.24, 180.66, 155.57, 134.86
])

# === Initialize Pyomo model ===
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)
model.S = pyo.Set(initialize=range(num_scenarios))

# === Parameters & Variables ===
# Load for each scenario
model.load_demand_scen = pyo.Param(model.S, model.T, initialize={
    (s, t): load_scenarios[s][t] for s in range(num_scenarios) for t in T
}, mutable=True)

# Scenario-dependent and common decision variables
model.P_conv_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.D_shift_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.P_solar = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.use_angle = pyo.Var(model.T, model.A, within=pyo.Binary)
model.P_wind = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_conv = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_batt_ch = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_batt_dis = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.SoC = pyo.Var(model.T, bounds=(0, battery_capacity))
model.P_EV_ch = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_EV_dis = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.SoC_EV = pyo.Var(model.T, bounds=(0, ev_capacity))
model.D_shift = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.Ramp = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_EV_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.beta = pyo.Param(initialize=0.5, mutable=True)
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)  # 1 = charging, 0 = discharging

# === Penalties ===
ramp_penalty = 10
shift_penalty_per_hour = np.array([
    20, 20, 20, 20, 20, 20, 20, 15,
    5, 2, 0, 0, 0, 0, 1, 3, 5, 8,
    15, 18, 20, 20, 20, 20
])

# === Objective: Expected cost + emissions + flexibility penalties ===
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
# === Constraint: EV cannot export more than it discharges ===
def ev_export_limit_rule(model, t):
    return model.P_EV_to_grid[t] <= model.P_EV_dis[t]
model.ev_export_limit = pyo.Constraint(model.T, rule=ev_export_limit_rule)

# === Constraint: Total shift must match total shiftable load ===
model.shift_balance = pyo.Constraint(
    expr=sum(model.D_shift[t] for t in model.T) == sum(shiftable_load)
)

# === Constraint: Per-hour shift limit ===
max_shiftable_per_hour = 1.5 * np.max(shiftable_load)
model.shift_limit = pyo.Constraint(
    model.T, rule=lambda m, t: m.D_shift[t] <= max_shiftable_per_hour
)

# === Constraints: Ramp penalties for load shifting ===
def ramp_up_rule(model, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.Ramp[t] >= model.D_shift[t] - model.D_shift[t-1]
model.ramp_up = pyo.Constraint(model.T, rule=ramp_up_rule)

def ramp_down_rule(model, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.Ramp[t] >= model.D_shift[t-1] - model.D_shift[t]
model.ramp_down = pyo.Constraint(model.T, rule=ramp_down_rule)

# === Constraint: Solar generation limited by irradiance and tilt angle ===
def solar_output_rule(model, t):
    return model.P_solar[t] <= (eta * A_s / 1_000_000) * sum(R_S[(t, a)] * model.use_angle[t, a] for a in model.A)
model.solar_output = pyo.Constraint(model.T, rule=solar_output_rule)

# === Constraint: Only one tilt angle can be used per hour ===
def one_angle_rule(model, t):
    return sum(model.use_angle[t, a] for a in model.A) == 1
model.one_angle = pyo.Constraint(model.T, rule=one_angle_rule)

# === Constraint: Wind power capped by hourly availability ===
model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_wind[t] <= P_wind_avail[t])

# === Scenario-wise power balance constraint ===
def power_balance_rule_s(model, s, t):
    return (
        model.P_solar[t] + model.P_wind[t] + model.P_conv_s[s, t] +
        model.P_batt_dis[t] + model.P_EV_dis[t] ==
        model.load_demand_scen[s, t] - shiftable_load[t] + model.D_shift_s[s, t] +
        model.P_batt_ch[t] + model.P_EV_ch[t]
    )
model.power_balance_s = pyo.Constraint(model.S, model.T, rule=power_balance_rule_s)

# === Scenario-wise shift total constraint ===
model.shift_balance_s = pyo.Constraint(
    model.S, rule=lambda m, s: sum(m.D_shift_s[s, t] for t in m.T) == sum(shiftable_load)
)

# === Scenario-wise hourly shift limit ===
model.shift_limit_s = pyo.Constraint(
    model.S, model.T,
    rule=lambda m, s, t: m.D_shift_s[s, t] <= max_shiftable_per_hour
)

# === Battery operational constraints ===
model.batt_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_ch[t] <= battery_charge_rate)
model.batt_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_dis[t] <= battery_charge_rate)

def soc_batt_rule(model, t):
    if t == 0:
        return model.SoC[t] == 20 + battery_eff * model.P_batt_ch[t] - (1/battery_eff) * model.P_batt_dis[t]
    return model.SoC[t] == model.SoC[t-1] + battery_eff * model.P_batt_ch[t-1] - (1/battery_eff) * model.P_batt_dis[t-1]
model.soc_batt = pyo.Constraint(model.T, rule=soc_batt_rule)

# === EV operational constraints ===
model.ev_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate)
model.ev_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate)
model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

# === Driving load during day (when EV is unavailable) ===
daily_ev_driving_energy = 3.0
driving_hours = [t for t in T if ev_available[t] == 0]
driving_load = np.zeros(len(T))
for t in driving_hours:
    driving_load[t] = daily_ev_driving_energy / len(driving_hours)

# === EV battery dynamics ===
def soc_ev_rule(model, t):
    if t == 0:
        return model.SoC_EV[t] == ev_initial_soc + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t] - driving_load[t]
    else:
        return model.SoC_EV[t] == model.SoC_EV[t-1] + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t] - driving_load[t]
model.soc_ev = pyo.Constraint(model.T, rule=soc_ev_rule)

model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC_EV[t] >= ev_min_soc)

# === Solve model ===
solver = pyo.SolverFactory('gurobi')
import time
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
print(f"\n⏱️ Model solved in {end_time - start_time:.2f} seconds.")

# === Extract results ===
from pyomo.environ import value
P_EV_ch = [pyo.value(model.P_EV_ch[t]) for t in T]
SoC = [pyo.value(model.SoC[t]) for t in T]
SoC_EV = [pyo.value(model.SoC_EV[t]) for t in T]
P_solar = [value(model.P_solar[t]) for t in T]
P_wind = [value(model.P_wind[t]) for t in T]
P_batt_dis = [value(model.P_batt_dis[t]) for t in T]
P_EV_dis = [value(model.P_EV_dis[t]) for t in T]
P_conv_avg = [sum(scenario_probabilities[s] * value(model.P_conv_s[s, t]) for s in range(num_scenarios)) for t in T]

# === Total energy per source over 24 hours ===
solar_energy = sum(P_solar)
wind_energy = sum(P_wind)
batt_dis_energy = sum(P_batt_dis)
ev_dis_energy = sum(P_EV_dis)
conv_energy = sum(
    scenario_probabilities[s] * sum(value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)

# === Print summary ===
print(f"SoC_EV[0] = {value(model.SoC_EV[0]):.2f}")
print(f"EV charge at t=0 = {value(model.P_EV_ch[0]):.2f}")
print(f"EV discharge at t=0 = {value(model.P_EV_dis[0]):.2f}")
print("🔋 Total Power Generated Over 24 Hours (MWh):")
print(f"  ☀️  Solar:           {solar_energy:.2f} MWh")
print(f"  🌬️  Wind:            {wind_energy:.2f} MWh")
print(f"  🔌 Battery Discharge: {batt_dis_energy:.2f} MWh")
print(f"  🚗 EV Discharge:      {ev_dis_energy:.2f} MWh")
print(f"  ⚡ Conventional:      {conv_energy:.2f} MWh (Expected)")

# === Compute cost and emissions ===
total_cost = sum(
    scenario_probabilities[s] * sum(hourly_prices[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)
total_emissions = sum(
    scenario_probabilities[s] * sum(carbon_intensity_kg_per_mwh[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)
total_emissions_tonnes = total_emissions / 1000

print("\n💰 Total Conventional Generation Cost: €{:.2f}".format(total_cost))
print("🌍 Total Emissions from Conventional Generation: {:.2f} kgCO₂".format(total_emissions))
print("🌍 Emissions in tonnes: {:.2f} tCO₂".format(total_emissions_tonnes))

# === Compute load statistics across scenarios ===
load_array = np.array(load_scenarios)
load_mean = np.mean(load_array, axis=0)
load_p5 = np.percentile(load_array, 5, axis=0)
load_p95 = np.percentile(load_array, 95, axis=0)

# === Plot: Generation profile vs load ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(T, P_solar, label="Solar", color='deepskyblue')
ax.fill_between(T, P_solar, color='deepskyblue', alpha=0.3)
ax.plot(T, P_wind, label="Wind", color='green')
ax.fill_between(T, P_wind, color='green', alpha=0.3)
ax.plot(T, P_conv_avg, label="Conventional", color='darkorange')
ax.fill_between(T, P_conv_avg, color='darkorange', alpha=0.3)
ax.plot(T, P_batt_dis, label="Battery Discharge", color='purple')
ax.fill_between(T, P_batt_dis, color='purple', alpha=0.3)
ax.plot(T, P_EV_dis, label="EV Discharge", color='red')
ax.fill_between(T, P_EV_dis, color='red', alpha=0.3)
ax.plot(T, load_mean, 'k--', label="Mean Load")
ax.fill_between(T, load_p5, load_p95, color='black', alpha=0.1, label='Load 90% Band')
ax.set_ylim(bottom=0)
ax.set_xlim(0, 23)
ax.set_xticks(np.arange(0, 24, 1))
ax.set_title("Power Generation (MW) with Load Demand Uncertainty")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/load_uncertainty_power_generation.png")
plt.show()

# === Plot: State of Charge ===
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(T, SoC, 'magenta', label="Battery SoC")
ax.plot(T, SoC_EV, 'cyan', label="EV SoC")
ax.set_ylim(bottom=0)
ax.set_xlim(0, 23)
ax.set_title("State of Charge")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Energy (MWh)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# === Plot: EV charging/discharging profile ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(T, P_EV_ch, label="EV Charging", color='blue')
ax.plot(T, P_EV_dis, label="EV Discharging (V2G)", color='red', linestyle='--')
ax.set_ylim(bottom=0)
ax.set_xlim(0, 23)
ax.set_title("EV Charging vs Discharging (V2G Profile)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
