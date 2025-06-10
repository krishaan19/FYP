import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# -------------------------------
# Real solar and wind data fetching
lat_wind = 52.42
lon_wind = 4.13
year = 2023
lat = 52.45
lon = 4.6
# Solar data for multiple tilt angles


num_scenarios = 10
scenario_probabilities = np.full(num_scenarios, 1 / num_scenarios)
sigma = 0.05  # 5% std dev




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
    irradiance_data[tilt] = df['poa_global']
    
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)
irr_data_24h = {tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values for tilt in tilt_angles}

n_hours = len(next(iter(irr_data_24h.values())))


# === Irradiance scenarios ===
sigma_irr = 0.05  # 7% std deviation
bias_irr = -0.03  # 5% negative bias

irradiance_scenarios = {
    s: {
        a: np.clip(
            irr_data_24h[a] + np.random.normal(
                loc=bias_irr * irr_data_24h[a],
                scale=sigma_irr * irr_data_24h[a]
            ), 0, None
        )
        for a in tilt_angles
    }
    for s in range(num_scenarios)
}

R_S_scen = {
    (s, t, a): irradiance_scenarios[s][a][t]
    for s in range(num_scenarios)
    for t in range(24)
    for a in tilt_angles
}
# Wind data
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

sigma_wind = 0.05  # 10% std deviation
bias_wind = -0.03  # 7% negative bias

wind_speed_scenarios = [
    np.clip(
        wind_data_24h + np.random.normal(
            loc=bias_wind * wind_data_24h,
            scale=sigma_wind * wind_data_24h
        ), 0, None
    )
    for _ in range(num_scenarios)
]

# === Wind Power Scenarios using turbine curve ===

total_wind_capacity_mw = 225  # 225 MW



# Wind turbine power curve parameters (Vestas V90/3000)
v_cut_in = 3.0    # m/s
v_rated = 12.0    # m/s
v_cut_out = 25.0  # m/s


def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

P_wind_avail_scen = {
    (s, t): wind_power_available(wind_speed_scenarios[s][t])
    for s in range(num_scenarios)
    for t in range(24)
}

# -------------------------------
# Time horizon
T = range(24)
# -------------------------------
# Load profile (example)
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])
# Step 1: Scenario generation



load_scenarios = [
    load_demand + np.random.normal(0, sigma * load_demand)  # perturb each hour
    for _ in range(num_scenarios)
]

carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])
# Shiftable and non-shiftable load
shiftable_load = 0.4 * load_demand  
non_shiftable_load = load_demand - shiftable_load
# -------------------------------
# Parameters
eta = 0.18  # PV efficiency
A_s = 1860000   # PV area (m²)
cost_conv = 80
emission_conv = 0.5
battery_capacity = 100
battery_charge_rate = 25
battery_eff = 0.95
ev_capacity = 201.25
ev_charge_rate = 30
ev_eff = 0.90
ev_initial_soc = 100
ev_min_soc = 40
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]
alpha = 10
hourly_prices = np.array([
    119.70, 110.77, 104.80, 103.92, 103.76, 107.02,
    126.56, 137.67, 144.55, 126.16, 103.02, 91.16,
    89.12, 86.50, 86.64, 94.60, 100.16, 113.62,
    138.17, 167.50, 190.24, 180.66, 155.57, 134.86
])
# -------------------------------
# Pyomo Model
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)
model.S = pyo.Set(initialize=range(num_scenarios))
# Variables
model.load_demand_scen = pyo.Param(model.S, model.T, initialize={
    (s, t): load_scenarios[s][t] for s in range(num_scenarios) for t in T
}, mutable=True)
model.R_S_scen = pyo.Param(model.S, model.T, model.A, initialize=R_S_scen, mutable=True)
model.P_wind_avail_scen = pyo.Param(model.S, model.T, initialize=P_wind_avail_scen, mutable=True)
model.P_conv_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.D_shift_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.P_solar_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.P_wind_s = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.use_angle = pyo.Var(model.T, model.A, within=pyo.Binary)
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
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)

ramp_penalty = 10  
shift_penalty_per_hour = np.array([
    20, 20, 20, 20, 20, 20, 20, 15,  # 00:00–07:00
    5, 2, 0, 0, 0, 0, 1, 3, 5, 8,    # 08:00–17:00
    15, 18, 20, 20, 20, 20           # 18:00–23:00 → total = 24
])
# -------------------------------
# Objective
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

# -------------------------------

def ev_export_limit_rule(model, t):
    return model.P_EV_to_grid[t] <= model.P_EV_dis[t]

model.ev_export_limit = pyo.Constraint(model.T, rule=ev_export_limit_rule)  

model.shift_balance = pyo.Constraint(
    expr=sum(model.D_shift[t] for t in model.T) == sum(shiftable_load)
)

max_shiftable_per_hour = 1.5 * np.max(shiftable_load)  # or define a realistic MW limit

model.shift_limit = pyo.Constraint(
    model.T,
    rule=lambda m, t: m.D_shift[t] <= max_shiftable_per_hour
)
# Ramp Up Constraint
def ramp_up_rule(model, t):
    if t == 0:
        return pyo.Constraint.Skip  # No previous hour for t=0
    return model.Ramp[t] >= model.D_shift[t] - model.D_shift[t-1]
model.ramp_up = pyo.Constraint(model.T, rule=ramp_up_rule)
# Ramp Down Constraint
def ramp_down_rule(model, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.Ramp[t] >= model.D_shift[t-1] - model.D_shift[t]
model.ramp_down = pyo.Constraint(model.T, rule=ramp_down_rule)

def solar_output_s_rule(model, s, t):
    return model.P_solar_s[s, t] <= (eta * A_s / 1_000_000) * sum(
        model.R_S_scen[s, t, a] * model.use_angle[t, a] for a in model.A
    )
model.solar_output_s = pyo.Constraint(model.S, model.T, rule=solar_output_s_rule)


def one_angle_rule(model, t):
    return sum(model.use_angle[t, a] for a in model.A) == 1
model.one_angle = pyo.Constraint(model.T, rule=one_angle_rule)

def wind_limit_s_rule(model, s, t):
    return model.P_wind_s[s, t] <= model.P_wind_avail_scen[s, t]
model.wind_limit_s = pyo.Constraint(model.S, model.T, rule=wind_limit_s_rule)


def power_balance_rule_s(model, s, t):
    return (
        model.P_solar_s[s, t] + model.P_wind_s[s, t] + model.P_conv_s[s, t] +
        model.P_batt_dis[t] + model.P_EV_dis[t]
        ==
        model.load_demand_scen[s, t] - shiftable_load[t] + model.D_shift_s[s, t] +
        model.P_batt_ch[t] + model.P_EV_ch[t]
    )
model.power_balance_s = pyo.Constraint(model.S, model.T, rule=power_balance_rule_s)



model.shift_balance_s = pyo.Constraint(
    model.S,
    rule=lambda m, s: sum(m.D_shift_s[s, t] for t in m.T) == sum(shiftable_load)
)

model.shift_limit_s = pyo.Constraint(
    model.S, model.T,
    rule=lambda m, s, t: m.D_shift_s[s, t] <= max_shiftable_per_hour
)


model.batt_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_ch[t] <= battery_charge_rate)
model.batt_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_dis[t] <= battery_charge_rate)

# Battery SoC dynamics
def soc_batt_rule(model, t):
    if t == 0:
        return model.SoC[t] == 20 + battery_eff * model.P_batt_ch[t] - (1/battery_eff) * model.P_batt_dis[t]
    return model.SoC[t] == model.SoC[t-1] + battery_eff * model.P_batt_ch[t-1] - (1/battery_eff) * model.P_batt_dis[t-1]
model.soc_batt = pyo.Constraint(model.T, rule=soc_batt_rule)

model.ev_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate)
model.ev_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate)

model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

def soc_ev_rule(model, t):
    if t == 0:
        return model.SoC_EV[t] == ev_initial_soc + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t]
    else:
        return model.SoC_EV[t] == model.SoC_EV[t-1] + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t]
model.soc_ev = pyo.Constraint(model.T, rule=soc_ev_rule)

model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC_EV[t] >= ev_min_soc)

# -------------------------------
# Solve
solver = pyo.SolverFactory('gurobi')
import time
start_time = time.time()

# Solve the model
results = solver.solve(model, tee=True)

# After solve
end_time = time.time()
print(f"\n⏱️ Model solved in {end_time - start_time:.2f} seconds.")



from pyomo.environ import value

# === Extract expected values ===
P_EV_ch = [value(model.P_EV_ch[t]) for t in T]
SoC = [value(model.SoC[t]) for t in T]
SoC_EV = [value(model.SoC_EV[t]) for t in T]
P_batt_dis = [value(model.P_batt_dis[t]) for t in T]
P_EV_dis = [value(model.P_EV_dis[t]) for t in T]

# NEW: Average solar and wind over scenarios
P_solar = [sum(scenario_probabilities[s] * value(model.P_solar_s[s, t]) for s in range(num_scenarios)) for t in T]
P_wind = [sum(scenario_probabilities[s] * value(model.P_wind_s[s, t]) for s in range(num_scenarios)) for t in T]
P_conv_avg = [sum(scenario_probabilities[s] * value(model.P_conv_s[s, t]) for s in range(num_scenarios)) for t in T]

# === Calculate total energy (MWh) from each source over 24 hours ===
solar_energy = sum(P_solar)
wind_energy = sum(P_wind)
batt_dis_energy = sum(P_batt_dis)
ev_dis_energy = sum(P_EV_dis)
conv_energy = sum(P_conv_avg)

# === Print Results ===
print("🔋 Total Power Generated Over 24 Hours (MWh):")
print(f"  ☀️  Solar:           {solar_energy:.2f} MWh")
print(f"  🌬️  Wind:            {wind_energy:.2f} MWh")
print(f"  🔌 Battery Discharge: {batt_dis_energy:.2f} MWh")
print(f"  🚗 EV Discharge:      {ev_dis_energy:.2f} MWh")
print(f"  ⚡ Conventional:      {conv_energy:.2f} MWh (Expected)")
total_cost = sum(
    scenario_probabilities[s] *
    sum(hourly_prices[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)

total_emissions = sum(
    scenario_probabilities[s] *
    sum(carbon_intensity_kg_per_mwh[t] * value(model.P_conv_s[s, t]) for t in T)
    for s in range(num_scenarios)
)

# Optional: Normalize emissions to tons if needed
total_emissions_tonnes = total_emissions / 1000

print("\n💰 Total Conventional Generation Cost: €{:.2f}".format(total_cost))
print("🌍 Total Emissions from Conventional Generation: {:.2f} kgCO₂".format(total_emissions))
print("🌍 Emissions in tonnes: {:.2f} tCO₂".format(total_emissions_tonnes))
# # === Load statistics ===
# load_array = np.array(load_scenarios)  # shape: (num_scenarios, 24)
# load_mean = np.mean(load_array, axis=0)
# load_p5 = np.percentile(load_array, 5, axis=0)
# load_p95 = np.percentile(load_array, 95, axis=0) 


# solar_array = np.array([
#     [value(model.P_solar_s[s, t]) for t in T]
#     for s in range(num_scenarios)
# ])

# solar_mean = np.mean(solar_array, axis=0)
# solar_p5 = np.percentile(solar_array, 5, axis=0)
# solar_p95 = np.percentile(solar_array, 95, axis=0) 

# wind_array = np.array([
#     [value(model.P_wind_s[s, t]) for t in T]
#     for s in range(num_scenarios)
# ])
# wind_mean = np.mean(wind_array, axis=0)
# wind_p5 = np.percentile(wind_array, 5, axis=0)
# wind_p95 = np.percentile(wind_array, 95, axis=0)
# # === Plot Power Generation ===
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(T, solar_mean, label="Solar", color='deepskyblue')
# ax.fill_between(T, solar_mean, color='deepskyblue', alpha=0.3)
# ax.fill_between(T, solar_p5, solar_p95, color='blue', alpha=0.1, label='Solar 90% Band')

# ax.plot(T, wind_mean, label="Wind", color='green')
# ax.fill_between(T, wind_mean, color='green', alpha=0.3)

# ax.fill_between(T, wind_p5, wind_p95, color='green', alpha=0.3)
# ax.plot(T, P_conv_avg, label="Conventional", color='darkorange')
# ax.fill_between(T, P_conv_avg, color='darkorange', alpha=0.3)

# ax.plot(T, P_batt_dis, label="Battery Discharge", color='purple')
# ax.fill_between(T, P_batt_dis, color='purple', alpha=0.3)

# ax.plot(T, P_EV_dis, label="EV Discharge", color='red')
# ax.fill_between(T, P_EV_dis, color='red', alpha=0.3)

# ax.plot(T, load_mean, 'k--', label="Mean Load")
# ax.fill_between(T, load_p5, load_p95, color='black', alpha=0.1, label='Load 90% Band')

# ax.set_ylim(bottom=0)
# ax.margins(y=0)
# ax.set_xlim(left=0, right=23)
# ax.set_xticks(np.arange(0, 24, 1))
# ax.set_title("Power Generation (MW) with Load, Solar, Wind Uncertainty")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Power (MW)")
# ax.legend()
# ax.grid(True, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("/Users/krishaan/Documents/FYP/Results/all_uncertainty_power_generation.png")
# plt.show()

# # === State of Charge Plot ===
# fig, ax = plt.subplots(figsize=(10, 5))
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# ax.plot(T, SoC, 'magenta', label="Battery SoC")
# ax.plot(T, SoC_EV, 'cyan', label="EV SoC")
# ax.set_ylim(bottom=0)
# ax.set_xlim(left=0, right=23)
# ax.set_title("State of Charge")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Energy (MWh)")
# ax.legend()
# ax.grid(True, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

# # === EV Charging Profile ===
# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')
# ax.plot(T, P_EV_ch, label="EV Charging", color='blue')
# ax.plot(T, P_EV_dis, label="EV Discharging (V2G)", color='red', linestyle='--')
# ax.set_ylim(bottom=0)
# ax.set_xlim(left=0, right=23)
# ax.set_title("EV Charging vs Discharging (V2G Profile)")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Power (MW)")
# ax.legend()
# ax.grid(True, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
