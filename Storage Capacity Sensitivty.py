import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# -------------------------------
# Real solar and wind data fetching
lat = 52.45
lon = 4.6

year = 2023

# Solar data for multiple tilt angles
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
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

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

# Total wind capacity in kW
total_wind_capacity_mw = 225  # 15 MW

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


P_wind_avail = {t:wind_power_available(wind_data_24h[t]) for t in range(n_hours)}


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






# Shiftable and non-shiftable load
# shiftable_load = 0.4 * load_demand  # for example, 20% shiftable
# non_shiftable_load = load_demand - shiftable_load

# -------------------------------
# Parameters
eta = 0.18  # PV efficiency
A_s = 1860000   # PV area (m²)
cost_conv = 80
emission_conv = 0.5
battery_capacity = 100
battery_charge_rate = 80
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

carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])



carbon_price = 0.05       # €/kg CO2

# -------------------------------
# Pyomo Model
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)

# Variables
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
# model.D_shift = pyo.Var(model.T, within=pyo.NonNegativeReals)
# model.Ramp = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_EV_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.beta = pyo.Param(initialize=0.5, mutable=True)
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)  # 1 = charging, 0 = discharging




# 1. Parameters
# ramp_penalty = 10  # can increase to 20 if you want even smoother
# daytime_shift_penalty = 20
# nighttime_shift_penalty = 5

shift_penalty_per_hour = np.array([
    20, 20, 20, 20, 20, 20, 20, 15,  # 00:00–07:00
    5, 2, 0, 0, 0, 0, 1, 3, 5, 8,    # 08:00–17:00
    15, 18, 20, 20, 20, 20           # 18:00–23:00 → total = 24
])


# -------------------------------
# Objective


model.obj = pyo.Objective(
    expr=model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
         (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T),
    sense=pyo.minimize
)

# -------------------------------





# Constraints
# Only allow charging when EV_mode[t] == 1
model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])

# Only allow discharging when EV_mode[t] == 0
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

def ev_export_limit_rule(model, t):
    return model.P_EV_to_grid[t] <= model.P_EV_dis[t]

model.ev_export_limit = pyo.Constraint(model.T, rule=ev_export_limit_rule)



def solar_output_rule(model, t):
    return model.P_solar[t] <= (eta * A_s / 1_000_000)* sum(R_S[(t, a)] * model.use_angle[t, a] for a in model.A)
model.solar_output = pyo.Constraint(model.T, rule=solar_output_rule)

def one_angle_rule(model, t):
    return sum(model.use_angle[t, a] for a in model.A) == 1
model.one_angle = pyo.Constraint(model.T, rule=one_angle_rule)

model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_wind[t] <= P_wind_avail[t])
def power_balance_rule(model, t):
    return (model.P_solar[t] + model.P_wind[t] + model.P_conv[t] +
            model.P_batt_dis[t] + model.P_EV_dis[t] ==
            load_demand[t] + model.P_batt_ch[t] + model.P_EV_ch[t])
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)


# Battery charge/discharge limits
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

model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC_EV[t] >= ev_min_soc if ev_available[t] == 0 else pyo.Constraint.Skip)




solver = pyo.SolverFactory('gurobi')
pareto_cost = []
pareto_emissions = []
beta_values = np.linspace(0, 1, 21)

for b in beta_values:
    model.beta = b
    solver.solve(model, tee=False)
    
    cost = sum(hourly_prices[t] * pyo.value(model.P_conv[t]) for t in T)
    emissions = sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T)
    
    pareto_cost.append(cost)
    pareto_emissions.append(emissions)






storage_caps = list(range(80, 401, 40))  # 80, 120, ..., 400
storage_costs = []
storage_emissions = []

for cap in storage_caps:
    # Update battery and EV capacity
    model.SoC.setlb(0)
    model.SoC.setub(cap)
    model.SoC_EV.setlb(0)
    model.SoC_EV.setub(cap)

    # Solve with updated capacity
    results = solver.solve(model, tee=False)

    # Extract cost and emissions
    total_cost = sum(hourly_prices[t] * pyo.value(model.P_conv[t]) for t in T) / 1e6  # € mln
    total_emissions = sum(carbon_intensity_kg_per_mwh[t] * pyo.value(model.P_conv[t]) for t in T) / 1000  # kg mln

    storage_costs.append(total_cost)
    storage_emissions.append(total_emissions)


fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Battery/EV Storage Capacity (MWh)')
ax1.set_ylabel('Cost (€ mln)', color='blue')
ax1.plot(storage_caps, storage_costs, 'o-', color='blue', label='Cost')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('CO₂ Emissions (kg mln)', color='red')
ax2.plot(storage_caps, storage_emissions, 's--', color='red', label='Emissions')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Sensitivity of Cost and Emissions to Storage Capacity')
fig.tight_layout()
plt.grid(True)
plt.show()
