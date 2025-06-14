

import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# -------------------------------
# Fetch real solar and wind data for a specific location
lat = 52.45
lon = 4.6
year = 2023

# Retrieve solar irradiance data for various tilt angles
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
    irradiance_data[tilt] = df['poa_global']  # Store plane-of-array irradiance

# Get timezone info and extract irradiance for June 21 (typical peak solar day)
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)

# Get 24-hour irradiance data for each tilt angle
irr_data_24h = {tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values for tilt in tilt_angles}
n_hours = len(next(iter(irr_data_24h.values())))

# Dictionary: irradiance at each hour for each angle
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

# -------------------------------
# Fetch wind data using PVGIS ERA5 dataset
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

# Total wind turbine capacity in kW
total_wind_capacity_mw = 225  # 15 turbines of 15 MW total

# Wind turbine operational parameters
v_cut_in = 3.0
v_rated = 12.0
v_cut_out = 25.0

# Power curve for wind turbine
def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

# Hourly available wind power
P_wind_avail = {t: wind_power_available(wind_data_24h[t]) for t in range(n_hours)}

# -------------------------------
# Time index for 24 hours
T = range(24)

# -------------------------------
# Hourly electricity demand (MW)
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# -------------------------------
# System parameters
eta = 0.18                # PV panel efficiency
A_s = 1860000             # Total solar panel area (m²)
cost_conv = 80            # Conventional generation cost per MWh (placeholder, unused)
emission_conv = 0.5       # CO₂ emission per MWh for conventional generation (placeholder, unused)
battery_capacity = 100    # Battery energy capacity (MWh)
battery_charge_rate = 25  # Max charge/discharge rate (MW)
battery_eff = 0.95        # Round-trip efficiency
ev_capacity = 201.25      # EV fleet capacity (MWh)
ev_charge_rate = 30       # Max charge/discharge rate (MW)
ev_eff = 0.90             # Round-trip efficiency
ev_initial_soc = 100      # Initial SoC (MWh)
ev_min_soc = 40           # Minimum SoC when EVs are away
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]  # EVs only available during night
alpha = 10                # Weight for penalty (unused)
hourly_prices = np.array([
    119.70, 110.77, 104.80, 103.92, 103.76, 107.02,
    126.56, 137.67, 144.55, 126.16, 103.02, 91.16,
    89.12, 86.50, 86.64, 94.60, 100.16, 113.62,
    138.17, 167.50, 190.24, 180.66, 155.57, 134.86
])

# CO₂ intensity per MWh of grid power
carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])

carbon_price = 0.05       # €/kg CO2

# -------------------------------
# Pyomo Model Definition
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)

# -------------------------------
# Decision Variables
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
model.P_EV_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.beta = pyo.Param(initialize=0.5, mutable=True)  # Weighting parameter for cost vs emissions
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)  # 1 = charging mode, 0 = discharging

# -------------------------------
# Objective: Weighted cost and emissions minimization
model.obj = pyo.Objective(
    expr=model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
         (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T),
    sense=pyo.minimize
)

# -------------------------------
# Constraints

# EV charging only allowed in charging mode
model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])

# EV discharging only allowed in discharging mode
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

# EV can only export energy if discharging
model.ev_export_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_to_grid[t] <= m.P_EV_dis[t])

# Solar power capped by irradiance at selected tilt angle
model.solar_output = pyo.Constraint(model.T, rule=lambda m, t: m.P_solar[t] <= (eta * A_s / 1_000_000)* sum(R_S[(t, a)] * m.use_angle[t, a] for a in m.A))

# Only one tilt angle used at each hour
model.one_angle = pyo.Constraint(model.T, rule=lambda m, t: sum(m.use_angle[t, a] for a in m.A) == 1)

# Wind power cannot exceed available
model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_wind[t] <= P_wind_avail[t])

# Power balance: Supply = Demand
model.power_balance = pyo.Constraint(model.T, rule=lambda m, t: (
    m.P_solar[t] + m.P_wind[t] + m.P_conv[t] + m.P_batt_dis[t] + m.P_EV_dis[t] ==
    load_demand[t] + m.P_batt_ch[t] + m.P_EV_ch[t]
))

# Battery charge/discharge rate limits
model.batt_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_ch[t] <= battery_charge_rate)
model.batt_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_dis[t] <= battery_charge_rate)

# Battery state of charge dynamics
def soc_batt_rule(m, t):
    if t == 0:
        return m.SoC[t] == 20 + battery_eff * m.P_batt_ch[t] - (1/battery_eff) * m.P_batt_dis[t]
    return m.SoC[t] == m.SoC[t-1] + battery_eff * m.P_batt_ch[t-1] - (1/battery_eff) * m.P_batt_dis[t-1]
model.soc_batt = pyo.Constraint(model.T, rule=soc_batt_rule)

# EV SoC dynamics
def soc_ev_rule(model, t):
    if t == 0:
        return model.SoC_EV[t] == ev_initial_soc + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t]
    else:
        return model.SoC_EV[t] == model.SoC_EV[t-1] + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t]
model.soc_ev = pyo.Constraint(model.T, rule=soc_ev_rule)

# Minimum EV SoC when away from home
model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC_EV[t] >= ev_min_soc if ev_available[t] == 0 else pyo.Constraint.Skip)

# -------------------------------
# Pareto Frontier Calculation
solver = pyo.SolverFactory('gurobi')
pareto_cost = []
pareto_emissions = []
beta_values = np.linspace(0, 1, 21)

for b in beta_values:
    model.beta = b
    solver.solve(model, tee=False)
    
    cost = sum(hourly_prices[t] * pyo.value(model.P_conv[t]) for t in T)
    emissions = sum(carbon_intensity_kg_per_mwh[t] * pyo.value(model.P_conv[t]) for t in model.T)
    
    pareto_cost.append(cost)
    pareto_emissions.append(emissions)

# -------------------------------
# Final model solve with selected beta
solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# Extract optimized variable values for plotting
P_solar = [pyo.value(model.P_solar[t]) for t in T]
P_wind = [pyo.value(model.P_wind[t]) for t in T]
P_conv = [pyo.value(model.P_conv[t]) for t in T]
SoC = [pyo.value(model.SoC[t]) for t in T]
SoC_EV = [pyo.value(model.SoC_EV[t]) for t in T]
P_EV_ch = [pyo.value(model.P_EV_ch[t]) for t in T]
P_EV_dis = [pyo.value(model.P_EV_dis[t]) for t in T]
P_batt_dis = [pyo.value(model.P_batt_dis[t]) for t in T]

# -------------------------------
# Plot: Power generation by source
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot each generation type with shading
ax.plot(T, P_solar, label="Solar", color='deepskyblue')
ax.fill_between(T, P_solar, color='deepskyblue', alpha=0.3)

ax.plot(T, P_wind, label="Wind", color='green')
ax.fill_between(T, P_wind, color='green', alpha=0.3)

ax.plot(T, P_conv, label="Conventional", color='darkorange')
ax.fill_between(T, P_conv, color='darkorange', alpha=0.3)

ax.plot(T, P_batt_dis, label="Battery Discharge", color='purple')
ax.fill_between(T, P_batt_dis, color='purple', alpha=0.3)

ax.plot(T, P_EV_dis, label="EV Discharge", color='red')
ax.fill_between(T, P_EV_dis, color='red', alpha=0.3)

# Load demand line
ax.plot(T, load_demand, 'k--', label="Load")
ax.fill_between(T, load_demand, color='black', alpha=0.1)

# Axis styling
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_title("Power Generation (MWh)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# -------------------------------
# Plot: State of charge (Battery & EV)
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, SoC, 'magenta', label="Battery SoC")
ax.plot(T, SoC_EV, 'cyan', label="EV SoC")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_title("State of Charge")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Energy (MWh)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# -------------------------------
# Plot: EV Charging vs Discharging
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, P_EV_ch, label="EV Charging", color='blue')
ax.plot(T, P_EV_dis, label="EV Discharging (V2G)", color='red', linestyle='--')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_title("EV Charging vs Discharging (V2G Profile)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# Baseline (non-optimized) scenario for comparison

baseline_tilt = 44  # Fixed panel angle (no optimization)
R_S_baseline = irr_data_24h[baseline_tilt]

# Compute baseline solar generation
P_solar_base = (eta * A_s / 1_000_000) * R_S_baseline

# Wind from measured availability (same as optimized case)
P_wind_base = np.array([wind_power_available(v) for v in wind_data_24h])

# Residual demand met by conventional generation
P_conv_base = load_demand - P_solar_base - P_wind_base
P_conv_base = np.maximum(P_conv_base, 0)  # No negative generation

# Cost & emissions for baseline
cost_conv_base = hourly_prices * P_conv_base
emissions_kg_base = P_conv_base * carbon_intensity_kg_per_mwh
carbon_cost_base = emissions_kg_base * carbon_price

# Summarize total baseline impact
total_cost_base = np.sum(cost_conv_base)
total_emissions_base = np.sum(emissions_kg_base)
total_carbon_cost_base = np.sum(carbon_cost_base)

# -------------------------------
# Plot: Cost comparison
cost_conv = [hourly_prices[t] * P_conv[t] for t in T]
total_cost = sum(cost_conv)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Optimized cost profile
ax.step(T, cost_conv, where='mid', color='orange', linewidth=2, label='With Optimization')

# Baseline cost profile
ax.step(T, cost_conv_base, where='mid', color='red', linewidth=2, linestyle='--', label='Baseline (Conventional Only)')

# Axis formatting
ax.set_xlim(left=0)
ax.set_title(f"Hourly Grid Energy Cost\nTotal: €{total_cost:,.2f} (Optimized) vs €{total_cost_base:,.2f} (Baseline)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Cost (€)")
ax.set_xticks(T)
ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

plt.show()

# -------------------------------
# Emissions calculation
emissions_kg = [P_conv[t] * carbon_intensity_kg_per_mwh[t] for t in T]
total_emissions_kg = sum(emissions_kg)

# Hourly carbon cost
carbon_cost = [e * carbon_price for e in emissions_kg]
total_carbon_cost = sum(carbon_cost)

# Output key metrics
print(f"Total Emissions: {total_emissions_kg:.2f} kg CO₂")
print(f"Total Carbon Cost: €{total_carbon_cost:.2f}")

# -------------------------------
# Plot: Emissions comparison
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Optimized emissions
ax.step(T, emissions_kg, where='mid', color='orange', linewidth=2, label='With Optimization')

# Baseline emissions
ax.step(T, emissions_kg_base, where='mid', color='red', linewidth=2, linestyle='--', label='Baseline (Conventional Only)')

# Axis labels and formatting
ax.set_xlim(left=0)
ax.set_title(f"Hourly CO₂ Emissions from Grid Power \nTotal: {total_emissions_kg:,.2f} kg (Optimized) vs €{total_emissions_base:,.2f} kg (Baseline)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("CO₂ Emissions (kg)")
ax.set_xticks(T)
ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

plt.show()

# -------------------------------
# Plot: Pareto Front (Cost vs Emissions)
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(pareto_emissions, pareto_cost, marker='o', linestyle='-', color='blue')

# Annotate each point with its beta value
for i, (x, y) in enumerate(zip(pareto_emissions, pareto_cost)):
    ax.text(x + 0.3, y, f'{beta_values[i]:.1f}', fontsize=8)

ax.set_title("Pareto Front: Trade-off between Cost and CO₂ Emissions")
ax.set_xlabel("Total CO₂ Emissions (kg)")
ax.set_ylabel("Total Cost (€)")
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


