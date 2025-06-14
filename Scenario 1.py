
# === Import necessary libraries ===
import pyomo.environ as pyo           # Pyomo for optimization modeling
import pandas as pd                   # For handling time series data
import numpy as np                    # For numerical operations
from pvlib.iotools import get_pvgis_hourly  # For retrieving solar/wind irradiance data
import matplotlib.pyplot as plt       # For plotting results

# === Set location and year for data retrieval ===
lat = 52.45
lon = 4.6
year = 2023

# === Retrieve solar irradiance data at multiple tilt angles ===
tilt_angles = [0, 15, 30, 44, 60]
irradiance_data = {}

# Download hourly irradiance data for each tilt angle
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

# Extract timezone from data
sample_tz = irradiance_data[tilt_angles[0]].index.tz

# Focus on 24h profile for June 21st
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)
irr_data_24h = {tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values for tilt in tilt_angles}
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

# === Retrieve wind speed data ===
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

# Wind turbine model parameters (Vestas V90/3000)
total_wind_capacity_mw = 225
v_cut_in = 3.0
v_rated = 12.0
v_cut_out = 25.0

# Power curve to estimate available wind power
def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

# Dictionary of available wind power per hour
P_wind_avail = {t: wind_power_available(wind_data_24h[t]) for t in range(24)}

# === Demand and parameter setup ===
T = range(24)
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# Solar panel and pricing parameters
eta = 0.18                # PV efficiency
A_s = 1860000             # PV surface area (m²)
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
carbon_price = 0.05       # €/kg CO₂

# === Pyomo model definition ===
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)

# Define decision variables
model.P_solar = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.use_angle = pyo.Var(model.T, model.A, within=pyo.Binary)  # angle selector
model.P_wind = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_conv = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.beta = pyo.Param(initialize=0.5, mutable=True)

# === Objective: Weighted combination of cost and emissions ===
model.obj = pyo.Objective(
    expr=model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
         (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T),
    sense=pyo.minimize
)

# === Constraints ===

# Max solar power by selected angle and irradiance
def solar_output_rule(model, t):
    return model.P_solar[t] <= (eta * A_s / 1_000_000) * sum(R_S[(t, a)] * model.use_angle[t, a] for a in model.A)
model.solar_output = pyo.Constraint(model.T, rule=solar_output_rule)

# Only one tilt angle per hour
def one_angle_rule(model, t):
    return sum(model.use_angle[t, a] for a in model.A) == 1
model.one_angle = pyo.Constraint(model.T, rule=one_angle_rule)

# Wind constraint using power availability
model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_wind[t] <= P_wind_avail[t])

# Power balance at each hour
def power_balance_rule(model, t):
    return model.P_solar[t] + model.P_wind[t] + model.P_conv[t] == load_demand[t]
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# === Solve the optimization ===
solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=True)

# === Extract results ===
P_solar = [pyo.value(model.P_solar[t]) for t in T]
P_wind = [pyo.value(model.P_wind[t]) for t in T]
P_conv = [pyo.value(model.P_conv[t]) for t in T]

# Compute total cost and emissions
cost_conv = [hourly_prices[t] * P_conv[t] for t in T]
total_cost = sum(cost_conv)

emissions_kg = [P_conv[t] * carbon_intensity_kg_per_mwh[t] for t in T]
total_emissions_kg = sum(emissions_kg)
carbon_cost = [e * carbon_price for e in emissions_kg]
total_carbon_cost = sum(carbon_cost)

print(f"Total Emissions: {total_emissions_kg:.2f} kg CO₂")
print(f"Total Carbon Cost: €{total_carbon_cost:.2f}")

# === Baseline dispatch with fixed 44° angle ===
baseline_tilt = 44
R_S_baseline = irr_data_24h[baseline_tilt]
P_solar_base = (eta * A_s / 1_000_000) * R_S_baseline
P_wind_base = np.array([wind_power_available(v) for v in wind_data_24h])
P_conv_base = np.maximum(load_demand - P_solar_base - P_wind_base, 0)

# Cost/emissions baseline
cost_conv_base = hourly_prices * P_conv_base
emissions_kg_base = P_conv_base * carbon_intensity_kg_per_mwh
carbon_cost_base = emissions_kg_base * carbon_price

total_cost_base = np.sum(cost_conv_base)
total_emissions_base = np.sum(emissions_kg_base)
total_carbon_cost_base = np.sum(carbon_cost_base)

# === Plot comparisons ===

# Grid power comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(T, P_conv_base, label="Baseline - Conventional", linestyle='--', color='gray')
ax.plot(T, P_conv, label="Optimized - Conventional", color='darkorange')
ax.set_title("Grid Power Comparison: Baseline vs Optimized")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Power source breakdown
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(T, P_solar, label="Solar", color='deepskyblue')
ax.plot(T, P_wind, label="Wind", color='green')
ax.plot(T, P_conv, label="Conventional", color='darkorange')
ax.plot(T, load_demand, 'k--', label="Load")
ax.set_title("Power Generation (MWh)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Cost comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.step(T, cost_conv, where='mid', color='orange', label='Optimized')
ax.step(T, cost_conv_base, where='mid', color='red', linestyle='--', label='Baseline')
ax.set_title(f"Energy Cost: €{total_cost:,.2f} (Opt.) vs €{total_cost_base:,.2f} (Base)")
ax.set_xlabel("Hour")
ax.set_ylabel("€")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Emissions comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.step(T, emissions_kg, where='mid', color='orange', label='Optimized')
ax.step(T, emissions_kg_base, where='mid', color='red', linestyle='--', label='Baseline')
ax.set_title(f"CO₂ Emissions: {total_emissions_kg:.2f} kg vs {total_emissions_base:.2f} kg")
ax.set_xlabel("Hour")
ax.set_ylabel("CO₂ (kg)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Solar energy comparison: optimized vs fixed tilt
total_solar_optimized = sum(P_solar)
total_solar_baseline = sum(P_solar_base)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(T, P_solar, label='Optimized Tilt', color='deepskyblue')
ax.plot(T, P_solar_base, label='Fixed Tilt (44°)', color='gray', linestyle='--')
ax.set_title(f"Solar Output: {total_solar_optimized:.2f} MWh vs {total_solar_baseline:.2f} MWh")
ax.set_xlabel("Hour")
ax.set_ylabel("Solar Power (MW)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

