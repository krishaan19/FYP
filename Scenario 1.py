import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# -------------------------------
# Location and year
lat = 52.45
lon = 4.6
year = 2023

# -------------------------------
# Solar irradiance data
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
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

# -------------------------------
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

total_wind_capacity_mw = 225
v_cut_in = 3.0
v_rated = 12.0
v_cut_out = 25.0

def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

P_wind_avail = {t: wind_power_available(wind_data_24h[t]) for t in range(24)}

# -------------------------------
# Load and parameters
T = range(24)
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

eta = 0.18
A_s = 1860000  # m²
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

# emission_factor = 0.370
carbon_price = 0.05

# -------------------------------
# Pyomo model
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)

model.P_solar = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.use_angle = pyo.Var(model.T, model.A, within=pyo.Binary)
model.P_wind = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_conv = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.beta = pyo.Param(initialize=0.5, mutable=True)

# Objective
model.obj = pyo.Objective(
    expr=model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
         (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T),
    sense=pyo.minimize
)

# Constraints
def solar_output_rule(model, t):
    return model.P_solar[t] <= (eta * A_s / 1_000_000) * sum(R_S[(t, a)] * model.use_angle[t, a] for a in model.A)
model.solar_output = pyo.Constraint(model.T, rule=solar_output_rule)

def one_angle_rule(model, t):
    return sum(model.use_angle[t, a] for a in model.A) == 1
model.one_angle = pyo.Constraint(model.T, rule=one_angle_rule)

model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_wind[t] <= P_wind_avail[t])

def power_balance_rule(model, t):
    return model.P_solar[t] + model.P_wind[t] + model.P_conv[t] == load_demand[t]
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# -------------------------------
# Solve
solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=True)

P_solar = [pyo.value(model.P_solar[t]) for t in T]
P_wind = [pyo.value(model.P_wind[t]) for t in T]
P_conv = [pyo.value(model.P_conv[t]) for t in T]

cost_conv = [hourly_prices[t] * P_conv[t] for t in T]
total_cost = sum(cost_conv)


P_conv = [pyo.value(model.P_conv[t]) for t in T]

# Hourly emissions (kg CO₂)
emissions_kg = [P_conv[t] * carbon_intensity_kg_per_mwh[t] for t in T]
total_emissions_kg = sum(emissions_kg)

# Hourly carbon cost (€)
carbon_price = 0.05  # €/kg CO₂
carbon_cost = [e * carbon_price for e in emissions_kg]
total_carbon_cost = sum(carbon_cost)

print(f"Total Emissions: {total_emissions_kg:.2f} kg CO₂")
print(f"Total Carbon Cost: €{total_carbon_cost:.2f}")

# -------------------------------
# Baseline (Unoptimized) Dispatch
baseline_tilt = 44  # Fixed tilt angle (not optimized)
R_S_baseline = irr_data_24h[baseline_tilt]

# Solar output using fixed tilt angle
P_solar_base = (eta * A_s / 1_000_000) * R_S_baseline

# Wind output directly from availability
P_wind_base = np.array([wind_power_available(v) for v in wind_data_24h])

# Residual load met by conventional power
P_conv_base = load_demand - P_solar_base - P_wind_base
P_conv_base = np.maximum(P_conv_base, 0)  # No negative power

# Cost and emissions
cost_conv_base = hourly_prices * P_conv_base
emissions_kg_base = P_conv_base * carbon_intensity_kg_per_mwh
carbon_cost_base = emissions_kg_base * carbon_price

# Totals
total_cost_base = np.sum(cost_conv_base)
total_emissions_base = np.sum(emissions_kg_base)
total_carbon_cost_base = np.sum(carbon_cost_base)

# print(f"[BASELINE] Total Emissions: {total_emissions_base:.2f} kg CO₂")
# print(f"[BASELINE] Total Carbon Cost: €{total_carbon_cost_base:.2f}")
# print(f"[BASELINE] Total Energy Cost: €{total_cost_base:.2f}")

# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(T, P_conv_base, label="Baseline - Conventional", linestyle='--', color='gray')
# ax.plot(T, P_conv, label="Optimized - Conventional", color='darkorange')
# ax.legend()
# ax.set_title("Grid Power Comparison: Baseline vs Optimized")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Power (MW)")
# ax.grid(True)
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_facecolor('white')     # Outside figure background
# ax.set_facecolor('white')            # Plot area background

# # Plot lines
# ax.plot(T, P_solar, label="Solar", color='deepskyblue')
# ax.fill_between(T, P_solar, color='deepskyblue', alpha=0.3)

# ax.plot(T, P_wind, label="Wind", color='green')
# ax.fill_between(T, P_wind, color='green', alpha=0.3)

# ax.plot(T, P_conv, label="Conventional", color='darkorange')
# ax.fill_between(T, P_conv, color='darkorange', alpha=0.3)

# ax.plot(T, load_demand, 'k--', label="Load")
# ax.fill_between(T, load_demand, color='black', alpha=0.1)
# ax.set_ylim(bottom=0)
# ax.margins(y=0)
# ax.set_xlim(left=0)
# ax.margins(x=0)
# # Styling
# ax.set_title("Power Generation (MWh)")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Power (MW)")
# ax.legend()
# ax.grid(True, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.savefig("/Users/krishaan/Documents/FYP/Results/base_power_generation.png")
# plt.show()


# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')

# # Existing cost plot (e.g., with V2G or optimization)
# ax.step(T, cost_conv, where='mid', color='orange', linewidth=2, label='With Optimization')

# # Baseline conventional cost (e.g., no solar/wind/V2G)
# ax.step(T, cost_conv_base, where='mid', color='red', linewidth=2, linestyle='--', label='Baseline (Conventional Only)')

# # Axis and style settings
# ax.set_xlim(left=0)
# ax.margins(x=0)
# ax.set_title(f"Hourly Grid Energy Cost\nTotal: €{total_cost:,.2f} (Optimized) vs €{total_cost_base:,.2f} (Baseline)")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Cost (€)")
# ax.set_xticks(T)
# ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.5)
# ax.legend()
# plt.tight_layout()
# plt.savefig("/Users/krishaan/Documents/FYP/Results/Baseline_costs.png")
# plt.show()


# # Plot
# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_facecolor('white')
# ax.set_facecolor('white')


# # Existing cost plot (e.g., with V2G or optimization)
# ax.step(T, emissions_kg, where='mid', color='orange', linewidth=2, label='With Optimization')

# # Baseline conventional cost (e.g., no solar/wind/V2G)
# ax.step(T, emissions_kg_base, where='mid', color='red', linewidth=2, linestyle='--', label='Baseline (Conventional Only)')

# # Axis and style settings
# ax.set_xlim(left=0)
# ax.margins(x=0)
# ax.set_title(f"Hourly CO₂ Emissions from Grid Power \nTotal: {total_emissions_kg:,.2f} kg (Optimized) vs €{total_emissions_base:,.2f} kg (Baseline)")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("CO₂ Emissions (kg)")
# ax.set_xticks(T)
# ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.5)
# ax.legend()
# plt.tight_layout()
# plt.savefig("/Users/krishaan/Documents/FYP/Results/Baseline_emissions.png")
# plt.show()

# -------------------------------
# Plot: Solar Energy Available - Optimized vs Baseline
# -------------------------------
# Calculate total solar energy produced (sum over 24 hours)
total_solar_optimized = sum(P_solar)
total_solar_baseline = sum(P_solar_base)

print(f"Total Solar Energy (Optimized): {total_solar_optimized:.2f} MWh")
print(f"Total Solar Energy (Fixed 44° Tilt): {total_solar_baseline:.2f} MWh")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, P_solar, label='Optimized Tilt', color='deepskyblue', linewidth=2)
ax.plot(T, P_solar_base, label='Fixed Tilt (44°)', color='gray', linestyle='--', linewidth=2)

ax.fill_between(T, P_solar, color='deepskyblue', alpha=0.2)
ax.fill_between(T, P_solar_base, color='gray', alpha=0.1)

ax.set_title(f"Hourly Solar Energy Available\n"
             f"Total: {total_solar_optimized:,.2f} MWh (Optimized) vs {total_solar_baseline:,.2f} MWh (Fixed 44°)")

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Solar Power (MW)")
ax.set_xticks(T)
ax.set_ylim(bottom=0)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/solar_comparison.png")
plt.show()
