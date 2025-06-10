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

daily_ev_driving_energy = 3.0  # total MWh consumed while driving per day
driving_hours = [t for t in T if ev_available[t] == 0]
driving_load = np.zeros(len(T))
for t in driving_hours:
    driving_load[t] = daily_ev_driving_energy / len(driving_hours)
    
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
    expr= model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
          (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T)+
          ramp_penalty * sum((model.D_shift[t] - model.D_shift[t-1])**2 for t in model.T if t > 0) +
          sum(shift_penalty_per_hour[t] * model.D_shift[t] for t in model.T),
    sense=pyo.minimize
)

# -------------------------------





# # Constraints
# # Only allow charging when EV_mode[t] == 1
# model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])

# # Only allow discharging when EV_mode[t] == 0
# model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))

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
            non_shiftable_load[t] + model.D_shift[t] +
            model.P_batt_ch[t] + model.P_EV_ch[t])
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

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


# Driving energy consumption per driving hour (MWh)


def soc_ev_rule(model, t):
    if t == 0:
        return model.SoC_EV[t] == ev_initial_soc + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t] - driving_load[t]
    else:
        return model.SoC_EV[t] == model.SoC_EV[t-1] + ev_eff * model.P_EV_ch[t] - (1/ev_eff) * model.P_EV_dis[t] - driving_load[t]

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

# -------------------------------
# Plot results
P_solar = [pyo.value(model.P_solar[t]) for t in T]
P_wind = [pyo.value(model.P_wind[t]) for t in T]
P_conv = [pyo.value(model.P_conv[t]) for t in T]
SoC = [pyo.value(model.SoC[t]) for t in T]
SoC_EV = [pyo.value(model.SoC_EV[t]) for t in T]
D_shift = [pyo.value(model.D_shift[t]) for t in T]
P_EV_ch = [pyo.value(model.P_EV_ch[t]) for t in T]
P_EV_dis = [pyo.value(model.P_EV_dis[t]) for t in T]
P_batt_dis = [pyo.value(model.P_batt_dis[t]) for t in T]

solar_energy = sum(value(model.P_solar[t]) for t in T)
wind_energy = sum(value(model.P_wind[t]) for t in T)
batt_dis_energy = sum(value(model.P_batt_dis[t]) for t in T)
ev_dis_energy = sum(value(model.P_EV_dis[t]) for t in T)
conv_energy = sum(value(model.P_conv[t]) for t in T)

# === Print Results ===
print("🔋 Total Power Generated Over 24 Hours (MWh):")
print(f"  ☀️  Solar:           {solar_energy:.2f} MWh")
print(f"  🌬️  Wind:            {wind_energy:.2f} MWh")
print(f"  🔌 Battery Discharge: {batt_dis_energy:.2f} MWh")
print(f"  🚗 EV Discharge:      {ev_dis_energy:.2f} MWh")
print(f"  ⚡ Conventional:      {conv_energy:.2f} MWh (Expected)")

# Plot with grey background and shaded areas
total_load_served = non_shiftable_load + np.array(D_shift)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')     # Outside figure background
ax.set_facecolor('white')            # Plot area background

# Plot lines


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

ax.plot(T, total_load_served, 'k--', label="Load")
ax.fill_between(T, total_load_served, color='black', alpha=0.1)
ax.set_ylim(bottom=0)
ax.margins(y=0)
ax.set_xlim(left=0)
ax.margins(x=0)
# Styling
ax.set_title("Power Generation (MWh) with Demand Shifting")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/V2G_demand_shift_power_generation.png")
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, SoC, 'magenta', label="Battery SoC")
ax.plot(T, SoC_EV, 'cyan', label="EV SoC")
ax.set_ylim(bottom=0)
ax.margins(y=0)
ax.set_xlim(left=0)
ax.margins(x=0)
ax.set_title("State of Charge")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Energy (MWh)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/V2G_demand_shift_SoC.png")
plt.show()


fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, load_demand, 'k--', label='Original Total Load')
ax.plot(T, total_load_served, 'b-', label='Final Load (after shifting)')
ax.set_ylim(bottom=0)
ax.margins(y=0)
ax.set_xlim(left=0)
ax.margins(x=0)
ax.set_title('Total Load Before and After Load Shifting')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Power (MW)')
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/krishaan/Documents/FYP/Results/V2G_demand_shift_demand_line.png")
plt.show()



fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, P_EV_ch, label="EV Charging", color='blue')
ax.plot(T, P_EV_dis, label="EV Discharging (V2G)", color='red', linestyle='--')
ax.set_ylim(bottom=0)
ax.margins(y=0)
ax.set_xlim(left=0)
ax.margins(x=0)
ax.set_title("EV Charging vs Discharging (V2G Profile)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.legend()
ax.grid(True, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# # -------------------------------
# # Baseline (Unoptimized) Dispatch
# baseline_tilt = 44  # Fixed tilt angle (not optimized)
# R_S_baseline = irr_data_24h[baseline_tilt]

# # Solar output using fixed tilt angle
# P_solar_base = (eta * A_s / 1_000_000) * R_S_baseline

# # Wind output directly from availability
# P_wind_base = np.array([wind_power_available(v) for v in wind_data_24h])

# # Residual load met by conventional power
# P_conv_base = load_demand - P_solar_base - P_wind_base
# P_conv_base = np.maximum(P_conv_base, 0)  # No negative power

# # Cost and emissions
# cost_conv_base = hourly_prices * P_conv_base
# emissions_kg_base = P_conv_base * carbon_intensity_kg_per_mwh
# carbon_cost_base = emissions_kg_base * carbon_price

# # Totals
# total_cost_base = np.sum(cost_conv_base)
# total_emissions_base = np.sum(emissions_kg_base)
# total_carbon_cost_base = np.sum(carbon_cost_base)


# cost_conv = [hourly_prices[t] * P_conv[t] for t in T]
# total_cost = sum(cost_conv)

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
# plt.savefig("/Users/krishaan/Documents/FYP/Results/V2G_demand_shift_costs.png")
# plt.show()


# emissions_kg = [P_conv[t] * carbon_intensity_kg_per_mwh[t] for t in T]
# total_emissions_kg = sum(emissions_kg)

# # Hourly carbon cost (€)
# carbon_price = 0.05  # €/kg CO₂
# carbon_cost = [e * carbon_price for e in emissions_kg]
# total_carbon_cost = sum(carbon_cost)

# print(f"Total Emissions: {total_emissions_kg:.2f} kg CO₂")
# print(f"Total Carbon Cost: €{total_carbon_cost:.2f}")

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
# plt.savefig("/Users/krishaan/Documents/FYP/Results/V2G_demand_shift_emissions.png")
# plt.show()



# fig, ax = plt.subplots(figsize=(10, 6))
# fig.patch.set_facecolor('lightgrey')
# ax.set_facecolor('gainsboro')

# ax.plot(pareto_emissions, pareto_cost, marker='o', linestyle='-', color='blue')
# for i, (x, y) in enumerate(zip(pareto_emissions, pareto_cost)):
#     ax.text(x + 0.3, y, f'{beta_values[i]:.1f}', fontsize=8)

# ax.set_title("Pareto Front: Trade-off between Cost and CO₂ Emissions")
# ax.set_xlabel("Total CO₂ Emissions (kg)")
# ax.set_ylabel("Total Cost (€)")
# ax.grid(True, color='gray', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


