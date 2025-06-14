import pyomo.environ as pyo
import pandas as pd
import numpy as np
from pvlib.iotools import get_pvgis_hourly
import matplotlib.pyplot as plt

# -------------------------------
# Fetching real solar and wind data
lat = 52.45
lon = 4.6
year = 2023

# Define tilt angles for solar panel simulations
tilt_angles = [0, 15, 30, 44, 60]
irradiance_data = {}

# Get hourly irradiance data for each tilt angle
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
    irradiance_data[tilt] = df['poa_global']  # Plane of array global irradiance

# Extract time zone and focus on June 21st (typical high solar day)
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)

# Build irradiance profile for 24 hours per tilt angle
irr_data_24h = {
    tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values
    for tilt in tilt_angles
}
n_hours = len(next(iter(irr_data_24h.values())))
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}  # Indexed irradiance

# -------------------------------
# Wind data fetching from ERA5
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
wind_data_24h = wind_data.loc[june_21:june_21 + pd.Timedelta("1D")].values  # 24h wind speeds

# Wind turbine characteristics (Vestas V90/3000)
total_wind_capacity_mw = 225  # total installed wind capacity
v_cut_in = 3.0
v_rated = 12.0
v_cut_out = 25.0

# Define wind power availability function based on cubic interpolation
def wind_power_available(v):
    if v < v_cut_in or v > v_cut_out:
        return 0.0
    elif v_cut_in <= v < v_rated:
        return total_wind_capacity_mw * ((v**3 - v_cut_in**3) / (v_rated**3 - v_cut_in**3))
    else:
        return total_wind_capacity_mw

# Calculate available wind power per hour
P_wind_avail = {t: wind_power_available(wind_data_24h[t]) for t in range(n_hours)}

# -------------------------------
# Time and load setup
T = range(24)  # 24-hour horizon

# Hourly load demand (synthetic or actual)
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# -------------------------------
# Technology and economic parameters
eta = 0.18             # PV efficiency
A_s = 1860000          # PV area (m²)
battery_capacity = 100 # MWh
battery_charge_rate = 80  # MW
battery_eff = 0.95
ev_capacity = 201.25   # MWh
ev_charge_rate = 30    # MW
ev_eff = 0.90
ev_initial_soc = 100   # Initial EV SoC (MWh)
ev_min_soc = 40        # Minimum SoC when not connected
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]  # EV available at night
hourly_prices = np.array([...])  # €/MWh energy price per hour
carbon_intensity_kg_per_mwh = np.array([...])  # kg CO₂/MWh per hour
carbon_price = 0.05  # €/kg CO₂

# -------------------------------
# Build Pyomo model
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)
model.A = pyo.Set(initialize=tilt_angles)

# -------------------------------
# Define decision variables
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
model.beta = pyo.Param(initialize=0.5, mutable=True)  # Cost-emissions weighting
model.EV_mode = pyo.Var(model.T, within=pyo.Binary)   # 1 = charging, 0 = discharging

# -------------------------------
# Objective: Weighted sum of energy cost and emissions
model.obj = pyo.Objective(
    expr=model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
         (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T),
    sense=pyo.minimize
)

# -------------------------------
# Constraints
model.ev_charge_only_if_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_ch[t] <= ev_charge_rate * m.EV_mode[t])
model.ev_discharge_only_if_not_mode = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_dis[t] <= ev_charge_rate * (1 - m.EV_mode[t]))
model.ev_export_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_EV_to_grid[t] <= m.P_EV_dis[t])

# Solar power constraint based on chosen tilt angle
model.solar_output = pyo.Constraint(model.T, rule=lambda m, t:
    m.P_solar[t] <= (eta * A_s / 1_000_000) * sum(R_S[(t, a)] * m.use_angle[t, a] for a in m.A)
)

# Ensure only one tilt angle is selected per hour
model.one_angle = pyo.Constraint(model.T, rule=lambda m, t: sum(m.use_angle[t, a] for a in m.A) == 1)

# Wind availability constraint
model.wind_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_wind[t] <= P_wind_avail[t])

# Power balance: generation = consumption + charging
model.power_balance = pyo.Constraint(model.T, rule=lambda m, t:
    m.P_solar[t] + m.P_wind[t] + m.P_conv[t] + m.P_batt_dis[t] + m.P_EV_dis[t] ==
    load_demand[t] + m.P_batt_ch[t] + m.P_EV_ch[t]
)

# Battery operational constraints
model.batt_charge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_ch[t] <= battery_charge_rate)
model.batt_discharge_limit = pyo.Constraint(model.T, rule=lambda m, t: m.P_batt_dis[t] <= battery_charge_rate)

# Battery state of charge update
model.soc_batt = pyo.Constraint(model.T, rule=lambda m, t:
    m.SoC[t] == (
        20 + battery_eff * m.P_batt_ch[t] - (1/battery_eff) * m.P_batt_dis[t]
        if t == 0 else
        m.SoC[t-1] + battery_eff * m.P_batt_ch[t-1] - (1/battery_eff) * m.P_batt_dis[t-1]
    )
)

# EV state of charge update
model.soc_ev = pyo.Constraint(model.T, rule=lambda m, t:
    m.SoC_EV[t] == (
        ev_initial_soc + ev_eff * m.P_EV_ch[t] - (1/ev_eff) * m.P_EV_dis[t]
        if t == 0 else
        m.SoC_EV[t-1] + ev_eff * m.P_EV_ch[t] - (1/ev_eff) * m.P_EV_dis[t]
    )
)

# Enforce minimum SoC when EV not available
model.ev_min_soc = pyo.Constraint(model.T, rule=lambda m, t:
    m.SoC_EV[t] >= ev_min_soc if ev_available[t] == 0 else pyo.Constraint.Skip
)

# -------------------------------
# Pareto front generation: trade-off between cost and emissions
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
# Sensitivity analysis: storage capacity
storage_caps = list(range(80, 401, 40))  # Test from 80 to 400 MWh
storage_costs = []
storage_emissions = []

for cap in storage_caps:
    # Update capacity bounds dynamically
    model.SoC.setlb(0)
    model.SoC.setub(cap)
    model.SoC_EV.setlb(0)
    model.SoC_EV.setub(cap)

    # Solve for this capacity
    results = solver.solve(model, tee=False)

    # Record cost and emissions
    total_cost = sum(hourly_prices[t] * pyo.value(model.P_conv[t]) for t in T) / 1e6
    total_emissions = sum(carbon_intensity_kg_per_mwh[t] * pyo.value(model.P_conv[t]) for t in T) / 1000
    storage_costs.append(total_cost)
    storage_emissions.append(total_emissions)

# -------------------------------
# Plot results: Cost and emissions vs storage capacity
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Battery/EV Storage Capacity (MWh)')
ax1.set_ylabel('Cost (€ mln)', color='blue')
ax1.plot(storage_caps, storage_costs, 'o-', color='blue', label='Cost')
ax1.tick_params(axis='y', labelcolor='blue')

# Second y-axis for emissions
ax2 = ax1.twinx()
ax2.set_ylabel('CO₂ Emissions (kg mln)', color='red')
ax2.plot(storage_caps, storage_emissions, 's--', color='red', label='Emissions')
ax2.tick_params(axis='y', labelcolor='red')

# Final layout tweaks
plt.title('Sensitivity of Cost and Emissions to Storage Capacity')
fig.tight_layout()
plt.grid(True)
plt.show()

