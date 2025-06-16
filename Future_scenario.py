# === Import Required Libraries ===
import pyomo.environ as pyo                         # Pyomo for optimization modeling
import pandas as pd                                # Pandas for data manipulation
import numpy as np                                 # Numpy for numerical operations
from pvlib.iotools import get_pvgis_hourly         # PVGIS for solar/wind data
import matplotlib.pyplot as plt                    # Plotting
from pyomo.environ import value                    # To extract values from Pyomo variables

# -------------------------------
# === Data Fetching ===

# Wind location coordinates (could differ from solar)
lat_wind = 52.42
lon_wind = 4.13

# Solar location coordinates
lat = 52.45
lon = 4.6

year = 2023  # Analysis year

# -------------------------------
# === Fetch Solar Irradiance Data ===
tilt_angles = [0, 15, 30, 44, 60]  # PV panel tilt angles to evaluate
irradiance_data = {}  # Dictionary to store irradiance data for each tilt

# Get irradiance data for each tilt angle using PVGIS
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
    irradiance_data[tilt] = df['poa_global']  # Store plane-of-array global irradiance

# Extract irradiance values for June 21 (summer solstice)
sample_tz = irradiance_data[tilt_angles[0]].index.tz
june_21 = pd.Timestamp(f"{year}-06-21", tz=sample_tz)

# Slice 24-hour irradiance values for all angles
irr_data_24h = {
    tilt: irradiance_data[tilt].loc[june_21:june_21 + pd.Timedelta("1D")].values
    for tilt in tilt_angles
}

# Total hours and structured irradiance dictionary for use in Pyomo
n_hours = len(next(iter(irr_data_24h.values())))
R_S = {(t, a): irr_data_24h[a][t] for t in range(24) for a in tilt_angles}

# -------------------------------
# === Fetch Wind Data ===
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

# Extract wind speeds for same 24-hour window
wind_data_24h = wind_data.loc[june_21:june_21 + pd.Timedelta("1D")].values

# -------------------------------
# Wind turbine specs (Vestas V90/3000 as example)
total_wind_capacity_mw = 2250  # Total capacity of wind farm

v_cut_in = 3.0    # m/s
v_rated = 12.0    # m/s
v_cut_out = 25.0  # m/s




def wind_power_output(V, V_I=3, V_R=12, V_O=25, Cp=0.4, rho=1.225, R=100, P_rated=11):
    A = np.pi * R**2  # Swept area
    
    if V < V_I or V > V_O:
        return 0
    elif V_I <= V < V_R:
        return 0.5 * Cp * rho * A * V**3 / 1000000  # Convert to kW
    else:  # V_R <= V <= V_O
        return P_rated



P_wind_avail = {t: wind_power_output(wind_data_24h[t]) * 100 for t in range(n_hours)}

# -------------------------------
# === Time Horizon ===
T = range(24)  # 24-hour optimization horizon (0 to 23)

# -------------------------------
# === Load Profile (MW) ===
load_demand = np.array([
    548.37, 538.31, 532.04, 527.97, 524.20, 526.02,
    533.78, 553.15, 578.53, 600.72, 608.33, 611.27,
    613.82, 611.80, 610.05, 611.62, 617.26, 629.92,
    639.53, 645.12, 639.20, 619.91, 595.82, 569.33
])

# === Grid Carbon Intensity per Hour (kg CO₂/MWh) ===
carbon_intensity_kg_per_mwh = np.array([
    295.91, 292.62, 303.61, 301.54, 289.90, 262.17,
    219.20, 181.58, 151.12, 137.93, 125.44, 114.22,
    102.33,  98.97, 114.30, 154.95, 185.88, 228.38,
    288.13, 343.84, 355.35, 371.06, 375.78, 366.44
])

# -------------------------------
# === Load Classification ===

# Define 40% of total load as shiftable
shiftable_load = 0.4 * load_demand  
non_shiftable_load = load_demand - shiftable_load

# -------------------------------
# === System Parameters ===
eta = 0.18  # PV efficiency
A_s = 15700000   # PV area (m²)
cost_conv = 80
emission_conv = 0.5
battery_capacity = 200
battery_charge_rate = 50
battery_eff = 0.95
ev_capacity = 201.25
ev_charge_rate = 50
ev_eff = 0.90
ev_initial_soc = 100
ev_min_soc = 20         # Minimum required SoC when EVs are away

# EV availability: 1 when home (night), 0 when away (day)
ev_available = [1 if (t >= 18 or t <= 8) else 0 for t in T]

alpha = 10  # penalty scaling factor (not directly used)

# Hourly electricity prices in €/MWh
hourly_prices = np.array([
    119.70, 110.77, 104.80, 103.92, 103.76, 107.02,
    126.56, 137.67, 144.55, 126.16, 103.02, 91.16,
    89.12, 86.50, 86.64, 94.60, 100.16, 113.62,
    138.17, 167.50, 190.24, 180.66, 155.57, 134.86
])

# -------------------------------
# === EV Driving Load ===



# -------------------------------
# === Carbon Pricing ===
carbon_price = 0.05  # €/kg CO₂

# -------------------------------
# === Pyomo Concrete Model Setup ===
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T)            # Time steps (0–23)
model.A = pyo.Set(initialize=tilt_angles)  # PV tilt angle options

# -------------------------------
# === Decision Variables ===

# Power outputs by source (all non-negative)
model.P_solar     = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.use_angle   = pyo.Var(model.T, model.A, within=pyo.Binary)  # Select one tilt per hour
model.P_wind      = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_conv      = pyo.Var(model.T, within=pyo.NonNegativeReals)

# Battery charge/discharge and SoC
model.P_batt_ch   = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_batt_dis  = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.SoC         = pyo.Var(model.T, bounds=(0, battery_capacity))

# EV charging/discharging and SoC
model.P_EV_ch     = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.P_EV_dis    = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.SoC_EV      = pyo.Var(model.T, bounds=(0, ev_capacity))

# Load shifting and ramping
model.D_shift     = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.Ramp        = pyo.Var(model.T, within=pyo.NonNegativeReals)

# EV export to grid and mode (1 = charging, 0 = discharging)
model.P_EV_to_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.EV_mode      = pyo.Var(model.T, within=pyo.Binary)

# Weighting factor for cost vs. emissions in objective
model.beta = pyo.Param(initialize=0.5, mutable=True)

# -------------------------------
# === Penalty Settings ===
ramp_penalty = 10  # Penalize large load shifting ramp rates

# Hourly penalty weights for shifting (higher at night)
shift_penalty_per_hour = np.array([
    20, 20, 20, 20, 20, 20, 20, 15,  # 00:00–07:00
    5, 2, 0, 0, 0, 0, 1, 3, 5, 8,    # 08:00–17:00
    15, 18, 20, 20, 20, 20           # 18:00–23:00
])

# -------------------------------
# === Objective Function ===

# Multi-objective: Cost + Emissions + Load shifting smoothness + Penalty
model.obj = pyo.Objective(
    expr= model.beta * sum(hourly_prices[t] * model.P_conv[t] for t in model.T) +
        (1 - model.beta) * sum(carbon_intensity_kg_per_mwh[t] * model.P_conv[t] for t in model.T)+
        ramp_penalty * sum((model.D_shift[t] - model.D_shift[t-1])**2 for t in model.T if t > 0) +
        sum(shift_penalty_per_hour[t] * model.D_shift[t] for t in model.T),
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

model.batt_ch_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_batt_ch[t] <= battery_charge_rate)
model.batt_dis_limit = pyo.Constraint(model.T, rule=lambda m, t: model.P_batt_dis[t] <= battery_charge_rate)

def soc_batt_rule(model, t):
    if t == 0:
        return model.SoC[t] == 0 + battery_eff * model.P_batt_ch[t] - (1/battery_eff) * model.P_batt_dis[t]
    else:
        return model.SoC[t] == model.SoC[t-1] + battery_eff * model.P_batt_ch[t] - (1/battery_eff) * model.P_batt_dis[t]


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
# -------------------------------
# === Pareto Frontier Loop ===

# Solver configuration
solver = pyo.SolverFactory('gurobi')



pareto_emissions = []
pareto_cost = []
beta_values = np.linspace(0, 1, 21)

for b in beta_values:
    model.beta = b
    solver.solve(model, tee=False)
    
    cost = sum(hourly_prices[t] * value(model.P_conv[t]) for t in model.T)
    emissions = sum(carbon_intensity_kg_per_mwh[t] * value(model.P_conv[t]) for t in model.T)
    
    pareto_cost.append(cost)
    pareto_emissions.append(emissions)


# -------------------------------
# === Solve Final Model with Selected Beta ===
solver = pyo.SolverFactory('gurobi')
import time
start_time = time.time()
results = solver.solve(model, tee=True)
end_time = time.time()
print(f"\n⏱️ Model solved in {end_time - start_time:.2f} seconds.")

# -------------------------------
# === Extract Variable Values for Plotting ===
P_solar    = [pyo.value(model.P_solar[t]) for t in T]
P_wind     = [pyo.value(model.P_wind[t]) for t in T]
P_conv     = [pyo.value(model.P_conv[t]) for t in T]
SoC        = [pyo.value(model.SoC[t]) for t in T]
SoC_EV     = [pyo.value(model.SoC_EV[t]) for t in T]
D_shift    = [pyo.value(model.D_shift[t]) for t in T]
P_EV_ch    = [pyo.value(model.P_EV_ch[t]) for t in T]
P_EV_dis   = [pyo.value(model.P_EV_dis[t]) for t in T]
P_batt_dis = [pyo.value(model.P_batt_dis[t]) for t in T]

# Calculate total energy from each source
solar_energy  = sum(value(model.P_solar[t]) for t in T)
wind_energy   = sum(value(model.P_wind[t]) for t in T)
batt_dis_energy = sum(value(model.P_batt_dis[t]) for t in T)
ev_dis_energy   = sum(value(model.P_EV_dis[t]) for t in T)
conv_energy     = sum(value(model.P_conv[t]) for t in T)

# -------------------------------
# === Print Key Results ===
print("🔋 Total Power Generated Over 24 Hours (MWh):")
print(f"  ☀️  Solar:           {solar_energy:.2f} MWh")
print(f"  🌬️  Wind:            {wind_energy:.2f} MWh")
print(f"  🔌 Battery Discharge: {batt_dis_energy:.2f} MWh")
print(f"  🚗 EV Discharge:      {ev_dis_energy:.2f} MWh")
print(f"  ⚡ Conventional:      {conv_energy:.2f} MWh (Expected)")

# -------------------------------
# === Plot: Power Generation Breakdown ===
total_load_served = non_shiftable_load + np.array(D_shift)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

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

ax.set_title("Power Generation (MWh) with Demand Shifting")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# -------------------------------
# === Plot: SoC for Battery and EVs ===
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, SoC, 'magenta', label="Battery SoC")
ax.plot(T, SoC_EV, 'cyan', label="EV SoC")
ax.set_title("State of Charge")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Energy (MWh)")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# -------------------------------
# === Plot: Original vs Shifted Load ===
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, load_demand, 'k--', label='Original Total Load')
ax.plot(T, total_load_served, 'b-', label='Final Load (after shifting)')
ax.set_title('Total Load Before and After Load Shifting')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Power (MW)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

# -------------------------------
# === Plot: EV Charging vs Discharging ===
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(T, P_EV_ch, label="EV Charging", color='blue')
ax.plot(T, P_EV_dis, label="EV Discharging (V2G)", color='red', linestyle='--')
ax.set_title("EV Charging vs Discharging (V2G Profile)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Power (MW)")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# === Baseline Scenario (No Optimization) ===

baseline_tilt = 44
R_S_baseline = irr_data_24h[baseline_tilt]

# Fixed angle solar generation
P_solar_base = (eta * A_s / 1_000_000) * R_S_baseline

# Wind generation unchanged
P_wind_base = np.array([wind_power_output(v) for v in wind_data_24h])

# Remainder met by conventional generation
P_conv_base = load_demand - P_solar_base - P_wind_base
P_conv_base = np.maximum(P_conv_base, 0)

# Baseline cost and emissions
cost_conv_base = hourly_prices * P_conv_base
emissions_kg_base = P_conv_base * carbon_intensity_kg_per_mwh
carbon_cost_base = emissions_kg_base * carbon_price

# Totals
total_cost_base = np.sum(cost_conv_base)
total_emissions_base = np.sum(emissions_kg_base)
total_carbon_cost_base = np.sum(carbon_cost_base)

# -------------------------------
# === Plot: Cost Comparison ===

cost_conv = [hourly_prices[t] * P_conv[t] for t in T]
total_cost = sum(cost_conv)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.step(T, cost_conv, where='mid', color='orange', linewidth=2, label='With Optimization')
ax.step(T, cost_conv_base, where='mid', color='red', linestyle='--', linewidth=2, label='Baseline (Conventional Only)')

ax.set_title(f"Hourly Grid Energy Cost\nTotal: €{total_cost:,.2f} (Optimized) vs €{total_cost_base:,.2f} (Baseline)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Cost (€)")
ax.set_xlim(left=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

plt.show()

# -------------------------------
# === Plot: Emissions Comparison ===

emissions_kg = [P_conv[t] * carbon_intensity_kg_per_mwh[t] for t in T]
total_emissions_kg = sum(emissions_kg)

carbon_cost = [e * carbon_price for e in emissions_kg]
total_carbon_cost = sum(carbon_cost)

print(f"Total Emissions: {total_emissions_kg:.2f} kg CO₂")
print(f"Total Carbon Cost: €{total_carbon_cost:.2f}")

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.step(T, emissions_kg, where='mid', color='orange', linewidth=2, label='With Optimization')
ax.step(T, emissions_kg_base, where='mid', color='red', linestyle='--', linewidth=2, label='Baseline (Conventional Only)')

ax.set_title(f"Hourly CO₂ Emissions\nTotal: {total_emissions_kg:,.2f} kg (Optimized) vs {total_emissions_base:,.2f} kg (Baseline)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("CO₂ Emissions (kg)")
ax.set_xlim(left=0)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()

plt.show()

# -------------------------------
# === Plot: Pareto Frontier (Cost vs Emissions) ===
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('lightgrey')
ax.set_facecolor('gainsboro')

ax.plot(pareto_emissions, pareto_cost, marker='o', linestyle='-', color='blue')
for i, (x, y) in enumerate(zip(pareto_emissions, pareto_cost)):
    ax.text(x + 0.3, y, f'{beta_values[i]:.1f}', fontsize=8)

ax.set_title("Pareto Front: Trade-off between Cost and CO₂ Emissions")
ax.set_xlabel("Total CO₂ Emissions (kg)")
ax.set_ylabel("Total Cost (€)")
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()