"""
Borealis高纬度被动式建筑热模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")

# 常数
EARTH_TILT_DEG = 23.44
SOLAR_CONSTANT = 1361.0
AIR_DENSITY = 1.2
AIR_CP = 1005.0

# 气候参数
BOREALIS_LAT = 65.0
BASE_TEMP = 5.0
SEASONAL_AMP = 16.0
DAILY_SWING = 8.0

# 围护
U_WALL = 0.15
U_WINDOW = 0.80
SHGC = 0.55


class SolarEngine:
    def __init__(self, latitude_deg):
        self.lat = np.deg2rad(latitude_deg)
        
    def solar_declination(self, day_of_year):
        return np.deg2rad(EARTH_TILT_DEG) * np.sin(2 * np.pi * (284 + day_of_year) / 365.25)
    
    def solar_altitude_azimuth(self, day, hour):
        decl = self.solar_declination(day)
        hour_angle = np.deg2rad((hour - 12.0) * 15.0)
        
        sin_alt = (np.sin(self.lat) * np.sin(decl) + 
                   np.cos(self.lat) * np.cos(decl) * np.cos(hour_angle))
        sin_alt = np.clip(sin_alt, -1.0, 1.0)
        altitude = np.arcsin(sin_alt)
        
        cos_az_num = (np.sin(decl) - np.sin(self.lat) * sin_alt)
        cos_az_den = (np.cos(self.lat) * np.cos(altitude))
        
        if abs(cos_az_den) < 1e-8:
            azimuth = 0.0
        else:
            cos_az = cos_az_num / cos_az_den
            cos_az = np.clip(cos_az, -1.0, 1.0)
            azimuth = np.arccos(cos_az)
            if hour > 12:
                azimuth = -azimuth
                
        return altitude, azimuth
    
    def extraterrestrial_irradiance(self, day):
        B = 2 * np.pi * day / 365.0
        correction = 1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B)
        return SOLAR_CONSTANT * correction
    
    def direct_normal_irradiance(self, altitude, day):
        if altitude <= 0:
            return 0.0
            
        zenith = np.pi / 2 - altitude
        zenith_deg = np.rad2deg(zenith)
        
        if zenith_deg >= 90:
            return 0.0
            
        air_mass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))
        tau_b = 0.56 * (np.exp(-0.65 * air_mass) + np.exp(-0.095 * air_mass))
        
        I_ext = self.extraterrestrial_irradiance(day)
        DNI = I_ext * tau_b
        
        return max(0.0, DNI)
    
    def diffuse_horizontal_irradiance(self, DNI, altitude):
        if altitude <= 0:
            return 0.0
        
        # 散射辐射约为直射的10-15%（晴天）
        DHI = 0.12 * DNI * np.sin(altitude)
        return max(0.0, DHI)
    
    def calc_irradiance_on_surface(self, day, hour, surface_tilt=90, surface_azimuth=0):
        alt, az = self.solar_altitude_azimuth(day, hour)
        
        if alt <= 0:
            return 0.0, alt, az
        
        DNI = self.direct_normal_irradiance(alt, day)
        DHI = self.diffuse_horizontal_irradiance(DNI, alt)
        
        beta = np.deg2rad(surface_tilt)
        gamma = np.deg2rad(surface_azimuth)
        
        cos_theta = max(0.0, np.sin(alt) * np.cos(beta) + 
                       np.cos(alt) * np.sin(beta) * np.cos(az - gamma))
        
        I_direct = DNI * cos_theta
        I_diffuse = DHI * (1 + np.cos(beta)) / 2
        
        albedo = 0.7 if (day < 90 or day > 300) else 0.2
        I_reflected = (DNI * np.sin(alt) + DHI) * albedo * (1 - np.cos(beta)) / 2
        
        I_total = I_direct + I_diffuse + I_reflected

        return I_total, alt, az

# ==========================================
# 2. 遮阳几何计算（几何意义明确）
# ==========================================

class ShadingGeometry:
    def __init__(self, overhang_depth, window_height):
        self.depth = overhang_depth
        self.height = window_height
        
    def calc_shading_fraction(self, altitude, azimuth):
        if altitude <= 0 or self.depth <= 0:
            return 0.0
        
        ratio = self.depth / self.height
        shadow_ratio = ratio / max(np.tan(altitude), 1e-6)
        azimuth_factor = max(0.0, np.cos(azimuth))
        
        return float(np.clip(shadow_ratio * azimuth_factor, 0.0, 1.0))


class BuildingThermalModel:
    def __init__(self, name, overhang_depth=0.0):
        self.name = name
        self.vent_active = False
        
        # 几何
        self.width = 60.0
        self.depth_dim = 24.0
        self.height = 7.0
        self.floor_area = self.width * self.depth_dim * 2
        self.volume = self.width * self.depth_dim * self.height
        self.window_area = (self.width * self.height) * 0.45
        
        wall_area = 2 * (self.width + self.depth_dim) * self.height
        roof_area = self.width * self.depth_dim
        opaque_area = wall_area + roof_area - self.window_area
        
        # 热阻
        self.UA_window = self.window_area * U_WINDOW
        self.UA_opaque = opaque_area * U_WALL
        self.UA_envelope = self.UA_window + self.UA_opaque
        
        self.ACH_infiltration = 0.5
        self.UA_infiltration = (self.volume * self.ACH_infiltration / 3600) * AIR_DENSITY * AIR_CP
        
        # 热容
        self.C_air = self.volume * AIR_DENSITY * AIR_CP
        
        concrete_thickness_eff = 0.10
        concrete_density = 2400.0
        concrete_cp = 880.0
        thermal_mass_area = self.floor_area * 1.2
        self.C_mass = thermal_mass_area * concrete_thickness_eff * concrete_density * concrete_cp
        
        h_convection = 2.5
        self.UA_coupling = thermal_mass_area * h_convection
        
        self.shading = ShadingGeometry(overhang_depth, window_height=self.height * 0.8)
        self.solar = SolarEngine(BOREALIS_LAT)
        
        dt = 3600
        tau_air = self.C_air / (self.UA_envelope + self.UA_infiltration + self.UA_coupling)
        tau_mass = self.C_mass / self.UA_coupling
        
        print(f"\n=== {self.name} ===")
        print(f"Size: {self.width}m x {self.depth_dim}m x {self.height}m")
        print(f"UA_envelope: {self.UA_envelope:.0f} W/K")
        print(f"C_air: {self.C_air/1e6:.1f} MJ/K, C_mass: {self.C_mass/1e6:.0f} MJ/K")
        print(f"Overhang: {overhang_depth:.2f} m")
        
    def calculate_outdoor_temperature(self, day, hour):
        T_seasonal = BASE_TEMP - SEASONAL_AMP * np.cos(2 * np.pi * (day - 15) / 365.0)
        T_daily = (DAILY_SWING / 2) * np.sin(2 * np.pi * (hour - 9) / 24.0)
        return T_seasonal + T_daily
    
    def calculate_solar_heat_gain(self, day, hour):
        I_total, alt, az = self.solar.calc_irradiance_on_surface(day, hour, 
                                                                   surface_tilt=90, 
                                                                   surface_azimuth=0)
        shading_fraction = self.shading.calc_shading_fraction(alt, az)
        Q_solar_total = I_total * (1 - shading_fraction) * self.window_area * SHGC
        
        Q_solar_to_mass = Q_solar_total * 0.6
        Q_solar_to_air = Q_solar_total * 0.4
        
        return Q_solar_to_mass, Q_solar_to_air, I_total, shading_fraction
    
    def calculate_internal_gains(self, hour):
        if 8 <= hour <= 18:
            Q_internal_total = 5.0 * self.floor_area
        else:
            Q_internal_total = 2.0 * self.floor_area
        
        return Q_internal_total * 0.5, Q_internal_total * 0.5
    
    def calculate_ventilation_load(self, T_air, T_out, day, hour, dt=3600):
        is_summer = (160 < day < 230)
        is_daytime = (8 <= hour <= 20)
        
        if is_summer and is_daytime:
            if not self.vent_active and T_air > 24.0:
                self.vent_active = True
            elif self.vent_active and T_air < 20.0:
                self.vent_active = False
        else:
            self.vent_active = False
            
        if self.vent_active and T_air > T_out:
            dT = T_air - T_out
            smooth_factor = np.clip((T_air - 20.0) / 4.0, 0.1, 1.0)
            ACH_natural = 0.5 + min(3.0, 0.4 * dT) * smooth_factor
            UA_vent = (self.volume * ACH_natural / 3600) * AIR_DENSITY * AIR_CP
        else:
            UA_vent = self.UA_infiltration
            
        phi = UA_vent * dt / self.C_air
        if phi > 0.5:
            UA_vent = UA_vent / (1 + phi)
            
        return UA_vent * (T_out - T_air)
    
    def thermal_step(self, T_air, T_mass, T_out, day, hour, dt=3600):
        Q_sol_mass, Q_sol_air, I_total, shade_frac = self.calculate_solar_heat_gain(day, hour)
        Q_int_mass, Q_int_air = self.calculate_internal_gains(hour)
        
        Q_envelope = self.UA_envelope * (T_out - T_air)
        Q_vent = self.calculate_ventilation_load(T_air, T_out, day, hour)
        Q_coupling = self.UA_coupling * (T_mass - T_air)
        
        alpha = 0.5
        dQ_air_explicit = Q_sol_air + Q_int_air + Q_envelope + Q_vent + alpha * Q_coupling
        implicit_factor = 1.0 / (1.0 + (1.0 - alpha) * self.UA_coupling * dt / self.C_air)
        
        dT_air = (dQ_air_explicit / self.C_air) * dt * implicit_factor
        T_air_new = np.clip(T_air + dT_air, -50, 60)
        
        dQ_mass = Q_sol_mass + Q_int_mass - Q_coupling
        T_mass_new = np.clip(T_mass + (dQ_mass / self.C_mass) * dt, -50, 60)
        
        heat_flows = {
            'Solar': Q_sol_mass + Q_sol_air,
            'Internal': Q_int_mass + Q_int_air,
            'Envelope': Q_envelope,
            'Ventilation': Q_vent,
            'MassRelease': Q_coupling,
            'I_total': I_total,
            'ShadeFraction': shade_frac
        }
        
        return T_air_new, T_mass_new, heat_flows
    
    def run_annual_simulation(self, setpoint_heating=20.0, setpoint_cooling=26.0):
        results = {
            'time_hours': [], 'day': [], 'hour': [],
            'T_out': [], 'T_air_free': [], 'T_mass': [], 'T_air_controlled': [],
            'Q_heating': [], 'Q_cooling': [],
            'Q_solar': [], 'Q_internal': [], 'Q_envelope': [], 'Q_ventilation': [],
            'Q_mass_release': [], 'I_solar': [], 'shade_fraction': []
        }
        
        print(f">>> Warmup (30 days)...")
        T_out_init = BASE_TEMP - SEASONAL_AMP
        T_air_free = T_out_init + 2
        T_mass = T_out_init + 1
        T_air_ctrl = setpoint_heating
        T_mass_ctrl = setpoint_heating - 2
        
        for day in range(-30, 0):
            for hour in range(24):
                T_out = self.calculate_outdoor_temperature(day, hour)
                T_air_free, T_mass, _ = self.thermal_step(T_air_free, T_mass, T_out, day, hour)
                T_air_ctrl, T_mass_ctrl, _ = self.thermal_step(T_air_ctrl, T_mass_ctrl, T_out, day, hour)
                
                if T_air_ctrl < setpoint_heating:
                    T_air_ctrl = setpoint_heating
                elif T_air_ctrl > setpoint_cooling:
                    T_air_ctrl = setpoint_cooling
        
        print(f"Warmup done: T_air={T_air_free:.1f}C, T_mass={T_mass:.1f}C")
        print(f">>> Running 365 days...")
        
        for day in range(365):
            if day % 60 == 0:
                print(f"  Day {day}...")
            
            for hour in range(24):
                time_h = day * 24 + hour
                T_out = self.calculate_outdoor_temperature(day, hour)
                
                T_air_free, T_mass, flows = self.thermal_step(T_air_free, T_mass, T_out, day, hour)
                T_pred, T_mass_ctrl, _ = self.thermal_step(T_air_ctrl, T_mass_ctrl, T_out, day, hour)
                
                Q_heat, Q_cool = 0.0, 0.0
                if T_pred < setpoint_heating:
                    Q_heat = (setpoint_heating - T_pred) * self.C_air / 3600.0
                    T_air_ctrl = setpoint_heating
                elif T_pred > setpoint_cooling:
                    Q_cool = (T_pred - setpoint_cooling) * self.C_air / 3600.0
                    T_air_ctrl = setpoint_cooling
                else:
                    T_air_ctrl = T_pred
                
                results['time_hours'].append(time_h)
                results['day'].append(day)
                results['hour'].append(hour)
                results['T_out'].append(T_out)
                results['T_air_free'].append(T_air_free)
                results['T_mass'].append(T_mass)
                results['T_air_controlled'].append(T_air_ctrl)
                results['Q_heating'].append(Q_heat)
                results['Q_cooling'].append(Q_cool)
                results['Q_solar'].append(flows['Solar'])
                results['Q_internal'].append(flows['Internal'])
                results['Q_envelope'].append(flows['Envelope'])
                results['Q_ventilation'].append(flows['Ventilation'])
                results['Q_mass_release'].append(flows['MassRelease'])
                results['I_solar'].append(flows['I_total'])
                results['shade_fraction'].append(flows['ShadeFraction'])
        
        print(">>> Done!")
        return pd.DataFrame(results)


# ========== 可视化 ==========

def plot_winter_temperature_contrast(df, start_day=15, days=3):
    start_h = start_day * 24
    end_h = start_h + days * 24
    df_slice = df.iloc[start_h:end_h]
    hours = np.arange(len(df_slice))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(hours, df_slice['T_out'], 'k--', linewidth=2, alpha=0.7, label='Outdoor')
    ax.plot(hours, df_slice['T_air_free'], 'r-', linewidth=2.5, label='Indoor Free-Running')
    ax.plot(hours, df_slice['T_mass'], 'orange', linewidth=1.5, alpha=0.6, label='Thermal Mass')
    
    ax.axhline(20, color='green', linestyle=':', linewidth=1.5, label='Heating Setpoint')
    ax.fill_between(hours, 18, 22, color='green', alpha=0.1, label='Comfort Zone')
    
    ax.fill_between(hours, df_slice['T_out'], df_slice['T_air_free'], 
                     where=(df_slice['T_air_free'] > df_slice['T_out']),
                     color='red', alpha=0.15, label='Passive Gain')
    
    ax.fill_between(hours, df_slice['T_air_free'], 20, 
                     where=(df_slice['T_air_free'] < 20),
                     color='blue', alpha=0.15, label='Heating Need')
    
    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.set_title(f'Winter Temperature (Jan {start_day}-{start_day+days})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Fig1_Win_Ctrst.png', dpi=300, bbox_inches='tight')
    print("Saved: Fig1_Win_Ctrst.png")

def plot_winter_heat_balance(df, start_day=15, days=2):
    start_h = start_day * 24
    end_h = start_h + days * 24
    df_slice = df.iloc[start_h:end_h]
    hours = np.arange(len(df_slice))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    solar = df_slice['Q_solar'].values / 1000
    internal = df_slice['Q_internal'].values / 1000
    mass_rel = df_slice['Q_mass_release'].values / 1000
    envelope = df_slice['Q_envelope'].values / 1000
    vent = df_slice['Q_ventilation'].values / 1000
    
    gains = np.zeros(len(hours))
    ax1.fill_between(hours, gains, gains + solar, color='#ff9800', alpha=0.7, label='Solar')
    gains += solar
    ax1.fill_between(hours, gains, gains + internal, color='#f44336', alpha=0.7, label='Internal')
    gains += internal
    ax1.fill_between(hours, gains, gains + np.maximum(mass_rel, 0), color='#9c27b0', alpha=0.7, label='Mass Release')
    
    losses = np.zeros(len(hours))
    ax1.fill_between(hours, losses, losses + envelope, color='#2196f3', alpha=0.7, label='Envelope')
    losses += envelope
    ax1.fill_between(hours, losses, losses + vent, color='#00bcd4', alpha=0.7, label='Ventilation')
    
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_ylabel('Heat Flow (kW)', fontsize=11)
    ax1.set_title('Winter Heat Balance', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(hours, df_slice['T_air_free'], 'r-', linewidth=2, label='Indoor')
    ax2.plot(hours, df_slice['T_mass'], 'orange', linewidth=2, label='Mass')
    ax2.plot(hours, df_slice['T_out'], 'k--', linewidth=1.5, alpha=0.6, label='Outdoor')
    ax2.axhline(20, color='green', linestyle=':', linewidth=1.5)
    ax2.fill_between(hours, 18, 22, color='green', alpha=0.1)
    
    ax2.set_xlabel('Hour', fontsize=11)
    ax2.set_ylabel('Temperature (C)', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Fig2_Win_Balance.png', dpi=300, bbox_inches='tight')
    print("Saved: Fig2_Win_Balance.png")

def plot_summer_shading_effectiveness(df_shaded, df_no_shade, start_day=170, days=3):
    start_h = start_day * 24
    end_h = start_h + days * 24
    df_s = df_shaded.iloc[start_h:end_h]
    df_n = df_no_shade.iloc[start_h:end_h]
    hours = np.arange(len(df_s))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(hours, df_n['T_air_free'], 'r-', linewidth=2.5, label='No Shading')
    ax1.plot(hours, df_s['T_air_free'], 'b-', linewidth=2.5, label='With Shading')
    ax1.plot(hours, df_s['T_out'], 'k--', linewidth=1.5, alpha=0.6, label='Outdoor')
    ax1.axhline(26, color='green', linestyle=':', linewidth=1.5, label='Cooling Limit')
    ax1.fill_between(hours, 22, 26, color='green', alpha=0.1)
    ax1.fill_between(hours, 26, df_n['T_air_free'], where=(df_n['T_air_free'] > 26),
                      color='red', alpha=0.2, label='Overheating')
    
    ax1.set_ylabel('Temperature (C)', fontsize=11)
    ax1.set_title('Summer Temperature', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2.plot(hours, df_s['I_solar'], 'orange', linewidth=2, alpha=0.7, label='Solar')
    ax2.fill_between(hours, 0, df_s['I_solar'], color='orange', alpha=0.2)
    ax2_twin.plot(hours, df_s['shade_fraction'] * 100, 'b-', linewidth=2, label='Shading %')
    
    ax2.set_xlabel('Hour', fontsize=11)
    ax2.set_ylabel('Solar Radiation (W/m2)', fontsize=11, color='orange')
    ax2_twin.set_ylabel('Shading (%)', fontsize=11, color='blue')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Fig3_Sum_Shade.png', dpi=300, bbox_inches='tight')
    print("Saved: Fig3_Sum_Shade.png")

def plot_annual_energy_summary(df_optimized, df_baseline):
    heat_opt = df_optimized['Q_heating'].sum() / 1000
    cool_opt = df_optimized['Q_cooling'].sum() / 1000
    total_opt = heat_opt + cool_opt
    
    heat_base = df_baseline['Q_heating'].sum() / 1000
    cool_base = df_baseline['Q_cooling'].sum() / 1000
    total_base = heat_base + cool_base
    
    savings = (total_base - total_opt) / total_base * 100 if total_base > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    width = 0.6
    x = [0, 1.2]
    
    ax1.bar(x, [heat_base, heat_opt], width, color='#f44336', alpha=0.8, label='Heating')
    ax1.bar(x, [cool_base, cool_opt], width, bottom=[heat_base, heat_opt], 
            color='#2196f3', alpha=0.8, label='Cooling')
    
    ax1.text(x[0], total_base + 50, f'{total_base:.0f}', ha='center', fontsize=11, fontweight='bold')
    ax1.text(x[1], total_opt + 50, f'{total_opt:.0f}', ha='center', fontsize=11, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(['No Shading', 'Optimized'], fontsize=12)
    ax1.set_ylabel('Annual Energy (kWh)', fontsize=12)
    ax1.set_title('Energy Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    categories = ['Heating', 'Cooling', 'Total']
    baseline_vals = [heat_base, cool_base, total_base]
    optimized_vals = [heat_opt, cool_opt, total_opt]
    
    x2 = np.arange(len(categories))
    width2 = 0.35
    
    ax2.bar(x2 - width2/2, baseline_vals, width2, color='gray', alpha=0.6, label='Baseline')
    ax2.bar(x2 + width2/2, optimized_vals, width2, color='green', alpha=0.8, label='Optimized')
    
    for i in range(len(categories)):
        if baseline_vals[i] > 0:
            pct = (baseline_vals[i] - optimized_vals[i]) / baseline_vals[i] * 100
            ax2.text(i, max(baseline_vals[i], optimized_vals[i]) + 50, 
                     f'-{pct:.1f}%', ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.set_ylabel('Energy (kWh)', fontsize=12)
    ax2.set_title(f'Savings: {savings:.1f}%', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Fig4_Ann_Energy_Sum.png', dpi=300, bbox_inches='tight')
    print("Saved: Fig4_Ann_Energy_Sum.png")
    
    print(f"\n=== Results ===")
    print(f"Baseline: Heat {heat_base:.0f}, Cool {cool_base:.0f}, Total {total_base:.0f} kWh")
    print(f"Optimized: Heat {heat_opt:.0f}, Cool {cool_opt:.0f}, Total {total_opt:.0f} kWh")
    print(f"Savings: {savings:.1f}%")


# ========== Main ==========

if __name__ == "__main__":
    print("=" * 60)
    print("Borealis Passive Building Analysis")
    print("=" * 60)
    
    print("\n[1] Optimized (with shading)")
    building_optimized = BuildingThermalModel("Optimized", overhang_depth=1.0)
    df_optimized = building_optimized.run_annual_simulation()
    
    print("\n[2] Baseline (no shading)")
    building_baseline = BuildingThermalModel("Baseline", overhang_depth=0.0)
    df_baseline = building_baseline.run_annual_simulation()
    
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)
    
    plot_winter_temperature_contrast(df_optimized, start_day=15, days=3)
    plot_winter_heat_balance(df_optimized, start_day=15, days=2)
    plot_summer_shading_effectiveness(df_optimized, df_baseline, start_day=170, days=3)
    plot_annual_energy_summary(df_optimized, df_baseline)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)