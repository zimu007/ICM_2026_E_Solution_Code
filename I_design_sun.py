"""
Sungrove被动式建筑热模型

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
CP_AIR = 1006.0      
RHO_AIR = 1.18       
SOLAR_CONSTANT = 1361 

class BuildingType(Enum):
    BASELINE = "baseline"
    ADVANCED = "advanced"

@dataclass
class BuildingConfig:
    building_type: BuildingType
    latitude: float = 25.0
    
    # 几何
    floor_area: float = 2880.0
    volume: float = 10080.0
    height: float = 7.0
    wall_area: float = 800.0
    window_area: float = 320.0
    window_height: float = 2.0
    roof_area: float = 1440.0
    
    # 材料参数
    glass_shgc: float = 0.0
    glass_u: float = 0.0
    glass_vlt: float = 0.0
    wall_thickness: float = 0.0
    wall_k: float = 0.0
    wall_rho: float = 0.0
    wall_cp: float = 0.0
    wall_alpha: float = 0.0
    roof_u: float = 0.0
    roof_alpha: float = 0.0
    overhang_depth: float = 0.0
    has_shading: bool = False
    ach_min: float = 0.5
    ach_max: float = 1.0
    
    # 内热
    internal_gain_occupied: float = 18.0
    internal_gain_unoccupied: float = 2.0
    occupancy_hours: tuple = (8, 18)
    t_comfort_upper: float = 26.0
    t_comfort_lower: float = 18.0

    def __post_init__(self):
        if self.building_type == BuildingType.BASELINE:
            self._set_baseline()
        else:
            self._set_advanced()
    
    def _set_baseline(self):
        # 单玻 
        self.glass_shgc = 0.86
        self.glass_u = 5.8
        self.glass_vlt = 0.89
        # 普通混凝土
        self.wall_thickness = 0.2
        self.wall_k = 1.7
        self.wall_rho = 2400
        self.wall_cp = 880
        self.wall_alpha = 0.6
        # 深色屋顶
        self.roof_u = 2.0
        self.roof_alpha = 0.7
        # 无遮阳
        self.overhang_depth = 0.0
        self.has_shading = False
        self.ach_min = 0.5
        self.ach_max = 1.5
    
    def _set_advanced(self):
        # 三银Low-E
        self.glass_shgc = 0.28
        self.glass_u = 1.6
        self.glass_vlt = 0.64
        # 夯土墙
        self.wall_thickness = 0.5
        self.wall_k = 0.85
        self.wall_rho = 1900
        self.wall_cp = 1200
        self.wall_alpha = 0.4
        # 绿屋顶
        self.roof_u = 0.35
        self.roof_alpha = 0.25
        self.overhang_depth = 1.0
        self.has_shading = True
        self.ach_min = 0.5
        self.ach_max = 10.0


def generate_outdoor_temperature_matrix() -> np.ndarray:
    """生成全年室外温度矩阵, 正弦日变化"""
    matrix = np.zeros((DAYS_IN_YEAR, HOURS_IN_DAY))
    
    for d in range(DAYS_IN_YEAR):
        # 季节基准温
        base_temp = 24.0
        seasonal_amp = 6.0
        thermal_lag = 30
        season_phase = 2 * np.pi * (d - thermal_lag) / 365.0
        t_seasonal = base_temp + seasonal_amp * (-np.cos(season_phase))
        
        # 日温差
        winter_factor = 0.5 * (1 + np.cos(season_phase))
        daily_swing = 6.0 + 4.0 * winter_factor
        
        for h in range(HOURS_IN_DAY):
            # 5点最低, 15点最高
            hour_phase = 2 * np.pi * (h - 5) / 24.0
            t_daily = -daily_swing/2 * np.cos(hour_phase)
            matrix[d, h] = t_seasonal + t_daily
    
    return np.clip(matrix, 12.0, 36.0)

T_OUTDOOR_MATRIX = None

def get_outdoor_temp_matrix():
    global T_OUTDOOR_MATRIX
    if T_OUTDOOR_MATRIX is None:
        T_OUTDOOR_MATRIX = generate_outdoor_temperature_matrix()
    return T_OUTDOOR_MATRIX


class ThermalEngine:
    """热平衡计算"""
    
    def __init__(self, config: BuildingConfig):
        self.cfg = config
        self.h_out = 25.0
        self.h_in = 8.0
        self.thermal_mass = self._calc_thermal_mass()
        
    def _calc_thermal_mass(self) -> float:
        participation = 0.35 if self.cfg.building_type == BuildingType.ADVANCED else 0.20
        wall_cap = (self.cfg.wall_area * self.cfg.wall_thickness * participation * 
                    self.cfg.wall_rho * self.cfg.wall_cp)
        air_cap = self.cfg.volume * RHO_AIR * CP_AIR
        furniture_cap = self.cfg.floor_area * (60000 if self.cfg.building_type == BuildingType.ADVANCED else 35000)
        return wall_cap + air_cap + furniture_cap
    
    def _calc_solar_position(self, day: int, hour: float) -> Tuple[float, float]:
        lat_rad = np.radians(self.cfg.latitude)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day) / 365))
        decl_rad = np.radians(declination)
        hour_angle = 15 * (hour - 12)
        h_rad = np.radians(hour_angle)
        
        sin_alt = (np.sin(lat_rad) * np.sin(decl_rad) + 
                   np.cos(lat_rad) * np.cos(decl_rad) * np.cos(h_rad))
        altitude = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))
        
        if np.cos(np.radians(altitude)) > 0.001:
            cos_az = (np.sin(decl_rad) - np.sin(lat_rad) * sin_alt) / \
                     (np.cos(lat_rad) * np.cos(np.radians(altitude)))
            azimuth = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
            if hour > 12:
                azimuth = 360 - azimuth
        else:
            azimuth = 180
        return altitude, azimuth
    
    def _calc_solar_radiation(self, day: int, hour: float) -> Dict:
        altitude, azimuth = self._calc_solar_position(day, hour)
        
        if altitude <= 0:
            return {'GHI': 0, 'I_south': 0, 'I_roof': 0, 'altitude': altitude, 'azimuth': azimuth}
        
        alt_rad = np.radians(altitude)
        zenith_deg = 90 - altitude
        
        if zenith_deg < 90:
            AM = 1.0 / (np.cos(np.radians(zenith_deg)) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))
        else:
            AM = 38.0
        
        tau = 0.70 ** (AM ** 0.678)
        DNI = SOLAR_CONSTANT * tau
        
        I_beam_h = DNI * np.sin(alt_rad)
        I_diff_h = SOLAR_CONSTANT * (1 - tau) * 0.3 * np.sin(alt_rad)
        GHI = I_beam_h + I_diff_h
        
        az_rad = np.radians(azimuth)
        cos_inc = np.cos(alt_rad) * abs(np.cos(az_rad))
        I_beam_south = DNI * cos_inc if cos_inc > 0 else 0
        ground_ref = 0.20 if self.cfg.building_type == BuildingType.ADVANCED else 0.15
        I_diff_south = I_diff_h * 0.5 + GHI * ground_ref * 0.5
        I_south = I_beam_south + I_diff_south
        
        return {'GHI': GHI, 'I_south': I_south, 'I_roof': GHI, 'altitude': altitude, 'azimuth': azimuth}
    
    def _calc_shading(self, solar_alt: float, solar_az: float) -> float:
        if not self.cfg.has_shading or self.cfg.overhang_depth <= 0 or solar_alt <= 0:
            return 0.0
        
        gamma = abs(solar_az - 180)
        if gamma > 90:
            return 1.0
        
        alt_rad = np.radians(solar_alt)
        gamma_rad = np.radians(gamma)
        tan_profile = np.tan(alt_rad) / max(0.01, np.cos(gamma_rad))
        
        if tan_profile < 0.01:
            return 0.0
        
        shadow_length = self.cfg.overhang_depth / np.tan(np.arctan(tan_profile))
        return np.clip(shadow_length / self.cfg.window_height, 0.0, 1.0)
    
    def _calc_ventilation_rate(self, T_out: float, T_in: float, hour: int) -> float:
        ach = self.cfg.ach_min
        is_night = hour < 7 or hour >= 20
        
        if self.cfg.building_type == BuildingType.ADVANCED:
            if self.cfg.t_comfort_lower <= T_out <= self.cfg.t_comfort_upper:
                ach = self.cfg.ach_max
            elif T_in > self.cfg.t_comfort_upper and T_out < T_in - 1:
                ach = self.cfg.ach_max if is_night else self.cfg.ach_max * 0.7
            elif T_out > self.cfg.t_comfort_upper + 2:
                ach = self.cfg.ach_min
        else:
            ach = self.cfg.ach_max
        
        return ach
    
    def simulate_hour(self, day: int, hour: int, T_in: float, T_outdoor_matrix: np.ndarray) -> Dict:
        T_out = T_outdoor_matrix[day % 365, hour % 24]
        solar = self._calc_solar_radiation(day, hour)
        shade_frac = self._calc_shading(solar['altitude'], solar['azimuth'])
        
        # 太阳得热
        I_transmitted = solar['I_south'] * (1 - shade_frac)
        Q_solar_window = I_transmitted * self.cfg.window_area * self.cfg.glass_shgc
        
        # Sol-air温度
        I_wall = solar['I_south'] * (1 - shade_frac * 0.3)
        T_sol_wall = T_out + self.cfg.wall_alpha * I_wall / self.h_out
        T_sol_roof = T_out + self.cfg.roof_alpha * solar['I_roof'] / self.h_out
        
        # 传导
        R_wall = self.cfg.wall_thickness / self.cfg.wall_k + 1/self.h_in + 1/self.h_out
        Q_cond_wall = self.cfg.wall_area * (T_sol_wall - T_in) / R_wall
        Q_cond_window = self.cfg.glass_u * self.cfg.window_area * (T_out - T_in)
        Q_cond_roof = self.cfg.roof_u * self.cfg.roof_area * (T_sol_roof - T_in)
        Q_cond_total = Q_cond_wall + Q_cond_window + Q_cond_roof
        
        # 通风
        ach = self._calc_ventilation_rate(T_out, T_in, hour)
        mass_flow = self.cfg.volume * ach / 3600 * RHO_AIR
        Q_vent = mass_flow * CP_AIR * (T_out - T_in)
        
        # 内热
        start_h, end_h = self.cfg.occupancy_hours
        Q_internal = (self.cfg.internal_gain_occupied if start_h <= hour < end_h 
                      else self.cfg.internal_gain_unoccupied) * self.cfg.floor_area
        
        Q_net = Q_solar_window + Q_cond_total + Q_vent + Q_internal
        Q_envelope_only = Q_solar_window + Q_cond_total + Q_internal
        
        return {
            'day': day, 'hour': hour,
            'T_out': T_out, 'T_in': T_in,
            'Q_solar_window': Q_solar_window,
            'Q_cond_wall': Q_cond_wall,
            'Q_cond_window': Q_cond_window,
            'Q_cond_roof': Q_cond_roof,
            'Q_cond_total': Q_cond_total,
            'Q_vent': Q_vent, 'ACH': ach,
            'Q_internal': Q_internal,
            'Q_net': Q_net,
            'Q_envelope_only': Q_envelope_only,
            'shade_fraction': shade_frac,
            'GHI': solar['GHI'], 'I_south': solar['I_south'],
            'hour_of_year': day * 24 + hour
        }


class AnnualSimulator:
    def __init__(self, building_type: BuildingType):
        self.building_type = building_type
        self.cfg = BuildingConfig(building_type)
        self.engine = ThermalEngine(self.cfg)
        self.T_outdoor = get_outdoor_temp_matrix()
        
    def run(self) -> pd.DataFrame:
        results = []
        T_in = 25.0
        dt = 3600
        
        for day in range(DAYS_IN_YEAR):
            for hour in range(HOURS_IN_DAY):
                result = self.engine.simulate_hour(day, hour, T_in, self.T_outdoor)
                
                dT = result['Q_net'] * dt / self.engine.thermal_mass
                T_in += dT
                
                T_out = result['T_out']
                max_indoor = T_out + 8.0
                min_indoor = T_out - 5.0
                T_in = np.clip(T_in, max(12.0, min_indoor), min(42.0, max_indoor))
                
                result['T_in'] = T_in
                results.append(result)
        
        return pd.DataFrame(results)


# ========== 可视化 ==========

def plot_figure1_material_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    
    materials = ['Single-pane\n(Baseline)', 'Clear IGU', 'Single-Ag\nLow-E', 'Double-Ag\nLow-E', 'Triple-Ag Low-E\n(This Design)']
    shgc_values = [0.86, 0.76, 0.34, 0.27, 0.28]
    u_values = [5.8, 2.7, 1.8, 1.7, 1.6]
    vlt_values = [0.89, 0.81, 0.47, 0.49, 0.64]
    
    x = np.arange(len(materials))
    width = 0.25
    
    ax.bar(x - width, shgc_values, width, label='SHGC', color='coral')
    ax.bar(x, [u/6 for u in u_values], width, label='U-value/6', color='skyblue')
    ax.bar(x + width, vlt_values, width, label='VLT', color='lightgreen')
    
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Fig.1: Glass Material Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(materials, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (s, u, v) in enumerate(zip(shgc_values, u_values, vlt_values)):
        ax.text(i - width, s + 0.03, f'{s:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i, u/6 + 0.03, f'{u:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width, v + 0.03, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')
    
    ax.text(0.02, 0.95, 
            'Triple-Ag Low-E vs Baseline:\n'
            '- SHGC: 67% reduction\n'
            '- U-value: 72% reduction\n'
            '- VLT: 72% retained',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('Sungrove_Fig1_Material_Comparison.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig1_Material_Comparison.png")
    plt.close()


def plot_figure2_annual_temperature(df_baseline: pd.DataFrame, df_advanced: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    daily_baseline = df_baseline.groupby('day')['T_in'].mean()
    daily_advanced = df_advanced.groupby('day')['T_in'].mean()
    daily_outdoor = df_baseline.groupby('day')['T_out'].mean()
    
    days = daily_outdoor.index
    
    ax.plot(days, daily_outdoor, 'k-', linewidth=2, label='Outdoor', alpha=0.7)
    ax.plot(days, daily_baseline, 'r-', linewidth=2.5, label='Baseline Building', alpha=0.9)
    ax.plot(days, daily_advanced, 'b-', linewidth=2.5, label='Energy-Efficient Building', alpha=0.9)
    
    ax.axhline(26, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Comfort Limit 26C')
    ax.axhline(18, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.fill_between(days, 18, 26, alpha=0.1, color='green')
    
    for day, label in [(0, 'Jan'), (90, 'Apr'), (180, 'Jul'), (270, 'Oct')]:
        ax.axvline(day, color='gray', linestyle=':', alpha=0.4)
        ax.text(day + 5, ax.get_ylim()[1] - 1, label, fontsize=11)
    
    ax.set_title('Fig.2: Annual Indoor Temperature (Daily Avg, No HVAC)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day of Year', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 364)
    
    avg_diff = (daily_baseline - daily_advanced).mean()
    max_diff = (daily_baseline - daily_advanced).max()
    overheating_baseline = (daily_baseline > 26).sum()
    overheating_advanced = (daily_advanced > 26).sum()
    
    ax.text(0.02, 0.95, 
            f'Avg Temp Diff: {avg_diff:.1f}C\n'
            f'Max Temp Diff: {max_diff:.1f}C\n'
            f'Overheating Days: {overheating_baseline} vs {overheating_advanced}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('Sungrove_Fig2_Ann_Temp.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig2_Ann_Temp.png")
    plt.close()


def plot_figure3_summer_week(df_baseline: pd.DataFrame, df_advanced: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 7))
    
    start_day, end_day = 180, 187
    sample_baseline = df_baseline[(df_baseline['day'] >= start_day) & (df_baseline['day'] < end_day)].copy()
    sample_advanced = df_advanced[(df_advanced['day'] >= start_day) & (df_advanced['day'] < end_day)].copy()
    
    n_hours = len(sample_baseline)
    hours = np.arange(n_hours)
    
    ax.plot(hours, sample_baseline['T_out'].values, 'k-', linewidth=2.5, label='Outdoor', alpha=0.8)
    ax.plot(hours, sample_baseline['T_in'].values, 'r-', linewidth=2, label='Baseline', alpha=0.9)
    ax.plot(hours, sample_advanced['T_in'].values, 'b-', linewidth=2, label='Efficient', alpha=0.9)
    
    ax.axhline(26, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='26C')
    ax.fill_between(hours, 18, 26, alpha=0.08, color='green')
    
    day_ticks = [i * 24 for i in range(8)]
    day_labels = [f'Day {start_day + i}' for i in range(8)]
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels, fontsize=10)
    
    ax.set_title(f'Fig.3: Summer Week (Day {start_day}-{end_day}, No HVAC)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_hours)
    
    temp_diff = (sample_baseline['T_in'].values - sample_advanced['T_in'].values).mean()
    peak_baseline = sample_baseline['T_in'].max()
    peak_advanced = sample_advanced['T_in'].max()
    outdoor_max = sample_baseline['T_out'].max()
    outdoor_min = sample_baseline['T_out'].min()
    
    ax.text(0.02, 0.95, 
            f'Outdoor: {outdoor_min:.1f}C - {outdoor_max:.1f}C\n'
            f'Avg Diff: {temp_diff:.1f}C\n'
            f'Peak: {peak_baseline:.1f}C vs {peak_advanced:.1f}C',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('Sungrove_Fig3_Summer_Week.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig3_Summer_Week.png")
    plt.close()


def plot_figure4_solar_heat_gain(df_baseline: pd.DataFrame, df_advanced: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    summer_day = 172
    baseline_day = df_baseline[df_baseline['day'] == summer_day]
    advanced_day = df_advanced[df_advanced['day'] == summer_day]
    
    hours = range(24)
    
    ax.fill_between(hours, 0, baseline_day['Q_solar_window']/1000, alpha=0.5, color='red', label='Baseline')
    ax.fill_between(hours, 0, advanced_day['Q_solar_window']/1000, alpha=0.5, color='blue', label='Efficient')
    ax.plot(hours, baseline_day['Q_solar_window']/1000, 'r-', linewidth=2)
    ax.plot(hours, advanced_day['Q_solar_window']/1000, 'b-', linewidth=2)
    
    peak_base = baseline_day['Q_solar_window'].max() / 1000
    peak_adv = advanced_day['Q_solar_window'].max() / 1000
    reduction = (1 - peak_adv / peak_base) * 100 if peak_base > 0 else 0
    
    ax.set_title(f'Fig.4: Solar Heat Gain (Summer Solstice)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Solar Heat Gain (kW)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    ax.text(0.02, 0.95, 
            f'Peak: {peak_base:.1f} vs {peak_adv:.1f} kW\n'
            f'Reduction: {reduction:.0f}%',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('Sungrove_Fig4_Solar_Gain.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig4_Solar_Gain.png")
    plt.close()


def plot_figure5_heat_balance(df_baseline: pd.DataFrame, df_advanced: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    summer_day = 172
    baseline_day = df_baseline[df_baseline['day'] == summer_day]
    advanced_day = df_advanced[df_advanced['day'] == summer_day]
    hours = range(24)
    
    ax = axes[0]
    ax.bar(hours, baseline_day['Q_solar_window']/1000, label='Solar', color='orange', alpha=0.8)
    ax.bar(hours, baseline_day['Q_cond_total']/1000, bottom=baseline_day['Q_solar_window']/1000,
           label='Envelope', color='red', alpha=0.7)
    ax.plot(hours, baseline_day['Q_envelope_only']/1000, 'k-', linewidth=2.5, label='Net')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_title('Baseline Heat Balance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=11)
    ax.set_ylabel('Heat Flow (kW)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    ax = axes[1]
    ax.bar(hours, advanced_day['Q_solar_window']/1000, label='Solar', color='orange', alpha=0.8)
    ax.bar(hours, advanced_day['Q_cond_total']/1000, bottom=advanced_day['Q_solar_window']/1000,
           label='Envelope', color='blue', alpha=0.5)
    ax.plot(hours, advanced_day['Q_envelope_only']/1000, 'k-', linewidth=2.5, label='Net')
    ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax.set_title('Efficient Building Heat Balance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour', fontsize=11)
    ax.set_ylabel('Heat Flow (kW)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)
    
    fig.suptitle('Fig.5: Summer Solstice Heat Balance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Sungrove_Fig5_Heat_Balance.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig5_Heat_Balance.png")
    plt.close()


def plot_figure6_performance_summary(df_baseline: pd.DataFrame, df_advanced: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df_baseline['month'] = df_baseline['day'] // 30
    df_advanced['month'] = df_advanced['day'] // 30
    
    monthly_baseline = df_baseline.groupby('month')['T_in'].mean()
    monthly_advanced = df_advanced.groupby('month')['T_in'].mean()
    monthly_outdoor = df_baseline.groupby('month')['T_out'].mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = np.arange(12)
    width = 0.25
    
    # 月均温度
    ax = axes[0, 0]
    ax.bar(x - width, monthly_outdoor.values[:12], width, label='Outdoor', color='gray', alpha=0.7)
    ax.bar(x, monthly_baseline.values[:12], width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width, monthly_advanced.values[:12], width, label='Efficient', color='blue', alpha=0.7)
    ax.axhline(26, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.set_ylabel('Temperature (C)', fontsize=11)
    ax.set_title('Monthly Avg Indoor Temp', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=9, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 过热时间
    ax = axes[0, 1]
    baseline_overheat = df_baseline[df_baseline['T_in'] > 26].groupby('month').size()
    advanced_overheat = df_advanced[df_advanced['T_in'] > 26].groupby('month').size()
    baseline_overheat = baseline_overheat.reindex(range(12), fill_value=0)
    advanced_overheat = advanced_overheat.reindex(range(12), fill_value=0)
    
    ax.bar(x - width/2, baseline_overheat.values, width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, advanced_overheat.values, width, label='Efficient', color='blue', alpha=0.7)
    ax.set_ylabel('Overheating Hours', fontsize=11)
    ax.set_title('Monthly Overheating (>26C)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=9, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 性能指标
    ax = axes[1, 0]
    
    solar_base = df_baseline['Q_solar_window'].sum()
    solar_adv = df_advanced['Q_solar_window'].sum()
    solar_reduction = (solar_base - solar_adv) / solar_base * 100 if solar_base > 0 else 0
    
    cond_base = df_baseline['Q_cond_total'].sum()
    cond_adv = df_advanced['Q_cond_total'].sum()
    cond_reduction = (abs(cond_base) - abs(cond_adv)) / abs(cond_base) * 100 if cond_base != 0 else 0
    
    temp_baseline_avg = df_baseline['T_in'].mean()
    temp_advanced_avg = df_advanced['T_in'].mean()
    temp_reduction = temp_baseline_avg - temp_advanced_avg
    
    overheat_base = baseline_overheat.sum()
    overheat_adv = advanced_overheat.sum()
    overheat_reduction = (overheat_base - overheat_adv) / overheat_base * 100 if overheat_base > 0 else 0
    
    categories = ['Solar Gain\nReduction', 'Envelope Heat\nReduction', 'Avg Temp\nReduction', 'Overheating\nReduction']
    values = [solar_reduction, max(0, cond_reduction), temp_reduction * 5, overheat_reduction]
    colors = ['coral', 'skyblue', 'lightgreen', 'gold']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    annotations = [f'{solar_reduction:.0f}%', f'{max(0, cond_reduction):.0f}%', f'{temp_reduction:.1f}C', f'{overheat_reduction:.0f}%']
    for bar, ann in zip(bars, annotations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, ann,
                ha='center', fontsize=10, fontweight='bold')
    
    # 汇总
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    PERFORMANCE SUMMARY (No HVAC)
    ==============================
    
    Material Upgrade:
      SHGC: 0.86 -> 0.28 (67% reduction)
      U-value: 5.8 -> 1.6 (72% reduction)
    
    Temperature:
      Avg reduction: {temp_reduction:.1f}C
      Peak reduction: {df_baseline['T_in'].max() - df_advanced['T_in'].max():.1f}C
      Overheating reduced: {overheat_base - overheat_adv:.0f} hours
    
    Solar Heat Gain:
      Annual reduction: {solar_reduction:.0f}%
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax.set_title('Summary', fontsize=12, fontweight='bold')
    
    fig.suptitle('Fig.6: Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Sungrove_Fig6_Perform_Summary.png', dpi=200, bbox_inches='tight')
    print("Saved: Sungrove_Fig6_Perform_Summary.png")
    plt.close()


# ========== 主程序 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("Sungrove Passive Building Thermal Analysis")
    print("=" * 60)
    print("\nBaseline: Single-pane (SHGC=0.86, U=5.8), no shading")
    print("Efficient: Triple-Ag Low-E (SHGC=0.28, U=1.6), shading+thermal mass")
    
    print("\n[1/3] Simulating baseline...")
    sim_baseline = AnnualSimulator(BuildingType.BASELINE)
    df_baseline = sim_baseline.run()
    
    print("[2/3] Simulating efficient building...")
    sim_advanced = AnnualSimulator(BuildingType.ADVANCED)
    df_advanced = sim_advanced.run()
    
    print("[3/3] Generating figures...")
    plot_figure1_material_comparison()
    plot_figure2_annual_temperature(df_baseline, df_advanced)
    plot_figure3_summer_week(df_baseline, df_advanced)
    plot_figure4_solar_heat_gain(df_baseline, df_advanced)
    plot_figure5_heat_balance(df_baseline, df_advanced)
    plot_figure6_performance_summary(df_baseline, df_advanced)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nOutdoor: {df_baseline['T_out'].min():.1f}C - {df_baseline['T_out'].max():.1f}C (avg {df_baseline['T_out'].mean():.1f}C)")
    print(f"Baseline Indoor: {df_baseline['T_in'].min():.1f}C - {df_baseline['T_in'].max():.1f}C (avg {df_baseline['T_in'].mean():.1f}C)")
    print(f"Efficient Indoor: {df_advanced['T_in'].min():.1f}C - {df_advanced['T_in'].max():.1f}C (avg {df_advanced['T_in'].mean():.1f}C)")
    
    temp_diff = df_baseline['T_in'].mean() - df_advanced['T_in'].mean()
    print(f"\nAvg temp reduction: {temp_diff:.1f}C")
    print(f"Peak reduction: {df_baseline['T_in'].max() - df_advanced['T_in'].max():.1f}C")
    
    print("\n" + "=" * 60)