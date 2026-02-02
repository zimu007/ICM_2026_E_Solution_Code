"""
Sungrove University - 窗墙配置对比与眩光分析

问题4: 不同窗墙配置对降温效果的影响
问题5: 眩光问题分析与解决方案
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 基础常数
EARTH_TILT_DEG = 23.44
DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
SUNGROVE_LAT = 25.0

# ==========================================
# 复用太阳计算函数
# ==========================================
def get_solar_position(day, hour, lat_deg):
    """计算太阳高度角和方位角"""
    declination = np.deg2rad(EARTH_TILT_DEG) * np.sin(2 * np.pi * (284 + day) / 365.0)
    hour_angle = (hour - 12.0) * (np.pi / 12.0)
    lat_rad = np.deg2rad(lat_deg)
    
    sin_alt = (np.sin(lat_rad) * np.sin(declination) + 
               np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    
    cos_alt_safe = max(np.cos(altitude), 1e-6)
    cos_az = (np.sin(altitude) * np.sin(lat_rad) - np.sin(declination)) / (cos_alt_safe * np.cos(lat_rad))
    azimuth = np.arccos(np.clip(cos_az, -1, 1))
    if hour_angle < 0:
        azimuth = -azimuth
        
    return altitude, azimuth

def calculate_solar_irradiance(altitude, day):
    """计算太阳辐射强度"""
    if altitude <= 0:
        return 0.0
    
    solar_constant = 1361.0
    day_angle = 2 * np.pi * day / 365.25
    distance_factor = 1.0 + 0.033 * np.cos(day_angle)
    
    zenith = np.pi/2 - altitude
    zenith_deg = np.rad2deg(zenith)
    
    if zenith_deg >= 90:
        return 0.0
    
    air_mass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))
    transmittance = 0.75 ** (air_mass ** 0.678)
    
    direct_normal = solar_constant * distance_factor * transmittance
    direct_horizontal = direct_normal * np.sin(altitude)
    diffuse_horizontal = solar_constant * distance_factor * (1 - transmittance) * 0.3
    
    return direct_horizontal + diffuse_horizontal

def calculate_shading_factor(altitude, azimuth, overhang_ratio):
    """计算遮阳系数"""
    if altitude <= 0:
        return 1.0
    if overhang_ratio <= 0:
        return 0.0
    
    gamma = azimuth
    if np.abs(gamma) > np.pi/2:
        return 1.0
    
    cos_gamma = max(np.cos(gamma), 1e-6)
    tan_profile_angle = np.tan(altitude) / cos_gamma
    
    shaded_fraction = overhang_ratio * tan_profile_angle
    return np.clip(shaded_fraction, 0.0, 1.0)

# ==========================================
# 眩光分析模型
# ==========================================
class GlareAnalysis:
    """
    眩光分析模型
    
    评估指标:
    - 日光眩光概率 (Daylight Glare Probability, DGP)
    - 照度 (Illuminance, lux)
    - 建议的遮阳策略
    """
    
    def __init__(self):
        # 眩光阈值
        self.DGP_IMPERCEPTIBLE = 0.35  # 不可察觉
        self.DGP_PERCEPTIBLE = 0.40    # 可察觉
        self.DGP_DISTURBING = 0.45     # 干扰
        self.DGP_INTOLERABLE = 0.50    # 无法忍受
        
        # 照度阈值(办公环境)
        self.ILLUM_MIN = 300    # lux - 最小推荐照度
        self.ILLUM_OPTIMAL = 500  # lux - 最优照度
        self.ILLUM_MAX = 2000   # lux - 最大舒适照度
    
    def calculate_dgp(self, direct_irradiance, window_area, view_direction_factor=1.0):
        """
        计算日光眩光概率 (DGP)
        
        简化模型基于:
        - 直接太阳辐射
        - 窗户面积
        - 视线方向系数
        """
        # 估算眼部垂直照度 (lux)
        # 1 W/m² ≈ 120 lux (daylight)
        E_v = direct_irradiance * 120 * view_direction_factor
        
        # 简化DGP计算 (Wienold & Christoffersen 2006)
        # DGP = 5.87e-5 * E_v + 9.18e-2 * log(1 + L_s) + constant
        # 这里使用简化版本
        if E_v < 100:
            dgp = 0.30  # 低照度,无眩光
        else:
            # 经验公式
            dgp = 0.35 + 0.00015 * E_v * (window_area / 10.0)
        
        return np.clip(dgp, 0.0, 1.0), E_v
    
    def assess_glare_risk(self, dgp):
        """评估眩光风险等级"""
        if dgp < self.DGP_IMPERCEPTIBLE:
            return "No Glare", "green"
        elif dgp < self.DGP_PERCEPTIBLE:
            return "Imperceptible", "lightgreen"
        elif dgp < self.DGP_DISTURBING:
            return "Perceptible", "yellow"
        elif dgp < self.DGP_INTOLERABLE:
            return "Disturbing", "orange"
        else:
            return "Intolerable", "red"
    
    def recommend_shading(self, dgp, illuminance):
        """推荐遮阳策略"""
        if dgp >= self.DGP_DISTURBING:
            return "Required: Automated blinds or external shading"
        elif dgp >= self.DGP_PERCEPTIBLE:
            return "Recommended: Manual blinds or light shelves"
        elif illuminance > self.ILLUM_MAX:
            return "Optional: Light-diffusing blinds to reduce glare"
        else:
            return "No additional shading needed"

# ==========================================
# 窗户配置对比模型
# ==========================================
class WindowConfigComparison:
    """
    窗户配置对比
    
    对比不同的:
    1. 玻璃类型 (SHGC, U值)
    2. 窗墙比
    3. 窗户朝向分布
    """
    
    def __init__(self):
        self.floor_area = 60 * 24 * 2  # 2880 m²
        
        # 基于数据表的玻璃类型
        self.glass_types = {
            'Clear Double': {
                'name': 'Transparent hollow  (6c+12A+6c)',
                'u_value': 2.7,
                'shgc': 0.76,
                'visible_transmittance': 0.81
            },
            'Tinted Double': {
                'name': 'Green tinted hollow (6 F-Green+12A+6c)',
                'u_value': 2.7,
                'shgc': 0.45,
                'visible_transmittance': 0.66
            },
            'Solar Control': {
                'name': 'Solar control coated hollow (6 CTS140+12A+6c)',
                'u_value': 2.5,
                'shgc': 0.38,
                'visible_transmittance': 0.37
            },
            'Single Low-E': {
                'name': 'Single silver LOW-E hollow (6 CEF11-38+12A+6C)',
                'u_value': 1.7,
                'shgc': 0.27,
                'visible_transmittance': 0.35
            },
            'Double Low-E': {
                'name': 'Double silver LOW-E hollow (6 CED12-53+12A+6C)',
                'u_value': 1.7,
                'shgc': 0.27,
                'visible_transmittance': 0.49
            }
        }
        
        # 墙体参数(固定)
        self.u_wall = 0.35
        self.alpha_wall = 0.6
        self.u_roof = 0.25
        self.alpha_roof = 0.5
        self.h_out = 25.0
        
        # 热容和通风
        self.c_mass = 1.0e9
        self.air_density = 1.2
        self.air_cp = 1005.0
        self.volume = 60 * 24 * 7
        self.ach = 1.0
        
        # 舒适温度
        self.temp_min = 18.0
        self.temp_max = 26.0
        
        # 遮阳
        self.overhang_ratio = 0.8
        self.blind_trigger = 200.0
        self.blind_shgc_factor = 0.35
    
    def run_config_simulation(self, glass_type_key, wwr_south, wwr_other, 
                             simulation_days=7, start_day=180):
        """
        运行特定配置的模拟
        
        Parameters:
        - glass_type_key: 玻璃类型
        - wwr_south: 南向窗墙比
        - wwr_other: 其他方向窗墙比
        - simulation_days: 模拟天数
        - start_day: 起始日(夏季)
        """
        glass = self.glass_types[glass_type_key]
        
        # 计算窗户和墙体面积
        wall_south_total = 60 * 7  # 420 m²
        walls_other_total = (24 * 7 * 2) + (60 * 7)  # 756 m²
        
        area_glass_south = wall_south_total * wwr_south
        area_wall_south = wall_south_total - area_glass_south
        
        area_glass_other = walls_other_total * wwr_other
        area_walls_other = walls_other_total - area_glass_other
        
        area_roof = 60 * 24
        
        # 初始化
        T_indoor = 24.0
        dt = 3600
        
        results = {
            'hour': [],
            'T_out': [],
            'T_in': [],
            'Q_cooling': [],
            'illuminance': [],
            'dgp': []
        }
        
        glare_model = GlareAnalysis()
        
        for d in range(start_day, start_day + simulation_days):
            season_offset = -np.cos(2 * np.pi * (d - 30) / 365) * 6.0
            avg_temp = 24.0 + season_offset
            
            for h in range(HOURS_IN_DAY):
                T_out = avg_temp + 6.0 * np.cos(2 * np.pi * (h - 15) / 24)
                
                # 太阳辐射
                alt, az = get_solar_position(d, h, SUNGROVE_LAT)
                I_horiz = calculate_solar_irradiance(alt, d)
                
                if alt > 0:
                    cos_inc = np.cos(alt) * np.cos(az)
                    I_vert = max(0, I_horiz * cos_inc / max(np.sin(alt), 1e-4))
                    I_vert += I_horiz * 0.15
                else:
                    I_vert = 0.0
                
                # 遮阳
                shade_factor = calculate_shading_factor(alt, az, self.overhang_ratio)
                I_direct_blocked = I_vert * shade_factor
                I_win_incident = I_vert - I_direct_blocked 
                I_wall_incident = I_vert - I_direct_blocked
                
                # 眩光分析
                dgp, illum = glare_model.calculate_dgp(
                    I_win_incident, 
                    area_glass_south,
                    view_direction_factor=max(0, np.cos(az))
                )
                
                # 百叶窗
                shgc_eff = glass['shgc']
                if I_win_incident > self.blind_trigger:
                    shgc_eff *= self.blind_shgc_factor
                
                # 热量平衡
                Q_sol_south = I_win_incident * area_glass_south * shgc_eff
                Q_sol_other = I_horiz * 0.3 * area_glass_other * glass['shgc'] * 0.5
                Q_sol_total = Q_sol_south + Q_sol_other
                
                T_sol_wall_south = T_out + (self.alpha_wall * I_wall_incident / self.h_out)
                T_sol_wall_other = T_out + (self.alpha_wall * I_horiz * 0.3 / self.h_out)
                T_sol_roof = T_out + (self.alpha_roof * I_horiz / self.h_out)
                
                Q_cond = (self.u_wall * area_wall_south * (T_sol_wall_south - T_indoor) +
                         self.u_wall * area_walls_other * (T_sol_wall_other - T_indoor) +
                         glass['u_value'] * area_glass_south * (T_out - T_indoor) +
                         glass['u_value'] * area_glass_other * (T_out - T_indoor) +
                         self.u_roof * area_roof * (T_sol_roof - T_indoor))
                
                m_air = self.ach * self.volume * self.air_density / 3600
                Q_vent = m_air * self.air_cp * (T_out - T_indoor)
                
                Q_internal = 2.5 * self.floor_area if 8 <= h <= 18 else 0.5 * self.floor_area
                
                Q_net = Q_sol_total + Q_cond + Q_vent + Q_internal
                
                # HVAC
                Q_cooling = 0.0
                if T_indoor > self.temp_max:
                    Q_cooling = self.c_mass * (T_indoor - self.temp_max) / dt
                    Q_net -= Q_cooling
                
                # 更新温度
                dT = Q_net * dt / self.c_mass
                T_indoor += dT
                T_indoor = np.clip(T_indoor, 10, 45)
                
                results['hour'].append(h)
                results['T_out'].append(T_out)
                results['T_in'].append(T_indoor)
                results['Q_cooling'].append(Q_cooling)
                results['illuminance'].append(illum)
                results['dgp'].append(dgp)
        
        df = pd.DataFrame(results)
        total_cooling_kWh = df['Q_cooling'].sum() / 1000
        avg_dgp = df['dgp'].mean()
        max_dgp = df['dgp'].max()
        
        return {
            'df': df,
            'cooling_kWh': total_cooling_kWh,
            'cooling_intensity': total_cooling_kWh / self.floor_area * (365 / simulation_days),
            'avg_dgp': avg_dgp,
            'max_dgp': max_dgp,
            'glass_type': glass_type_key,
            'wwr_south': wwr_south,
            'wwr_other': wwr_other
        }

# ==========================================
# 可视化
# ==========================================
def plot_window_comparison():
    """绘制窗户配置对比图"""
    
    model = WindowConfigComparison()
    glare = GlareAnalysis()
    
    print("\n" + "="*60)
    print("窗户配置对比分析")
    print("="*60)
    
    # === 对比1: 不同玻璃类型(固定窗墙比) ===
    print("\n[1] 不同玻璃类型对比 (WWR=45% south, 30% other)")
    glass_comparison = []
    
    for glass_key in model.glass_types.keys():
        result = model.run_config_simulation(glass_key, 0.45, 0.30)
        glass_comparison.append(result)
        print(f"  {glass_key}: {result['cooling_intensity']:.1f} kWh/(m²·a), "
              f"Avg DGP={result['avg_dgp']:.3f}, Max DGP={result['max_dgp']:.3f}")
    
    # === 对比2: 不同窗墙比(固定玻璃类型) ===
    print("\n[2] 不同窗墙比对比 (使用Double Low-E玻璃)")
    wwr_comparison = []
    wwr_values = [0.30, 0.40, 0.45, 0.50, 0.60]
    
    for wwr in wwr_values:
        result = model.run_config_simulation('Double Low-E', wwr, wwr * 0.7 )
        wwr_comparison.append(result)
        print(f"  WWR={wwr:.0%}: {result['cooling_intensity']:.1f} kWh/(m²·a), "
              f"Max DGP={result['max_dgp']:.3f}")
    
    # === 绘图 ===
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 玻璃类型对制冷能耗的影响
    ax1 = plt.subplot(2, 3, 1)
    glass_names = [model.glass_types[r['glass_type']]['name'] for r in glass_comparison]
    glass_names = [name.split('(')[0].strip() for name in glass_names]  # 简化名称
    cooling_vals = [r['cooling_intensity'] for r in glass_comparison]
    
    bars = ax1.bar(range(len(glass_names)), cooling_vals, 
                   color=['lightcoral', 'khaki', 'lightblue', 'lightgreen', 'darkseagreen'],
                   edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, cooling_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Cooling Intensity [kWh/(m²·a)]', fontsize=10)
    ax1.set_title('Glass Type Impact on Cooling', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(len(glass_names)))
    ax1.set_xticklabels(glass_names, rotation=15, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 玻璃类型对眩光的影响
    ax2 = plt.subplot(2, 3, 2)
    dgp_vals = [r['max_dgp'] for r in glass_comparison]
    
    colors_dgp = []
    for dgp in dgp_vals:
        _, color = glare.assess_glare_risk(dgp)
        colors_dgp.append(color)
    
    bars = ax2.bar(range(len(glass_names)), dgp_vals, color=colors_dgp, 
                   edgecolor='black', linewidth=1.5)
    
    ax2.axhline(glare.DGP_PERCEPTIBLE, color='orange', linestyle='--', 
                linewidth=1.5, label='Perceptible Threshold')
    ax2.axhline(glare.DGP_DISTURBING, color='red', linestyle='--', 
                linewidth=1.5, label='Disturbing Threshold')
    
    for bar, val in zip(bars, dgp_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Maximum DGP', fontsize=10)
    ax2.set_title('Glass Type Impact on Glare', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(glass_names)))
    ax2.set_xticklabels(glass_names, rotation=15, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 玻璃性能散点图
    ax3 = plt.subplot(2, 3, 3)
    shgc_vals = [model.glass_types[r['glass_type']]['shgc'] for r in glass_comparison]
    u_vals = [model.glass_types[r['glass_type']]['u_value'] for r in glass_comparison]
    
    scatter = ax3.scatter(shgc_vals, cooling_vals, s=200, c=dgp_vals, 
                         cmap='RdYlGn_r', edgecolors='black', linewidths=2)
    
    for i, name in enumerate(glass_names):
        ax3.annotate(name, (shgc_vals[i], cooling_vals[i]), 
                    fontsize=7, ha='center', va='bottom')
    
    ax3.set_xlabel('SHGC (Solar Heat Gain Coefficient)', fontsize=10)
    ax3.set_ylabel('Cooling Intensity [kWh/(m²·a)]', fontsize=10)
    ax3.set_title('Glass Performance Trade-off', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Max DGP', fontsize=9)
    
    # 子图4: 窗墙比对制冷能耗的影响
    ax4 = plt.subplot(2, 3, 4)
    wwr_pct = [r['wwr_south'] * 100 for r in wwr_comparison]
    wwr_cooling = [r['cooling_intensity'] for r in wwr_comparison]
    
    ax4.plot(wwr_pct, wwr_cooling, marker='o', linewidth=2.5, 
            markersize=10, color='steelblue', markeredgecolor='black', markeredgewidth=1.5)
    ax4.fill_between(wwr_pct, 0, wwr_cooling, alpha=0.2, color='steelblue')
    
    for x, y in zip(wwr_pct, wwr_cooling):
        ax4.text(x, y + 0.5, f'{y:.1f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Window-to-Wall Ratio (%)', fontsize=10)
    ax4.set_ylabel('Cooling Intensity [kWh/(m²·a)]', fontsize=10)
    ax4.set_title('WWR Impact on Cooling (Double Low-E)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 窗墙比对眩光的影响
    ax5 = plt.subplot(2, 3, 5)
    wwr_dgp = [r['max_dgp'] for r in wwr_comparison]
    
    ax5.plot(wwr_pct, wwr_dgp, marker='s', linewidth=2.5, 
            markersize=10, color='coral', markeredgecolor='black', markeredgewidth=1.5)
    
    ax5.axhline(glare.DGP_PERCEPTIBLE, color='orange', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Perceptible')
    ax5.axhline(glare.DGP_DISTURBING, color='red', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Disturbing')
    ax5.fill_between(wwr_pct, 0, glare.DGP_PERCEPTIBLE, 
                    color='green', alpha=0.1, label='Acceptable')
    
    for x, y in zip(wwr_pct, wwr_dgp):
        ax5.text(x, y + 0.01, f'{y:.2f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    ax5.set_xlabel('Window-to-Wall Ratio (%)', fontsize=10)
    ax5.set_ylabel('Maximum DGP', fontsize=10)
    ax5.set_title('WWR Impact on Glare', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8, loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 推荐配置总结
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 找出最优配置
    best_glass = min(glass_comparison, key=lambda x: x['cooling_intensity'])
    best_wwr = min(wwr_comparison, key=lambda x: x['cooling_intensity'])
    
    summary_text = f"""
    === Window Configuration Optimization Suggestions ===
    
    [Best Glass Type]
    {model.glass_types[best_glass['glass_type']]['name']}
    • SHGC: {model.glass_types[best_glass['glass_type']]['shgc']:.2f}
    • U-value: {model.glass_types[best_glass['glass_type']]['u_value']:.1f} W/(m²·K)
    • Cooling intensity: {best_glass['cooling_intensity']:.1f} kWh/(m²·a)
    • Maximum DGP: {best_glass['max_dgp']:.2f}
    
    [Best Window-to-Wall Ratio]
    South: {best_wwr['wwr_south']:.0%}
    Other: {best_wwr['wwr_other']:.0%}
    • Cooling intensity: {best_wwr['cooling_intensity']:.1f} kWh/(m²·a)
    • Maximum DGP: {best_wwr['max_dgp']:.2f}
    
    [Glare Control Suggestions]
    • Use low SHGC glass (≤0.30)
    • South WWR suggested ≤45%
    • Use automatic blind system
    •Consider light shelf design
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('Window_Analysis.png', 
                dpi=300, bbox_inches='tight')
    print("\n✓ 已生成窗户配置和眩光分析图")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    plot_window_comparison()
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)