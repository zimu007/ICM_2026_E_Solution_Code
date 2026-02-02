"""
Borealis University - 高纬地区三方对比热工模型

重点:
- 冬季供暖优化
- 热质量蓄热
- 季节性遮阳策略
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 常数
EARTH_TILT_DEG = 23.44
DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
BOREALIS_LAT = 65.0  # 高纬度

# ==========================================
# 太阳计算(复用)
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
    """计算太阳辐射"""
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
    
    # 高纬度大气透过率降低
    transmittance = 0.65 ** (air_mass ** 0.70)
    
    direct_normal = solar_constant * distance_factor * transmittance
    direct_horizontal = direct_normal * np.sin(altitude)
    diffuse_horizontal = solar_constant * distance_factor * (1 - transmittance) * 0.35
    
    return direct_horizontal + diffuse_horizontal

def calculate_shading_factor(altitude, azimuth, overhang_ratio):
    """计算遮阳系数"""
    if altitude <= 0:
        return 0.0  # 冬季低角度太阳应该能够照射进来
    if overhang_ratio <= 0:
        return 0.0
    
    gamma = azimuth
    if np.abs(gamma) > np.pi/2:
        return 0.0
    
    cos_gamma = max(np.cos(gamma), 1e-6)
    tan_profile_angle = np.tan(altitude) / cos_gamma
    
    shaded_fraction = overhang_ratio * tan_profile_angle
    return np.clip(shaded_fraction, 0.0, 1.0)

# ==========================================
# 建筑模型
# ==========================================
class BorealisBuilding:
    """Borealis大学建筑热工模型"""
    
    def __init__(self, name, config_type='optimized'):
        self.name = name
        self.config_type = config_type
        
        # 几何
        self.floor_area = 60 * 24 * 2  # 2880 m²
        self.volume = 60 * 24 * 7  # 10080 m³
        
        # 围护结构面积
        self.area_wall_south = 60 * 7  # 420 m²
        self.area_roof = 60 * 24  # 1440 m²
        self.area_other_walls = (24 * 7 * 2) + (60 * 7)  # 756 m²
        
        self._set_parameters()
        
    def _set_parameters(self):
        """设置参数"""
        
        if self.config_type == 'optimized':
            # ===== 优化设计: 极致保温 + 大热质量 =====
            
            # 窗墙比(南向偏大以获取太阳得热)
            self.area_glass_south = self.area_wall_south * 0.50  # 210 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south
            
            # 其他方向窗墙比小(减少热损失)
            self.area_glass_other = self.area_other_walls * 0.20  # 151 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other
            
            # 玻璃: 三银LOW-E中空 (6 CED13-45+12A+6C)
            self.u_glass = 1.6  # W/(m²·K) - 最优保温
            self.shgc_glass = 0.28  # 适中得热
            
            # 墙体: 超厚保温
            self.u_wall = 0.15  # W/(m²·K)
            self.alpha_wall = 0.8  # 深色外墙(吸收冬季太阳热)
            
            # 屋顶: 超厚保温
            self.u_roof = 0.12  # W/(m²·K)
            self.alpha_roof = 0.8
            
            # 遮阳: 季节性(夏季遮阳,冬季允许)
            self.overhang_ratio_summer = 0.6
            self.overhang_ratio_winter = 0.0
            
            # 热质量: 极大(混凝土+蓄热材料)
            self.c_mass_air = 1.0e9  # 空气节点热容
            self.c_mass_thermal = 3.0e9  # 热质量节点热容
            self.r_coupling = 0.01  # 空气-热质量耦合热阻
            
        elif self.config_type == 'code_baseline':
            # ===== 现行规范: 标准保温 =====
            
            self.area_glass_south = self.area_wall_south * 0.40  # 168 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south
            self.area_glass_other = self.area_other_walls * 0.25  # 189 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other
            
            # 玻璃: 双银LOW-E
            self.u_glass = 1.7  # W/(m²·K)
            self.shgc_glass = 0.27
            
            # 墙体: 标准寒冷地区保温
            self.u_wall = 0.30  # W/(m²·K)
            self.alpha_wall = 0.7
            
            # 屋顶
            self.u_roof = 0.20  # W/(m²·K)
            self.alpha_roof = 0.7
            
            # 遮阳
            self.overhang_ratio_summer = 0.4
            self.overhang_ratio_winter = 0.0
            
            # 热质量: 中等
            self.c_mass_air = 8.0e8
            self.c_mass_thermal = 1.5e9
            self.r_coupling = 0.02
            
        else:  # conventional
            # ===== 普通建筑: 基础保温 =====
            
            self.area_glass_south = self.area_wall_south * 0.35  # 147 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south
            self.area_glass_other = self.area_other_walls * 0.30  # 227 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other
            
            # 玻璃: 普通中空
            self.u_glass = 2.7  # W/(m²·K)
            self.shgc_glass = 0.76
            
            # 墙体: 基础保温
            self.u_wall = 0.60  # W/(m²·K)
            self.alpha_wall = 0.6
            
            # 屋顶
            self.u_roof = 0.40  # W/(m²·K)
            self.alpha_roof = 0.6
            
            # 无季节性遮阳
            self.overhang_ratio_summer = 0.0
            self.overhang_ratio_winter = 0.0
            
            # 热质量: 小
            self.c_mass_air = 6.0e8
            self.c_mass_thermal = 8.0e8
            self.r_coupling = 0.05
        
        # 通用参数
        self.h_out = 25.0
        self.air_density = 1.2
        self.air_cp = 1005.0
        self.ach_min = 0.3  # 高纬度更低换气
        self.ach_max = 2.0
        
        # 舒适温度
        self.temp_min = 20.0  # 冬季供暖
        self.temp_max = 24.0  # 夏季制冷
    
    def get_internal_gain(self, hour):
        """内部热源"""
        if 8 <= hour <= 18:
            return 3.0 * self.floor_area  # 3 W/m²
        else:
            return 0.8 * self.floor_area
    
    def run_annual_simulation(self):
        """运行全年模拟"""
        print(f"\n运行 {self.name} ({self.config_type}) 年度模拟...")
        
        # 双节点: 空气 + 热质量
        T_air = 5.0
        T_mass = 5.0
        dt = 3600
        
        results = {
            'hour_idx': [],
            'day': [],
            'hour': [],
            'T_out': [],
            'T_air': [],
            'T_mass': [],
            'I_solar': [],
            'Q_solar_gain': [],
            'Q_heating': [],
            'Q_cooling': [],
            'Q_mass_exchange': []
        }
        
        hour_counter = 0
        
        for d in range(DAYS_IN_YEAR):
            # 环境温度
            season_offset = -np.cos(2 * np.pi * (d - 30) / 365) * 16.0
            avg_temp = 5.0 + season_offset
            
            # 季节性遮阳切换
            if 120 < d < 260:  # 夏季
                overhang_ratio = self.overhang_ratio_summer
            else:
                overhang_ratio = self.overhang_ratio_winter
            
            for h in range(HOURS_IN_DAY):
                T_out = avg_temp + 4.0 * np.cos(2 * np.pi * (h - 14) / 24)
                
                # 太阳辐射
                alt, az = get_solar_position(d, h, BOREALIS_LAT)
                I_horiz = calculate_solar_irradiance(alt, d)
                
                if alt > 0:
                    cos_inc = np.cos(alt) * np.cos(az)
                    I_vert = max(0, I_horiz * cos_inc / max(np.sin(alt), 1e-4))
                    I_vert += I_horiz * 0.20
                else:
                    I_vert = 0.0
                
                # 遮阳
                shade_factor = calculate_shading_factor(alt, az, overhang_ratio)
                I_win_incident = I_vert * (1 - shade_factor)
                I_wall_incident = I_vert * (1 - shade_factor)
                
                # 雪反射(冬季)
                if d < 90 or d > 300:
                    albedo = 0.7
                else:
                    albedo = 0.2
                I_reflected = I_horiz * albedo
                I_win_incident += I_reflected * 0.5
                
                # 太阳得热
                Q_sol_south = I_win_incident * self.area_glass_south * self.shgc_glass
                Q_sol_other = (I_horiz * 0.2 + I_reflected * 0.3) * self.area_glass_other * self.shgc_glass * 0.5
                Q_sol_total = Q_sol_south + Q_sol_other
                
                # 一半太阳热量直接加热空气,一半加热热质量
                Q_sol_to_air = Q_sol_total * 0.3
                Q_sol_to_mass = Q_sol_total * 0.7
                
                # 墙体传热
                T_sol_wall_south = T_out + (self.alpha_wall * I_wall_incident / self.h_out)
                T_sol_wall_other = T_out + (self.alpha_wall * I_horiz * 0.2 / self.h_out)
                T_sol_roof = T_out + (self.alpha_roof * I_horiz / self.h_out)
                
                Q_cond_air = (self.u_wall * self.area_wall_south_net * (T_sol_wall_south - T_air) +
                             self.u_wall * self.area_other_walls_net * (T_sol_wall_other - T_air) +
                             self.u_glass * self.area_glass_south * (T_out - T_air) +
                             self.u_glass * self.area_glass_other * (T_out - T_air) +
                             self.u_roof * self.area_roof * (T_sol_roof - T_air))
                
                # 通风
                if T_out < T_air and T_air > self.temp_max and d > 120 and d < 260:
                    ach = self.ach_max
                else:
                    ach = self.ach_min
                
                m_air = ach * self.volume * self.air_density / 3600
                Q_vent = m_air * self.air_cp * (T_out - T_air)
                
                # 内部热源
                Q_internal = self.get_internal_gain(h)
                
                # 空气-热质量耦合
                Q_coupling = (T_mass - T_air) / self.r_coupling
                
                # 空气节点热平衡
                Q_air_net = Q_sol_to_air + Q_cond_air + Q_vent + Q_internal + Q_coupling
                
                # 热质量节点热平衡
                Q_mass_net = Q_sol_to_mass - Q_coupling
                
                # HVAC
                Q_heating = 0.0
                Q_cooling = 0.0
                
                if T_air < self.temp_min:
                    Q_heating = self.c_mass_air * (self.temp_min - T_air) / dt
                    Q_air_net += Q_heating
                elif T_air > self.temp_max:
                    Q_cooling = self.c_mass_air * (T_air - self.temp_max) / dt
                    Q_air_net -= Q_cooling
                
                # 更新温度
                dT_air = Q_air_net * dt / self.c_mass_air
                T_air += dT_air
                T_air = np.clip(T_air, -30, 40)
                
                dT_mass = Q_mass_net * dt / self.c_mass_thermal
                T_mass += dT_mass
                T_mass = np.clip(T_mass, -30, 50)
                
                # 记录
                results['hour_idx'].append(hour_counter)
                results['day'].append(d)
                results['hour'].append(h)
                results['T_out'].append(T_out)
                results['T_air'].append(T_air)
                results['T_mass'].append(T_mass)
                results['I_solar'].append(I_vert)
                results['Q_solar_gain'].append(Q_sol_total)
                results['Q_heating'].append(Q_heating)
                results['Q_cooling'].append(Q_cooling)
                results['Q_mass_exchange'].append(Q_coupling)
                
                hour_counter += 1
        
        df = pd.DataFrame(results)
        
        # 能耗统计
        heating_kWh = df['Q_heating'].sum() / 1000
        cooling_kWh = df['Q_cooling'].sum() / 1000
        total_kWh = heating_kWh + cooling_kWh
        
        heating_intensity = heating_kWh / self.floor_area
        cooling_intensity = cooling_kWh / self.floor_area
        total_intensity = total_kWh / self.floor_area
        
        print(f"  制热需求: {heating_kWh:.1f} kWh ({heating_intensity:.2f} kWh/(m²·a))")
        print(f"  制冷需求: {cooling_kWh:.1f} kWh ({cooling_intensity:.2f} kWh/(m²·a))")
        print(f"  总能耗: {total_kWh:.1f} kWh ({total_intensity:.2f} kWh/(m²·a))")
        
        self.df_results = df
        self.energy_summary = {
            'heating_kWh': heating_kWh,
            'cooling_kWh': cooling_kWh,
            'total_kWh': total_kWh,
            'heating_intensity': heating_intensity,
            'cooling_intensity': cooling_intensity,
            'total_intensity': total_intensity
        }
        
        return df

# ==========================================
# 可视化
# ==========================================
def plot_borealis_comparison(buildings_dict):
    """Borealis三方对比可视化"""
    
    fig = plt.figure(figsize=(14, 10))
    
    names = list(buildings_dict.keys())
    x = np.arange(len(names))
    width = 0.25
    
    # === 子图1: 能耗对比 ===
    ax1 = plt.subplot(2, 2, 1)
    
    heating_vals = [b.energy_summary['heating_kWh'] for b in buildings_dict.values()]
    cooling_vals = [b.energy_summary['cooling_kWh'] for b in buildings_dict.values()]
    total_vals = [b.energy_summary['total_kWh'] for b in buildings_dict.values()]
    
    ax1.bar(x - width, heating_vals, width, label='Heating', color='orangered', edgecolor='black')
    ax1.bar(x, cooling_vals, width, label='Cooling', color='skyblue', edgecolor='black')
    ax1.bar(x + width, total_vals, width, label='Total', color='gold', edgecolor='black')
    
    for i, (h, c, t) in enumerate(zip(heating_vals, cooling_vals, total_vals)):
        ax1.text(i - width, h + 500, f'{h:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, c + 500, f'{c:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, t + 500, f'{t:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Annual Energy (kWh)', fontsize=10)
    ax1.set_title('Annual Energy Consumption', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === 子图2: 能耗强度 ===
    ax2 = plt.subplot(2, 2, 2)
    
    heating_int = [b.energy_summary['heating_intensity'] for b in buildings_dict.values()]
    cooling_int = [b.energy_summary['cooling_intensity'] for b in buildings_dict.values()]
    total_int = [b.energy_summary['total_intensity'] for b in buildings_dict.values()]
    
    ax2.bar(x - width, heating_int, width, label='Heating', color='darkorange', edgecolor='black')
    ax2.bar(x, cooling_int, width, label='Cooling', color='deepskyblue', edgecolor='black')
    ax2.bar(x + width, total_int, width, label='Total', color='darkgoldenrod', edgecolor='black')
    
    for i, (h, c, t) in enumerate(zip(heating_int, cooling_int, total_int)):
        ax2.text(i - width, h + 2, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, c + 2, f'{c:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width, t + 2, f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Energy Intensity [kWh/(m²·a)]', fontsize=10)
    ax2.set_title('Energy Intensity Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === 子图3: 节能百分比 ===
    ax3 = plt.subplot(2, 2, 3)
    
    baseline_total = buildings_dict['Conventional'].energy_summary['total_kWh']
    savings_pct = [(baseline_total - b.energy_summary['total_kWh']) / baseline_total * 100 
                   for b in buildings_dict.values()]
    
    colors = ['green' if s > 0 else 'red' for s in savings_pct]
    bars = ax3.bar(names, savings_pct, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, pct in zip(bars, savings_pct):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', 
                va='bottom' if pct > 0 else 'top', fontsize=10, fontweight='bold')
    
    ax3.axhline(0, color='black', linewidth=1)
    ax3.set_ylabel('Energy Savings (%)', fontsize=10)
    ax3.set_title('Energy Savings vs Conventional', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === 子图4: 参数对比表 ===
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    table_data = []
    table_data.append(['Parameter', 'Optimized', 'Code Baseline', 'Conventional'])
    table_data.append(['────────', '────────', '────────', '────────'])
    
    opt = buildings_dict['Optimized']
    code = buildings_dict['Code Baseline']
    conv = buildings_dict['Conventional']
    
    table_data.append(['Window U [W/(m²·K)]', f'{opt.u_glass:.2f}', f'{code.u_glass:.2f}', f'{conv.u_glass:.2f}'])
    table_data.append(['Window SHGC', f'{opt.shgc_glass:.2f}', f'{code.shgc_glass:.2f}', f'{conv.shgc_glass:.2f}'])
    table_data.append(['Wall U [W/(m²·K)]', f'{opt.u_wall:.2f}', f'{code.u_wall:.2f}', f'{conv.u_wall:.2f}'])
    table_data.append(['Roof U [W/(m²·K)]', f'{opt.u_roof:.2f}', f'{code.u_roof:.2f}', f'{conv.u_roof:.2f}'])
    table_data.append(['Thermal Mass [GJ/K]', f'{opt.c_mass_thermal/1e9:.1f}', 
                      f'{code.c_mass_thermal/1e9:.1f}', f'{conv.c_mass_thermal/1e9:.1f}'])
    table_data.append(['South WWR', f'{opt.area_glass_south/opt.area_wall_south:.0%}', 
                      f'{code.area_glass_south/code.area_wall_south:.0%}', 
                      f'{conv.area_glass_south/conv.area_wall_south:.0%}'])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Building Parameters Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Bo_3_Compare.png', dpi=300, bbox_inches='tight')
    print("\n✓ 已生成Borealis三方对比图")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("Borealis University - 三方对比热工分析")
    print("="*60)
    
    buildings = {
        'Optimized': BorealisBuilding('Optimized', 'optimized'),
        'Code Baseline': BorealisBuilding('Code Baseline', 'code_baseline'),
        'Conventional': BorealisBuilding('Conventional', 'conventional')
    }
    
    for name, building in buildings.items():
        building.run_annual_simulation()
    
    plot_borealis_comparison(buildings)
    
    print("\n" + "="*60)
    print("能耗对比总结")
    print("="*60)
    baseline = buildings['Conventional'].energy_summary['total_kWh']
    for name, building in buildings.items():
        total = building.energy_summary['total_kWh']
        savings = (baseline - total) / baseline * 100
        print(f"\n{name}:")
        print(f"  总能耗: {total:.1f} kWh ({building.energy_summary['total_intensity']:.2f} kWh/(m²·a))")
        print(f"  制热: {building.energy_summary['heating_kWh']:.1f} kWh ({building.energy_summary['heating_intensity']:.2f} kWh/(m²·a))")
        print(f"  相对普通建筑节能: {savings:.1f}%")
    
    print("\n" + "="*60)