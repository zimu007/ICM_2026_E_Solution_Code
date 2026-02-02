"""
Sungrove University - 三方对比热工模型
对比对象:
1. 优化设计 (Optimized): 高性能材料 + 被动式遮阳
2. 现行规范 (Code Baseline): 符合当前建筑规范的标准设计
3. 普通建筑 (Conventional): 无遮阳,普通材料

改进点:
- 使用真实的建筑材料参数(基于数据表)
- 添加国际标准能耗单位 kWh/(m²·a)
- 修正物理模型逻辑
- 完整的可视化对比分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ==========================================
# 1. 物理常数
# ==========================================
EARTH_TILT_DEG = 23.44
DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
SUNGROVE_LAT = 25.0  # 低纬度(类似迈阿密/台北)

# ==========================================
# 2. 太阳几何计算
# ==========================================
def get_solar_position(day, hour, lat_deg):
    """计算太阳高度角和方位角"""
    declination = np.deg2rad(EARTH_TILT_DEG) * np.sin(2 * np.pi * (284 + day) / 365.0)
    hour_angle = (hour - 12.0) * (np.pi / 12.0)
    lat_rad = np.deg2rad(lat_deg)
    
    sin_alt = (np.sin(lat_rad) * np.sin(declination) + 
               np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    
    # 方位角计算
    cos_alt_safe = max(np.cos(altitude), 1e-6)
    cos_az = (np.sin(altitude) * np.sin(lat_rad) - np.sin(declination)) / (cos_alt_safe * np.cos(lat_rad))
    azimuth = np.arccos(np.clip(cos_az, -1, 1))
    if hour_angle < 0:
        azimuth = -azimuth
        
    return altitude, azimuth

def calculate_solar_irradiance(altitude, day):
    """计算太阳辐射强度 (W/m²)"""
    if altitude <= 0:
        return 0.0
    
    # 大气层外辐射
    solar_constant = 1361.0
    day_angle = 2 * np.pi * day / 365.25
    distance_factor = 1.0 + 0.033 * np.cos(day_angle)
    
    # 大气质量
    zenith = np.pi/2 - altitude
    zenith_deg = np.rad2deg(zenith)
    
    if zenith_deg >= 90:
        return 0.0
    
    air_mass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))
    
    # 大气透过率(清洁大气)
    transmittance = 0.75 ** (air_mass ** 0.678)
    
    # 直接辐射
    direct_normal = solar_constant * distance_factor * transmittance
    direct_horizontal = direct_normal * np.sin(altitude)
    
    # 散射辐射
    diffuse_horizontal = solar_constant * distance_factor * (1 - transmittance) * 0.3
    
    return direct_horizontal + diffuse_horizontal

def calculate_shading_factor(altitude, azimuth, overhang_ratio, window_azimuth=0):
    """计算遮阳系数 (0=无遮挡, 1=完全遮挡)"""
    if altitude <= 0:
        return 1.0
    if overhang_ratio <= 0:
        return 0.0
    
    gamma = azimuth - window_azimuth
    if np.abs(gamma) > np.pi/2:
        return 1.0
    
    cos_gamma = max(np.cos(gamma), 1e-6)
    tan_profile_angle = np.tan(altitude) / cos_gamma
    
    shaded_fraction = overhang_ratio * tan_profile_angle
    return np.clip(shaded_fraction, 0.0, 1.0)

# ==========================================
# 3. 建筑热工模型 - 三种配置
# ==========================================
class BuildingThermalModel:
    """
    物理一致性建筑热工模型
    支持三种配置:优化设计/现行规范/普通建筑
    """
    def __init__(self, name, config_type='optimized'):
        self.name = name
        self.config_type = config_type
        
        # 建筑几何(Academic Hall North)
        self.floor_area = 60 * 24 * 2  # 2880 m²(双层)
        self.volume = 60 * 24 * 7  # 10080 m³
        
        # 围护结构面积
        self.area_wall_south = 60 * 7  # 南向墙体 420 m²
        self.area_roof = 60 * 24  # 屋顶 1440 m²
        self.area_other_walls = (24 * 7 * 2) + (60 * 7)  # 东西墙+北墙 756 m²
        
        # 根据配置类型设置参数
        self._set_parameters()
        
    def _set_parameters(self):
        """根据配置类型设置建筑参数"""
        
        if self.config_type == 'optimized':
            # ===== 优化设计 =====
            # 基于数据表: 双银LOW-E中空玻璃 + 高性能墙体
            
            # 南向窗户(窗墙比45%)
            self.area_glass_south = self.area_wall_south * 0.45  # 189 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south  # 231 m²
            
            # 其他方向窗户(窗墙比30%)
            self.area_glass_other = self.area_other_walls * 0.30  # 227 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other  # 529 m²
            
            # 玻璃参数: 双银LOW-E中空 (6 CED12-53+12A+6C)
            self.u_glass = 1.7  # W/(m²·K)
            self.shgc_glass = 0.27  # 得热系数
            
            # 墙体参数: 外保温墙体
            self.u_wall = 0.35  # W/(m²·K) - 高性能保温
            self.alpha_wall = 0.6  # 浅色外墙吸收率
            
            # 屋顶: 高性能保温
            self.u_roof = 0.25  # W/(m²·K)
            self.alpha_roof = 0.5  # 浅色屋顶
            
            # 遮阳: 优化挑檐
            self.overhang_ratio = 0.8  # D/H = 0.8
            
            # 热容: 大热质量(混凝土暴露)
            self.c_mass = 1.2e9  # J/K
            
        elif self.config_type == 'code_baseline':
            # ===== 现行规范基线 =====
            # 符合当前建筑节能规范的标准设计
            
            # 窗墙比: 南45%, 其他30%
            self.area_glass_south = self.area_wall_south * 0.45  # 189 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south  # 231 m²
            self.area_glass_other = self.area_other_walls * 0.30  # 227 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other  # 529 m²
            
            # 玻璃参数: 阳光控制镀膜中空 (6 CTS140+12A+6c)
            self.u_glass = 2.5  # W/(m²·K)
            self.shgc_glass = 0.38
            
            # 墙体: 自保温墙体
            self.u_wall = 0.60  # W/(m²·K) - 规范要求
            self.alpha_wall = 0.7
            
            # 屋顶: 标准保温
            self.u_roof = 0.40  # W/(m²·K)
            self.alpha_roof = 0.6
            
            # 遮阳: 基本挑檐
            self.overhang_ratio = 0.5  # D/H = 0.5
            
            # 热容: 中等
            self.c_mass = 8.0e8  # J/K
            
        else:  # conventional
            # ===== 普通建筑 =====
            # 无遮阳,普通材料
            
            # 窗墙比: 南50%, 其他35% (更大窗户)
            self.area_glass_south = self.area_wall_south * 0.50  # 210 m²
            self.area_wall_south_net = self.area_wall_south - self.area_glass_south  # 210 m²
            self.area_glass_other = self.area_other_walls * 0.35  # 265 m²
            self.area_other_walls_net = self.area_other_walls - self.area_glass_other  # 491 m²
            
            # 玻璃参数: 透明中空玻璃 (6c+12A+6c)
            self.u_glass = 2.7  # W/(m²·K)
            self.shgc_glass = 0.76  # 高得热
            
            # 墙体: 普通砖墙
            self.u_wall = 1.2  # W/(m²·K) - 无保温
            self.alpha_wall = 0.8  # 深色外墙
            
            # 屋顶: 基础保温
            self.u_roof = 0.65  # W/(m²·K)
            self.alpha_roof = 0.75  # 深色屋顶
            
            # 无遮阳
            self.overhang_ratio = 0.0
            
            # 热容: 较小
            self.c_mass = 6.0e8  # J/K
        
        # 通用参数
        self.h_out = 25.0  # 外表面对流换热系数 W/(m²·K)
        self.air_density = 1.2  # kg/m³
        self.air_cp = 1005.0  # J/(kg·K)
        self.ach_min = 0.5  # 最小换气次数
        self.ach_max = 5.0  # 最大换气次数(夜间通风)
        
        # 百叶窗
        self.blind_trigger = 200.0  # W/m²
        self.blind_shgc_factor = 0.35
        
        # 舒适温度
        self.temp_min = 18.0  # °C
        self.temp_max = 26.0  # °C
    
    def get_internal_gain(self, hour):
        """内部热源: 人员+设备+照明 (W)"""
        if 8 <= hour <= 18:
            return 2.5 * self.floor_area  # 白天 2.5 W/m²
        else:
            return 0.5 * self.floor_area  # 夜间 0.5 W/m²
    
    def run_annual_simulation(self):
        """运行全年模拟"""
        print(f"\n运行 {self.name} ({self.config_type}) 年度模拟...")
        
        T_indoor = 24.0  # 初始室内温度
        dt = 3600  # 时间步长 1小时
        
        results = {
            'hour_idx': [],
            'day': [],
            'hour': [],
            'T_out': [],
            'T_in': [],
            'I_solar': [],
            'Q_solar_gain': [],
            'Q_cond_loss': [],
            'Q_vent_loss': [],
            'Q_internal': [],
            'Q_heating': [],
            'Q_cooling': [],
            'shade_factor': []
        }
        
        hour_counter = 0
        
        for d in range(DAYS_IN_YEAR):
            # 季节性环境温度
            season_offset = -np.cos(2 * np.pi * (d - 30) / 365) * 6.0
            avg_temp_day = 24.0 + season_offset  # 低纬度年均24°C
            
            for h in range(HOURS_IN_DAY):
                # 1. 室外温度(考虑昼夜变化)
                T_out = avg_temp_day + 6.0 * np.cos(2 * np.pi * (h - 15) / 24)
                
                # 2. 太阳辐射
                alt, az = get_solar_position(d, h, SUNGROVE_LAT)
                I_horiz = calculate_solar_irradiance(alt, d)
                
                # 垂直面辐射(南向)
                if alt > 0:
                    cos_inc = np.cos(alt) * np.cos(az)
                    I_vert = max(0, I_horiz * cos_inc / max(np.sin(alt), 1e-4))
                    I_vert += I_horiz * 0.15  # 加散射和反射
                else:
                    I_vert = 0.0
                
                # 3. 遮阳系数
                shade_factor = calculate_shading_factor(alt, az, self.overhang_ratio)
                I_direct_blocked = I_vert * shade_factor
                I_win_incident = I_vert - I_direct_blocked
                I_wall_incident = I_vert - I_direct_blocked
                
                # 4. 百叶窗调节
                shgc_eff = self.shgc_glass
                if I_win_incident > self.blind_trigger:
                    shgc_eff *= self.blind_shgc_factor
                
                # 5. 热量平衡各项
                # 太阳得热(窗户)
                Q_sol_south = I_win_incident * self.area_glass_south * shgc_eff
                Q_sol_other = I_horiz * 0.3 * self.area_glass_other * self.shgc_glass * 0.5
                Q_sol_total = Q_sol_south + Q_sol_other
                
                # 墙体传热(含太阳辐射影响)
                T_sol_wall_south = T_out + (self.alpha_wall * I_wall_incident / self.h_out)
                T_sol_wall_other = T_out + (self.alpha_wall * I_horiz * 0.3 / self.h_out)
                T_sol_roof = T_out + (self.alpha_roof * I_horiz / self.h_out)
                
                Q_cond_wall_south = self.u_wall * self.area_wall_south_net * (T_sol_wall_south - T_indoor)
                Q_cond_wall_other = self.u_wall * self.area_other_walls_net * (T_sol_wall_other - T_indoor)
                Q_cond_glass_south = self.u_glass * self.area_glass_south * (T_out - T_indoor)
                Q_cond_glass_other = self.u_glass * self.area_glass_other * (T_out - T_indoor)
                Q_cond_roof = self.u_roof * self.area_roof * (T_sol_roof - T_indoor)
                
                Q_cond_total = (Q_cond_wall_south + Q_cond_wall_other + 
                               Q_cond_glass_south + Q_cond_glass_other + Q_cond_roof)
                
                # 通风换热(自适应策略)
                if T_out < T_indoor and T_indoor > self.temp_max:
                    # 夜间通风冷却
                    ach = self.ach_max
                else:
                    ach = self.ach_min
                
                m_air = ach * self.volume * self.air_density / 3600
                Q_vent = m_air * self.air_cp * (T_out - T_indoor)
                
                # 内部热源
                Q_internal = self.get_internal_gain(h)
                
                # 6. 总热平衡
                Q_net = Q_sol_total + Q_cond_total + Q_vent + Q_internal
                
                # 7. HVAC需求
                Q_heating = 0.0
                Q_cooling = 0.0
                
                if T_indoor < self.temp_min:
                    Q_heating = self.c_mass * (self.temp_min - T_indoor) / dt
                    Q_net += Q_heating
                elif T_indoor > self.temp_max:
                    Q_cooling = self.c_mass * (T_indoor - self.temp_max) / dt
                    Q_net -= Q_cooling
                
                # 8. 更新室内温度
                dT = Q_net * dt / self.c_mass
                T_indoor += dT
                T_indoor = np.clip(T_indoor, -10, 50)
                
                # 记录结果
                results['hour_idx'].append(hour_counter)
                results['day'].append(d)
                results['hour'].append(h)
                results['T_out'].append(T_out)
                results['T_in'].append(T_indoor)
                results['I_solar'].append(I_vert)
                results['Q_solar_gain'].append(Q_sol_total)
                results['Q_cond_loss'].append(Q_cond_total)
                results['Q_vent_loss'].append(Q_vent)
                results['Q_internal'].append(Q_internal)
                results['Q_heating'].append(Q_heating)
                results['Q_cooling'].append(Q_cooling)
                results['shade_factor'].append(shade_factor)
                
                hour_counter += 1
        
        df = pd.DataFrame(results)
        
        # 计算能耗统计
        total_cooling_Wh = df['Q_cooling'].sum()
        total_heating_Wh = df['Q_heating'].sum()
        total_energy_Wh = total_cooling_Wh + total_heating_Wh
        
        # 转换为kWh
        cooling_kWh = total_cooling_Wh / 1000
        heating_kWh = total_heating_Wh / 1000
        total_kWh = total_energy_Wh / 1000
        
        # 能耗强度 kWh/(m²·a)
        cooling_intensity = cooling_kWh / self.floor_area
        heating_intensity = heating_kWh / self.floor_area
        total_intensity = total_kWh / self.floor_area
        
        print(f"  制冷需求: {cooling_kWh:.1f} kWh ({cooling_intensity:.2f} kWh/(m²·a))")
        print(f"  制热需求: {heating_kWh:.1f} kWh ({heating_intensity:.2f} kWh/(m²·a))")
        print(f"  总能耗: {total_kWh:.1f} kWh ({total_intensity:.2f} kWh/(m²·a))")
        
        self.df_results = df
        self.energy_summary = {
            'cooling_kWh': cooling_kWh,
            'heating_kWh': heating_kWh,
            'total_kWh': total_kWh,
            'cooling_intensity': cooling_intensity,
            'heating_intensity': heating_intensity,
            'total_intensity': total_intensity
        }
        
        return df

# ==========================================
# 4. 可视化函数
# ==========================================
def plot_three_way_comparison(buildings_dict):
    """三方对比可视化"""
    
    fig = plt.figure(figsize=(14, 12))
    
    names = list(buildings_dict.keys())
    x = np.arange(len(names))
    width = 0.25
    
    # === 子图1: 能耗对比柱状图 ===
    ax1 = plt.subplot(3, 2, 1)
    
    cooling_vals = [b.energy_summary['cooling_kWh'] for b in buildings_dict.values()]
    heating_vals = [b.energy_summary['heating_kWh'] for b in buildings_dict.values()]
    total_vals = [b.energy_summary['total_kWh'] for b in buildings_dict.values()]
    
    ax1.bar(x - width, cooling_vals, width, label='Cooling', color='skyblue', edgecolor='black')
    ax1.bar(x, heating_vals, width, label='Heating', color='coral', edgecolor='black')
    ax1.bar(x + width, total_vals, width, label='Total', color='gold', edgecolor='black')
    
    # 添加数值标签
    for i, (c, h, t) in enumerate(zip(cooling_vals, heating_vals, total_vals)):
        ax1.text(i - width, c + 50, f'{c:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i, h + 50, f'{h:.0f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width, t + 50, f'{t:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_ylabel('Annual Energy (kWh)', fontsize=10)
    ax1.set_title('Annual Energy Consumption Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === 子图2: 能耗强度对比 kWh/(m²·a) ===
    ax2 = plt.subplot(3, 2, 2)
    
    cooling_int = [b.energy_summary['cooling_intensity'] for b in buildings_dict.values()]
    heating_int = [b.energy_summary['heating_intensity'] for b in buildings_dict.values()]
    total_int = [b.energy_summary['total_intensity'] for b in buildings_dict.values()]
    
    ax2.bar(x - width, cooling_int, width, label='Cooling', color='deepskyblue', edgecolor='black')
    ax2.bar(x, heating_int, width, label='Heating', color='orangered', edgecolor='black')
    ax2.bar(x + width, total_int, width, label='Total', color='darkgoldenrod', edgecolor='black')
    
    for i, (c, h, t) in enumerate(zip(cooling_int, heating_int, total_int)):
        ax2.text(i - width, c + 1, f'{c:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, h + 1, f'{h:.1f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width, t + 1, f'{t:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylabel('Energy Intensity [kWh/(m²·a)]', fontsize=10)
    ax2.set_title('Energy Intensity Comparison (International Standard)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === 子图3: 节能百分比 ===
    ax3 = plt.subplot(3, 2, 3)
    
    # 以普通建筑为基准计算节能率
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
    ax3.set_title('Energy Savings vs Conventional Building', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === 子图4: 月度制冷需求对比 ===
    ax4 = plt.subplot(3, 2, 4)
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    
    for name, building in buildings_dict.items():
        df = building.df_results
        monthly_cooling = []
        for i in range(12):
            start_day = month_starts[i]
            end_day = month_starts[i+1]
            start_h = start_day * 24
            end_h = end_day * 24
            cooling = df.iloc[start_h:end_h]['Q_cooling'].sum() / 1000  # kWh
            monthly_cooling.append(cooling)
        
        ax4.plot(months, monthly_cooling, marker='o', linewidth=2, label=name, alpha=0.8)
    
    ax4.set_xlabel('Month', fontsize=10)
    ax4.set_ylabel('Cooling Energy (kWh)', fontsize=10)
    ax4.set_title('Monthly Cooling Demand', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # === 子图5: 材料参数对比表 ===
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('off')
    
    # 创建对比表
    table_data = []
    table_data.append(['Parameter', 'Optimized', 'Code Baseline', 'Conventional'])
    table_data.append(['───────────', '─────────', '─────────', '─────────'])
    
    opt = buildings_dict['Optimized']
    code = buildings_dict['Code Baseline']
    conv = buildings_dict['Conventional']
    
    table_data.append(['Window U [W/(m²·K)]', f'{opt.u_glass:.2f}', f'{code.u_glass:.2f}', f'{conv.u_glass:.2f}'])
    table_data.append(['Window SHGC', f'{opt.shgc_glass:.2f}', f'{code.shgc_glass:.2f}', f'{conv.shgc_glass:.2f}'])
    table_data.append(['Wall U [W/(m²·K)]', f'{opt.u_wall:.2f}', f'{code.u_wall:.2f}', f'{conv.u_wall:.2f}'])
    table_data.append(['Roof U [W/(m²·K)]', f'{opt.u_roof:.2f}', f'{code.u_roof:.2f}', f'{conv.u_roof:.2f}'])
    table_data.append(['Overhang D/H', f'{opt.overhang_ratio:.2f}', f'{code.overhang_ratio:.2f}', f'{conv.overhang_ratio:.2f}'])
    table_data.append(['Thermal Mass [MJ/K]', f'{opt.c_mass/1e6:.0f}', f'{code.c_mass/1e6:.0f}', f'{conv.c_mass/1e6:.0f}'])
    
    # 绘制表格
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Building Parameters Comparison', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('Sun_3_Compare.png', dpi=300, bbox_inches='tight')
    print("\n✓ 已生成三方对比图: Sun_3_Compare.png")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("Sungrove University - 三方对比热工分析")
    print("="*60)
    
    # 创建三种建筑配置
    buildings = {
        'Optimized': BuildingThermalModel('Optimized', 'optimized'),
        'Code Baseline': BuildingThermalModel('Code Baseline', 'code_baseline'),
        'Conventional': BuildingThermalModel('Conventional', 'conventional')
    }
    
    # 运行模拟
    for name, building in buildings.items():
        building.run_annual_simulation()
    
    # 生成对比图
    plot_three_way_comparison(buildings)
    
    # 打印总结
    print("\n" + "="*60)
    print("能耗对比总结")
    print("="*60)
    baseline = buildings['Conventional'].energy_summary['total_kWh']
    for name, building in buildings.items():
        total = building.energy_summary['total_kWh']
        savings = (baseline - total) / baseline * 100
        print(f"\n{name}:")
        print(f"  总能耗: {total:.1f} kWh ({building.energy_summary['total_intensity']:.2f} kWh/(m²·a))")
        print(f"  相对普通建筑节能: {savings:.1f}%")
    
    print("\n" + "="*60)