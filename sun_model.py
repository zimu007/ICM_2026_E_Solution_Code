import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False

try:
    from solar import UNIVERSITIES, EARTH_TILT_DEG, DAYS_IN_YEAR, HOURS_IN_DAY
except ImportError:
    raise ImportError('无法找到\'solar.p\'')

# 太阳几何
def get_precise_solar_position(day, hour, lat_deg):
    declination = np.deg2rad(EARTH_TILT_DEG) * np.sin(2 * np.pi * (284 + day) / 365.0)
    hour_angle = (hour - 12.0) * (np.pi / 12.0)
    lat_rad = np.deg2rad(lat_deg)
    
    sin_alt = (np.sin(lat_rad) * np.sin(declination) + 
               np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    
    cos_alt_safe = np.cos(altitude) if np.abs(np.cos(altitude)) > 1e-6 else 1e-6
    
    cos_az = (np.sin(altitude) * np.sin(lat_rad) - np.sin(declination)) / (cos_alt_safe * np.cos(lat_rad))
    azimuth = np.arccos(np.clip(cos_az, -1, 1))
    if hour_angle < 0: azimuth = -azimuth
        
    return altitude, azimuth

def calculate_shading_factor(altitude, azimuth, overhang_ratio, window_azimuth=0):
    if altitude <= 0: return 1.0
    if overhang_ratio <= 0: return 0.0
    
    gamma = azimuth - window_azimuth
    if np.abs(gamma) > np.pi/2: return 1.0
    
    cos_gamma = np.cos(gamma) if np.abs(np.cos(gamma)) > 1e-6 else 1e-6
    
    tan_profile_angle = np.tan(altitude) / cos_gamma
    
    shaded_fraction = overhang_ratio * tan_profile_angle
    return np.clip(shaded_fraction, 0.0, 1.0)


class PhysicallyConsistentBuilding:
    def __init__(self, name, lat, base_temp, dh_ratio, c_mass_factor=1.0, warming_offset=0.0):
        self.name = name
        self.lat = lat
        self.base_temp = base_temp
        self.dh_ratio = dh_ratio 
        self.warming_offset = warming_offset
        
        # 几何结构
        self.floor_area = 60 * 24 * 2
        self.volume = 60 * 24 * 7
        
        # 南墙
        self.area_wall_south = 480.0 
        self.u_wall = 1.2
        self.alpha_wall = 0.7
        self.h_out = 25.0
        
        # 南窗
        self.area_glass_south = 390.0
        self.u_glass = 2.8
        self.shgc_glass = 0.65 
        
        # 屋顶
        self.area_roof = 60 * 24
        self.u_roof = 0.4
        self.alpha_roof = 0.6
        
        # 其他墙
        self.area_other = 235.0
        self.u_other = 1.0
        
        # 其他窗
        self.area_other_glass = 50.0
        self.u_other_glass = 2.8
        self.shgc_other_glass = 0.65
        
        # 热容
        self.c_mass = 8.0e8 * c_mass_factor 
        
        # 通风
        self.air_density = 1.2
        self.air_cp = 1005.0
        self.ach_min = 0.5
        self.ach_max = 5.0
        
        # 百叶
        self.blind_trigger = 200.0
        self.blind_shgc_factor = 0.4
        self.temp_min = 18.0
        self.temp_max = 26.0

    def get_internal_gain(self, hour):
        if 8 <= hour <= 18:
            return 25.0 * self.floor_area * 0.1
        else:
            return 5.0 * self.floor_area * 0.1

    def run_year(self):
        T_indoor = self.base_temp
        T_indoor_no_shade = self.base_temp
        dt = 3600
        
        results = {'hour_idx': [], 'T_out': [], 'T_in': [], 'T_in_no_shade': [],
                   'Q_sol_win': [], 'Q_cond': [], 'Q_vent': [],
                   'Cooling_Load': [], 'Heating_Load': [],
                   'Cooling_Load_no_shade': [], 'Heating_Load_no_shade': []}
        
        for d in range(DAYS_IN_YEAR):
            season_offset = -np.cos(2 * np.pi * (d - 30) / 365) * (15 if self.lat > 50 else 5)
            avg_temp = self.base_temp + season_offset + self.warming_offset
            
            for h in range(HOURS_IN_DAY):
                T_out = avg_temp + 5 * np.cos(2 * np.pi * (h - 15) / 24)
                
                # 太阳辐射
                alt, az = get_precise_solar_position(d, h, self.lat)
                I_dir_vert = 0.0; I_horiz_global = 0.0
                if alt > 0:
                    I_ext = 1361 * np.sin(alt)
                    diffuse = 0.1 + 0.1 * (1 - np.sin(alt))
                    I_dir_h = I_ext * (1 - diffuse)
                    I_diff_h = I_ext * diffuse
                    I_horiz_global = I_dir_h + I_diff_h 
                    cos_inc = np.cos(alt) * np.cos(az) 
                    if cos_inc > 0:
                        I_dir_vert = (I_ext * (1 - diffuse) / max(np.sin(alt), 1e-4)) * cos_inc
                    I_vert_global = I_dir_vert + (I_diff_h * 0.5) + (I_horiz_global * 0.2 * 0.5)
                else:
                    I_vert_global = 0.0; I_horiz_global = 0.0

                # 遮阳
                f_shade = calculate_shading_factor(alt, az, self.dh_ratio)
                I_dir_blocked = I_dir_vert * f_shade

                I_win_incident = I_vert_global - I_dir_blocked
                shgc_curr = self.shgc_glass
                if I_win_incident > self.blind_trigger: 
                    shgc_curr *= self.blind_shgc_factor

                I_wall_incident = I_vert_global - I_dir_blocked

                # 无遮阳
                I_win_incident_no = I_vert_global
                I_wall_incident_no = I_vert_global
                shgc_curr_no = self.shgc_glass

                # 热平衡 (有遮阳)
                Q_sol_win = I_win_incident * self.area_glass_south * shgc_curr

                T_sol_wall = T_out + (self.alpha_wall * I_wall_incident / self.h_out)
                Q_cond_wall = self.u_wall * self.area_wall_south * (T_sol_wall - T_indoor)

                T_sol_roof = T_out + (self.alpha_roof * I_horiz_global / self.h_out)
                Q_cond_roof = self.u_roof * self.area_roof * (T_sol_roof - T_indoor)

                Q_cond_glass = self.u_glass * self.area_glass_south * (T_out - T_indoor)
                Q_cond_other = self.u_other * self.area_other * (T_out - T_indoor)
                Q_cond_other_glass = self.u_other_glass * self.area_other_glass * (T_out - T_indoor)
                Q_cond_total = Q_cond_wall + Q_cond_roof + Q_cond_glass + Q_cond_other + Q_cond_other_glass

                Q_int = self.get_internal_gain(h)

                current_ach = self.ach_min 
                if (T_indoor > 23.0) and (T_indoor > T_out): 
                    current_ach = self.ach_max 

                mass_flow = self.volume * (current_ach / 3600.0) * self.air_density
                Q_vent = mass_flow * self.air_cp * (T_out - T_indoor)

                Q_net = Q_sol_win + Q_cond_total + Q_int + Q_vent
                dT = (Q_net / self.c_mass) * dt
                T_indoor += dT

                # 热平衡 (无遮阳)
                Q_sol_win_no = I_win_incident_no * self.area_glass_south * shgc_curr_no

                T_sol_wall_no = T_out + (self.alpha_wall * I_wall_incident_no / self.h_out)
                Q_cond_wall_no = self.u_wall * self.area_wall_south * (T_sol_wall_no - T_indoor_no_shade)

                Q_cond_roof_no = self.u_roof * self.area_roof * (T_sol_roof - T_indoor_no_shade)

                Q_cond_glass_no = self.u_glass * self.area_glass_south * (T_out - T_indoor_no_shade)
                Q_cond_other_no = self.u_other * self.area_other * (T_out - T_indoor_no_shade)
                Q_cond_other_glass_no = self.u_other_glass * self.area_other_glass * (T_out - T_indoor_no_shade)
                Q_cond_total_no = Q_cond_wall_no + Q_cond_roof_no + Q_cond_glass_no + Q_cond_other_no + Q_cond_other_glass_no
                
                current_ach_no = self.ach_min 
                if (T_indoor_no_shade > 23.0) and (T_indoor_no_shade > T_out): 
                    current_ach_no = self.ach_max 
                
                mass_flow_no = self.volume * (current_ach_no / 3600.0) * self.air_density
                Q_vent_no = mass_flow_no * self.air_cp * (T_out - T_indoor_no_shade)
                
                Q_net_no = Q_sol_win_no + Q_cond_total_no + Q_int + Q_vent_no
                dT_no = (Q_net_no / self.c_mass) * dt
                T_indoor_no_shade += dT_no
                
                # 能耗计算
                mass_flow_sealed = self.volume * (self.ach_min / 3600.0) * self.air_density
                ua_system = (self.u_wall*self.area_wall_south + 
                             self.u_glass*self.area_glass_south + 
                             self.u_roof*self.area_roof + 
                             self.u_other*self.area_other + 
                             mass_flow_sealed*self.air_cp) 
                
                occupancy_factor = 1.0 if 8 <= h <= 18 else 0.1
                
                heat_load = 0
                cool_load = 0
                
                if T_indoor > self.temp_max:
                    if T_out > self.temp_max:
                        cool_load = ua_system * (T_indoor - self.temp_max) * (dt/3600.0) / 1000.0 * occupancy_factor
                    else:
                        cool_load = 0 
                
                elif T_indoor < self.temp_min:
                    heat_load = ua_system * (self.temp_min - T_indoor) * (dt/3600.0) / 1000.0 * occupancy_factor
                
                heat_load_no = 0
                cool_load_no = 0
                
                if T_indoor_no_shade > self.temp_max:
                    if T_out > self.temp_max:
                        cool_load_no = ua_system * (T_indoor_no_shade - self.temp_max) * (dt/3600.0) / 1000.0 * occupancy_factor
                    else:
                        cool_load_no = 0 
                elif T_indoor_no_shade < self.temp_min:
                    heat_load_no = ua_system * (self.temp_min - T_indoor_no_shade) * (dt/3600.0) / 1000.0 * occupancy_factor
                
                results['Cooling_Load'].append(cool_load)
                results['Heating_Load'].append(heat_load)
                results['Cooling_Load_no_shade'].append(cool_load_no)
                results['Heating_Load_no_shade'].append(heat_load_no)
                
                results['hour_idx'].append(d*24+h)
                results['T_out'].append(T_out)
                results['T_in'].append(T_indoor)
                results['T_in_no_shade'].append(T_indoor_no_shade)
                results['Q_sol_win'].append(Q_sol_win)
                results['Q_cond'].append(Q_cond_total)
                results['Q_vent'].append(Q_vent)
        
        return pd.DataFrame(results)


def run_phys_optimization():
    print("Running optimization...")
    params = UNIVERSITIES["Sungrove"]
    ratios = np.linspace(0, 1.5, 11)
    summary = []
    
    for r in ratios:
        sim = PhysicallyConsistentBuilding(
            name="Sungrove_Phys",
            lat=params['lat'],
            base_temp=params['base_temp'],
            dh_ratio=r,
            c_mass_factor=1.0, 
            warming_offset=1.5
        )
        df = sim.run_year()
        
        summary.append({
            'Ratio': r,
            'Cooling_kWh': df['Cooling_Load'].sum(),
            'Heating_kWh': df['Heating_Load'].sum(),
            'Total_kWh': df['Cooling_Load'].sum() + df['Heating_Load'].sum()
        })
    return pd.DataFrame(summary)


if __name__ == "__main__":
    df_opt = run_phys_optimization()
    
    best_idx = df_opt['Total_kWh'].idxmin()
    best_r = df_opt.loc[best_idx, 'Ratio']
    print(f"\nOptimal D/H: {best_r:.2f}")
    
    sim_best = PhysicallyConsistentBuilding(
        name="Sungrove_Best",
        lat=UNIVERSITIES["Sungrove"]['lat'],
        base_temp=UNIVERSITIES["Sungrove"]['base_temp'],
        dh_ratio=best_r,
        c_mass_factor=1.0,
        warming_offset=1.5
    )
    df_detail = sim_best.run_year()
    
    # 热平衡验证图
    day_sel = 200
    start = day_sel * 24
    end = start + 24
    day_data = df_detail.iloc[start:end]
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.title(f"Temperature Dynamics (Day {day_sel}) - D/H={best_r:.2f}")
    plt.plot(day_data['hour_idx'], day_data['T_out'], 'k--', label='Outdoor')
    plt.plot(day_data['hour_idx'], day_data['T_in'], 'r-', linewidth=2, label='Indoor')
    plt.axhline(26, color='g', linestyle=':', label='Comfort Limit')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.title("Heat Flow Components")
    plt.plot(day_data['hour_idx'], day_data['Q_sol_win'], label='Solar')
    plt.plot(day_data['hour_idx'], day_data['Q_cond'], label='Conduction')
    plt.plot(day_data['hour_idx'], day_data['Q_vent'], label='Ventilation')
    plt.legend()
    plt.grid(True)
    plt.xlabel("Hour")
    plt.ylabel("Heat Flow (W)")
    
    plt.tight_layout()
    plt.savefig('Phys_Ver.png')
    print("Saved: Phys_Ver.png")
    
    # 夏季温度图
    start_h = 4320
    end_h = start_h + 7 * 24
    sample = df_detail.iloc[start_h:end_h]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(sample['hour_idx'], sample['T_out'], color='gray', linestyle='--', label='Outdoor')
    plt.plot(sample['hour_idx'], sample['T_in'], color='red', linewidth=2.5, label='Indoor (Passive)')
    
    plt.axhline(26, color='green', linestyle=':', linewidth=2, label='26C')
    plt.axhline(18, color='green', linestyle=':', linewidth=2, label='18C')
    
    plt.fill_between(sample['hour_idx'], 26, sample['T_in'], 
                     where=(sample['T_in'] > 26), 
                     color='red', alpha=0.2, label='Cooling Zone')
    
    plt.fill_between(sample['hour_idx'], sample['T_out'], sample['T_in'],
                     where=(sample['Q_vent'] < -100),
                     color='lavender', alpha=1.0, hatch='//', edgecolor='purple', label='Ventilation Cooling')

    plt.title(f'Summer Temperature (D/H={best_r:.2f})', fontsize=12)
    plt.xlabel('Hour of Year')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='-')
    plt.xlim(sample['hour_idx'].min(), sample['hour_idx'].max())
    
    plt.tight_layout()
    plt.savefig('Fig1_Sungrove_Summer_Dynamics.png', dpi=300)
    print("Saved: Fig1_Sungrove_Summer_Dynamics.png")

    # 遮阳对比图
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2,1,1)
    plt.plot(sample['hour_idx'], sample['T_out'], color='gray', linestyle='--', linewidth=1.5, label='Outdoor')
    plt.plot(sample['hour_idx'], sample['T_in'], color='blue', linewidth=2.5, label=f'With Shading (D/H={best_r:.2f})')
    plt.plot(sample['hour_idx'], sample['T_in_no_shade'], color='red', linewidth=2.5, linestyle='-', label='No Shading')
    
    plt.axhline(26, color='green', linestyle=':', linewidth=2, label='26C')
    plt.axhline(18, color='green', linestyle=':', linewidth=2, label='18C')
    
    plt.fill_between(sample['hour_idx'], 26, sample['T_in'], 
                     where=(sample['T_in'] > 26), 
                     color='blue', alpha=0.15, label='Cooling (Shaded)')
    
    plt.fill_between(sample['hour_idx'], 26, sample['T_in_no_shade'], 
                     where=(sample['T_in_no_shade'] > 26), 
                     color='red', alpha=0.15, label='Cooling (No Shade)')
    
    plt.title(f'Shading Comparison - Summer Week', fontsize=13)
    plt.xlabel('Hour of Year')
    plt.ylabel('Temperature (C)')
    plt.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(sample['hour_idx'].min(), sample['hour_idx'].max())
    
    plt.subplot(2,1,2)
    temp_diff = sample['T_in_no_shade'] - sample['T_in']
    
    plt.plot(sample['hour_idx'], temp_diff, color='purple', linewidth=2, label='Temp Diff (No Shade - Shaded)')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.fill_between(sample['hour_idx'], 0, temp_diff, 
                     where=(temp_diff > 0), 
                     color='red', alpha=0.2, label='No Shade Warmer')
    plt.fill_between(sample['hour_idx'], 0, temp_diff, 
                     where=(temp_diff < 0), 
                     color='blue', alpha=0.2, label='Shaded Warmer')
    
    plt.title('Shading Effect', fontsize=13)
    plt.xlabel('Hour of Year')
    plt.ylabel('Temp Diff (C)')
    plt.legend(loc='upper right', framealpha=0.9, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(sample['hour_idx'].min(), sample['hour_idx'].max())
    
    plt.tight_layout()
    plt.savefig('Fig2_Shading_Comparison.png', dpi=300)
    print("Saved: Fig2_Shading_Comparison.png")

    # 能效曲线
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_opt['Ratio'], df_opt['Cooling_kWh'], 
             marker='o', linestyle='-', linewidth=2, color='tab:blue', 
             label='Cooling')
    
    plt.plot(df_opt['Ratio'], df_opt['Heating_kWh'], 
             marker='x', linestyle='--', linewidth=2, color='tab:red', 
             label='Heating')
    
    plt.plot(df_opt['Ratio'], df_opt['Total_kWh'], 
             marker='s', linestyle='-', linewidth=2.5, color='black', 
             label='Total')
    
    plt.xlabel('Overhang Ratio (D/H)', fontsize=11)
    plt.ylabel('Annual Load (kWh)', fontsize=11)
    plt.title('Shading Optimization: Energy Trade-off', fontsize=13)
    
    plt.grid(True, which='major', linestyle='-', alpha=0.4)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.minorticks_on()
    
    plt.legend(loc='best', fontsize=10, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('Fig3_Shading_Energy_off.png', dpi=300)
    print("Saved: Fig3_Shading_Energy_off.png")
    
    # 年能耗对比
    total_cooling_no_shade = df_detail['Cooling_Load_no_shade'].sum()
    total_heating_no_shade = df_detail['Heating_Load_no_shade'].sum()
    total_energy_no_shade = total_cooling_no_shade + total_heating_no_shade
    
    total_cooling_with_shade = df_detail['Cooling_Load'].sum()
    total_heating_with_shade = df_detail['Heating_Load'].sum()
    total_energy_with_shade = total_cooling_with_shade + total_heating_with_shade
    
    plt.figure(figsize=(12, 6))
    
    categories = ['Cooling', 'Heating', 'Total']
    with_shade = [total_cooling_with_shade, total_heating_with_shade, total_energy_with_shade]
    no_shade = [total_cooling_no_shade, total_heating_no_shade, total_energy_no_shade]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, with_shade, width, label=f'With Shading (D/H={best_r:.2f})', color='blue', alpha=0.7)
    bars2 = plt.bar(x + width/2, no_shade, width, label='No Shading', color='red', alpha=0.7)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Energy Type', fontsize=12)
    plt.ylabel('Annual Energy (kWh)', fontsize=12)
    plt.title('Annual Energy: Shading vs No Shading', fontsize=14)
    plt.xticks(x, categories)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    energy_saving = (total_energy_no_shade - total_energy_with_shade) / total_energy_no_shade * 100
    cooling_saving = (total_cooling_no_shade - total_cooling_with_shade) / total_cooling_no_shade * 100 if total_cooling_no_shade > 0 else 0
    
    textstr = f'Total Saving: {energy_saving:.1f}%\nCooling Saving: {cooling_saving:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('Fig4_Shading_Comparison_Energy.png', dpi=300)
    print("Saved: Fig4_Shading_Comparison_Energy.png")
    print(f"\nResults:")
    print(f"No Shading: {total_energy_no_shade:.2f} kWh")
    print(f"With Shading: {total_energy_with_shade:.2f} kWh")
    print(f"Saving: {energy_saving:.2f}%")
    
    plt.show()