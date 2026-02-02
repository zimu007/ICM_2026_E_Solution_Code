import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

plt.rcParams['axes.unicode_minus'] = False

EARTH_TILT_DEG = 23.44
DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24

UNIVERSITIES = {
    "Sungrove": {
        "lat": 25.0,
        "base_temp": 24.0,
        "seasonal_amp": 6.0,
        "daily_swing": 12.0,
        "solar_sensitivity": 15.0,
        "climate_desc": "Warm subtropical/tropical"
    },
    "Borealis": {
        "lat": 65.0,
        "base_temp": 1.0,
        "seasonal_amp": 16.0,
        "daily_swing": 8.0,
        "solar_sensitivity": 8.0,
        "climate_desc": "Subarctic/polar"
    }
}


def get_solar_declination(day):
    day_angle = 2 * np.pi * (day + 10) / 365.25
    declination = np.deg2rad(EARTH_TILT_DEG) * np.sin(2 * np.pi * (284 + day) / 365.25)
    return declination

def get_solar_geometry(day, hour, lat_deg):
    declination = get_solar_declination(day)
    hour_angle = (hour - 12.0) * (np.pi / 12.0)
    lat_rad = np.deg2rad(lat_deg)
    
    sin_alt = (np.sin(lat_rad) * np.sin(declination) + 
               np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    return altitude

def calculate_solar_irradiance(altitude, day):
    if altitude <= 0:
        return 0.0
    
    solar_constant = 1367.0
    
    day_angle = 2 * np.pi * day / 365.25
    distance_factor = 1.0 + 0.033 * np.cos(day_angle)
    
    zenith = np.pi/2 - altitude
    zenith_deg = np.rad2deg(zenith)
    
    if zenith_deg < 90:
        air_mass = 1.0 / (np.cos(zenith) + 0.50572 * (96.07995 - zenith_deg)**(-1.6364))
    else:
        air_mass = 38.0
    
    transmittance = 0.75 ** (air_mass ** 0.678)
    
    direct_normal = solar_constant * distance_factor * transmittance
    direct_horizontal = direct_normal * np.sin(altitude)
    diffuse_horizontal = solar_constant * distance_factor * (1 - transmittance) * 0.3
    global_horizontal = direct_horizontal + diffuse_horizontal
    
    return global_horizontal

def calculate_temperature_matrix(univ_params):
    matrix = np.zeros((DAYS_IN_YEAR, HOURS_IN_DAY))
    lat = univ_params['lat']
    
    days = np.arange(DAYS_IN_YEAR).reshape(-1, 1)
    hours = np.arange(HOURS_IN_DAY).reshape(1, -1)
    
    # 季节基温
    thermal_lag = 30
    season_phase = 2 * np.pi * (days - thermal_lag) / 365.0
    t_seasonal = univ_params['base_temp'] + univ_params['seasonal_amp'] * (-np.cos(season_phase))
    
    # 昼夜波动
    peak_hour = 14
    day_phase = 2 * np.pi * (hours - peak_hour) / 24.0
    t_daily = univ_params['daily_swing'] * 0.5 * np.cos(day_phase)
    
    # 太阳辐射
    declination = get_solar_declination(days)
    hour_angle = (hours - 12.0) * (np.pi / 12.0)
    lat_rad = np.deg2rad(lat)
    
    sin_alt = (np.sin(lat_rad) * np.sin(declination) + 
               np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
    
    altitude = np.arcsin(np.clip(sin_alt, -1, 1))
    
    irradiance = np.zeros_like(altitude)
    mask = altitude > 0
    
    if np.any(mask):
        for i in range(DAYS_IN_YEAR):
            for j in range(HOURS_IN_DAY):
                if altitude[i, j] > 0:
                    irradiance[i, j] = calculate_solar_irradiance(altitude[i, j], i)
    
    solar_heating_factor = univ_params['solar_sensitivity'] / 1000.0
    t_solar = irradiance * solar_heating_factor
    
    # 高纬修正
    if lat > 60:
        summer_mask = (days >= 140) & (days <= 200)
        night_mask = (hours < 4) | (hours > 20)
        t_daily = np.where(summer_mask & night_mask, 
                          t_daily * 0.5,
                          t_daily)
    
    matrix = t_seasonal + t_daily + t_solar
    matrix = np.clip(matrix, -60, 50)
    
    return matrix


def validate_temperature_data(data, name, params):
    print(f"\n{'='*60}")
    print(f"Validation: {name}")
    print(f"{'='*60}")
    print(f"Lat: {params['lat']}°")
    print(f"Climate: {params['climate_desc']}")
    print(f"\nAnnual:")
    print(f"  Min: {data.min():.1f}°C")
    print(f"  Max: {data.max():.1f}°C")
    print(f"  Avg: {data.mean():.1f}°C")
    print(f"  Range: {data.max() - data.min():.1f}°C")
    
    winter_days = np.concatenate([np.arange(0, 80), np.arange(355, 365)])
    summer_days = np.arange(172, 264)
    
    print(f"\nSeason:")
    print(f"  Winter avg: {data[winter_days, :].mean():.1f}°C")
    print(f"  Summer avg: {data[summer_days, :].mean():.1f}°C")
    print(f"  Diff: {data[summer_days, :].mean() - data[winter_days, :].mean():.1f}°C")
    
    day_hours = np.arange(8, 18)
    night_hours = np.concatenate([np.arange(0, 6), np.arange(20, 24)])
    
    print(f"\nDaily:")
    print(f"  Day avg: {data[:, day_hours].mean():.1f}°C")
    print(f"  Night avg: {data[:, night_hours].mean():.1f}°C")
    print(f"  Diff: {data[:, day_hours].mean() - data[:, night_hours].mean():.1f}°C")
    
    idx_max = np.unravel_index(data.argmax(), data.shape)
    idx_min = np.unravel_index(data.argmin(), data.shape)
    
    print(f"\nExtremes:")
    print(f"  Max: Day {idx_max[0]}, Hour {idx_max[1]}")
    print(f"  Min: Day {idx_min[0]}, Hour {idx_min[1]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Solar Engine")
    print("="*60)
    
    print("\n[Step 1] Generating temperature maps...")
    
    data_sungrove = calculate_temperature_matrix(UNIVERSITIES['Sungrove'])
    data_borealis = calculate_temperature_matrix(UNIVERSITIES['Borealis'])
    
    validate_temperature_data(data_sungrove, "Sungrove University", 
                            UNIVERSITIES['Sungrove'])
    validate_temperature_data(data_borealis, "Borealis University", 
                            UNIVERSITIES['Borealis'])
    
    colors_cold = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
    colors_warm = ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
    all_colors = colors_cold + ['#ffffff'] + colors_warm
    n_bins = 256  # 增加颜色分辨率
    cmap = LinearSegmentedColormap.from_list('temperature', all_colors, N=n_bins)
    
    # 使用插值平滑数据 (4x放大)
    zoom_factor = 4
    data_sungrove_smooth = zoom(data_sungrove.T, zoom_factor, order=3)
    data_borealis_smooth = zoom(data_borealis.T, zoom_factor, order=3)
    
    fig_tables, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Sungrove
    im1 = ax1.imshow(data_sungrove_smooth, aspect='auto', cmap=cmap, origin='lower',
                     extent=[0, 365, 0, 24], vmin=-5, vmax=45, interpolation='bilinear')
    ax1.set_title(f"Sungrove University: Annual Temperature Pattern\n"
                  f"Latitude {UNIVERSITIES['Sungrove']['lat']}°N - {UNIVERSITIES['Sungrove']['climate_desc']}", 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("Hour of Day", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Day of Year", fontsize=12, fontweight='bold')
    
    season_marks = [
        (0, 'Jan'), (31, 'Feb'), (59, 'Mar'), (90, 'Apr'),
        (120, 'May'), (151, 'Jun'), (181, 'Jul'), (212, 'Aug'),
        (243, 'Sep'), (273, 'Oct'), (304, 'Nov'), (334, 'Dec')
    ]
    ax1.set_xticks([m[0] for m in season_marks])
    ax1.set_xticklabels([m[1] for m in season_marks], fontsize=10)
    
    ax1.set_yticks([0, 6, 12, 18, 24])
    ax1.set_yticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'], fontsize=10)
    
    # 添加网格线
    for m in season_marks:
        ax1.axvline(x=m[0], color='white', linestyle='-', linewidth=0.3, alpha=0.5)
    for h in [6, 12, 18]:
        ax1.axhline(y=h, color='white', linestyle='-', linewidth=0.3, alpha=0.5)
    
    ax1.axvline(x=172, color='gold', linestyle='--', linewidth=2, 
                alpha=0.7, label='Summer Solstice')
    ax1.axvline(x=355, color='cyan', linestyle='--', linewidth=2, 
                alpha=0.7, label='Winter Solstice')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.8)
    
    cbar1 = fig_tables.colorbar(im1, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label('Temperature (°C)', fontsize=11, fontweight='bold')
    
    # Borealis
    im2 = ax2.imshow(data_borealis_smooth, aspect='auto', cmap=cmap, origin='lower',
                     extent=[0, 365, 0, 24], vmin=-35, vmax=25, interpolation='bilinear')
    ax2.set_title(f"Borealis University: Annual Temperature Pattern\n"
                  f"Latitude {UNIVERSITIES['Borealis']['lat']}°N - {UNIVERSITIES['Borealis']['climate_desc']}", 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel("Hour of Day", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Day of Year", fontsize=12, fontweight='bold')
    
    ax2.set_xticks([m[0] for m in season_marks])
    ax2.set_xticklabels([m[1] for m in season_marks], fontsize=10)
    ax2.set_yticks([0, 6, 12, 18, 24])
    ax2.set_yticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'], fontsize=10)
    
    # 添加网格线
    for m in season_marks:
        ax2.axvline(x=m[0], color='white', linestyle='-', linewidth=0.3, alpha=0.5)
    for h in [6, 12, 18]:
        ax2.axhline(y=h, color='white', linestyle='-', linewidth=0.3, alpha=0.5)
    
    ax2.axvline(x=172, color='gold', linestyle='--', linewidth=2, 
                alpha=0.7, label='Summer Solstice (Midnight Sun)')
    ax2.axvline(x=355, color='cyan', linestyle='--', linewidth=2, 
                alpha=0.7, label='Winter Solstice (Polar Night)')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.8)
    
    cbar2 = fig_tables.colorbar(im2, ax=ax2, orientation='vertical', pad=0.02)
    cbar2.set_label('Temperature (°C)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("Fig1_Tables.png", dpi=300, bbox_inches='tight')
    print("\n  Saved: Fig1_Tables.png")
    
    print("\n[Step 2] Generating comparison...")
    
    fig_compare, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    days = np.arange(365)
    
    daily_avg_sun = data_sungrove.mean(axis=1)
    daily_avg_bor = data_borealis.mean(axis=1)
    
    daily_max_sun = data_sungrove.max(axis=1)
    daily_min_sun = data_sungrove.min(axis=1)
    daily_max_bor = data_borealis.max(axis=1)
    daily_min_bor = data_borealis.min(axis=1)
    
    ax.plot(days, daily_avg_sun, 'r-', linewidth=2.5, label='Sungrove (Avg)')
    ax.fill_between(days, daily_min_sun, daily_max_sun, color='red', alpha=0.2)
    
    ax.plot(days, daily_avg_bor, 'b-', linewidth=2.5, label='Borealis (Avg)')
    ax.fill_between(days, daily_min_bor, daily_max_bor, color='blue', alpha=0.2)
    
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Day of Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Annual Temperature Variation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    hours = np.arange(24)
    
    summer_day = 180
    ax.plot(hours, data_sungrove[summer_day, :], 'r-', linewidth=2.5, 
            marker='o', markersize=4, label='Sungrove Summer')
    ax.plot(hours, data_borealis[summer_day, :], 'b-', linewidth=2.5,
            marker='o', markersize=4, label='Borealis Summer')
    
    winter_day = 15
    ax.plot(hours, data_sungrove[winter_day, :], 'r--', linewidth=2,
            marker='s', markersize=4, alpha=0.7, label='Sungrove Winter')
    ax.plot(hours, data_borealis[winter_day, :], 'b--', linewidth=2,
            marker='s', markersize=4, alpha=0.7, label='Borealis Winter')
    
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Typical Day Temperature Profiles', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 6, 12, 18, 24])
    
    ax = axes[1, 0]
    
    sun_irrad_summer = []
    bor_irrad_summer = []
    
    for h in range(24):
        alt_sun = get_solar_geometry(summer_day, h, UNIVERSITIES['Sungrove']['lat'])
        alt_bor = get_solar_geometry(summer_day, h, UNIVERSITIES['Borealis']['lat'])
        
        sun_irrad_summer.append(calculate_solar_irradiance(alt_sun, summer_day))
        bor_irrad_summer.append(calculate_solar_irradiance(alt_bor, summer_day))
    
    ax.plot(hours, sun_irrad_summer, 'r-', linewidth=2.5, marker='o',
            markersize=4, label=f'Sungrove ({UNIVERSITIES["Sungrove"]["lat"]}°N)')
    ax.plot(hours, bor_irrad_summer, 'b-', linewidth=2.5, marker='o',
            markersize=4, label=f'Borealis ({UNIVERSITIES["Borealis"]["lat"]}°N)')
    
    ax.fill_between(hours, 0, sun_irrad_summer, color='red', alpha=0.1)
    ax.fill_between(hours, 0, bor_irrad_summer, color='blue', alpha=0.1)
    
    ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax.set_ylabel('Solar Irradiance (W/m²)', fontsize=11, fontweight='bold')
    ax.set_title(f'Summer Solstice Solar Irradiance (Day {summer_day})', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 6, 12, 18, 24])
    
    ax = axes[1, 1]
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    
    monthly_avg_sun = []
    monthly_avg_bor = []
    
    for i in range(12):
        start = month_starts[i]
        end = month_starts[i+1]
        monthly_avg_sun.append(data_sungrove[start:end, :].mean())
        monthly_avg_bor.append(data_borealis[start:end, :].mean())
    
    x_pos = np.arange(12)
    width = 0.35
    
    ax.bar(x_pos - width/2, monthly_avg_sun, width, label='Sungrove',
           color='coral', edgecolor='black', linewidth=1)
    ax.bar(x_pos + width/2, monthly_avg_bor, width, label='Borealis',
           color='skyblue', edgecolor='black', linewidth=1)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Temperature (°C)', fontsize=11, fontweight='bold')
    ax.set_title('Monthly Average Temperatures', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(months, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("Fig2_Compare.png", dpi=300, bbox_inches='tight')
    print("  Saved: Fig2_Compare.png")
    
    print("\n[Step 3] Preparing 3D...")
    
    ORBIT_RADIUS = 6.0
    EARTH_RADIUS = 1.0
    ANIMATION_STEP = 3
    
    fig_3d = plt.figure(figsize=(14, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.set_facecolor('#000814')
    ax_3d.grid(False)
    
    ax_3d.scatter([0], [0], [0], color='#FDB813', s=2000, 
                  edgecolors='#FF6B35', linewidths=2, label='Sun')
    
    theta = np.linspace(0, 2*np.pi, 100)
    ax_3d.plot(ORBIT_RADIUS*np.cos(theta), ORBIT_RADIUS*np.sin(theta), 0, 
              color='gray', linestyle=':', alpha=0.4, linewidth=1)
    
    earth_surf = None
    dot_sungrove, = ax_3d.plot([], [], [], 'o', markersize=14, 
                              markeredgecolor='white', markeredgewidth=2,
                              label='Sungrove')
    dot_borealis, = ax_3d.plot([], [], [], 'o', markersize=14,
                              markeredgecolor='white', markeredgewidth=2,
                              label='Borealis')
    
    txt_info = ax_3d.text2D(0.02, 0.97, "", transform=ax_3d.transAxes,
                           color='white', fontsize=11, family='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', 
                                   alpha=0.7, edgecolor='white'))
    
    def get_location_pos(lat_deg, center, earth_r, day_of_year):
        lat_rad = np.deg2rad(lat_deg)
        cx, cy, cz = center
        
        vec_to_sun = np.array([-cx, -cy, 0])
        norm = np.linalg.norm(vec_to_sun)
        if norm > 0:
            vec_to_sun = vec_to_sun / norm
        else:
            vec_to_sun = np.array([1, 0, 0])
        
        tilt_rad = np.deg2rad(EARTH_TILT_DEG)
        orbit_angle = 2 * np.pi * day_of_year / 365.0
        
        axis_y = np.sin(tilt_rad) * np.cos(orbit_angle)
        axis_z = np.cos(tilt_rad)
        axis_up = np.array([0, axis_y, axis_z])
        axis_up = axis_up / np.linalg.norm(axis_up)
        
        vec_perp = vec_to_sun - np.dot(vec_to_sun, axis_up) * axis_up
        norm_perp = np.linalg.norm(vec_perp)
        if norm_perp > 0:
            vec_perp = vec_perp / norm_perp
        
        pos_vec = (np.sin(lat_rad) * axis_up + np.cos(lat_rad) * vec_perp) * earth_r
        
        return np.array([cx, cy, cz]) + pos_vec
    
    def update_3d(frame):
        global earth_surf
        day = frame
        
        angle = 2 * np.pi * day / 365.0
        ex = ORBIT_RADIUS * np.cos(angle)
        ey = ORBIT_RADIUS * np.sin(angle)
        ez = 0
        
        if earth_surf:
            earth_surf.remove()
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:12j]
        x = EARTH_RADIUS * np.cos(u) * np.sin(v) + ex
        y = EARTH_RADIUS * np.sin(u) * np.sin(v) + ey
        z = EARTH_RADIUS * np.cos(v) + ez
        earth_surf = ax_3d.plot_wireframe(x, y, z, color='#00B4D8', 
                                         alpha=0.2, linewidth=0.8)
        
        norm = plt.Normalize(-10, 40)
        
        temp_s = data_sungrove[day, 12]
        pos_s = get_location_pos(UNIVERSITIES['Sungrove']['lat'], 
                                [ex, ey, ez], EARTH_RADIUS, day)
        dot_sungrove.set_data([pos_s[0]], [pos_s[1]])
        dot_sungrove.set_3d_properties([pos_s[2]])
        dot_sungrove.set_color(cmap(norm(temp_s)))
        
        temp_b = data_borealis[day, 12]
        pos_b = get_location_pos(UNIVERSITIES['Borealis']['lat'],
                                [ex, ey, ez], EARTH_RADIUS, day)
        dot_borealis.set_data([pos_b[0]], [pos_b[1]])
        dot_borealis.set_3d_properties([pos_b[2]])
        dot_borealis.set_color(cmap(norm(temp_b)))
        
        season_name = ""
        if 80 < day < 172: season_name = "Spring"
        elif 172 <= day < 264: season_name = "Summer"
        elif 264 <= day < 355: season_name = "Autumn"
        else: season_name = "Winter"
        
        info = (f"Day: {day:03d}/365 ({season_name})\n"
               f"───────────────────────\n"
               f"Sungrove (Noon):\n"
               f"  {temp_s:+.1f}°C\n"
               f"Borealis (Noon):\n"
               f"  {temp_b:+.1f}°C")
        txt_info.set_text(info)
        
        return earth_surf, dot_sungrove, dot_borealis, txt_info
    
    ax_3d.set_xlim(-ORBIT_RADIUS-2, ORBIT_RADIUS+2)
    ax_3d.set_ylim(-ORBIT_RADIUS-2, ORBIT_RADIUS+2)
    ax_3d.set_zlim(-3, 3)
    ax_3d.view_init(elev=25, azim=-70)
    ax_3d.set_xlabel('X', fontsize=10, color='white')
    ax_3d.set_ylabel('Y', fontsize=10, color='white')
    ax_3d.set_zlabel('Z', fontsize=10, color='white')
    ax_3d.legend(loc='upper left', fontsize=10, framealpha=0.8)
    
    ani = animation.FuncAnimation(fig_3d, update_3d, 
                                 frames=range(0, 365, ANIMATION_STEP),
                                 interval=40, blit=False, repeat=True)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print("\nFiles:")
    print("  1. Improved_Thermal_Tables.png")
    print("  2. Improved_Comparison_Analysis.png")
    print("  3. 3D Animation (plt.show())")
    print("="*60)
    
    plt.show()