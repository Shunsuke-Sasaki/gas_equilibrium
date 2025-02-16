import numpy as np
import matplotlib.pyplot as plt

# 定数の定義
R = 8.314 * 10**-5  # kJ/(mol*K)
P0 = 1e5  # Pa
hp = 6.62607015e-34  # J*s (プランク定数)
k_B = 1.380649e-23  # J/K (ボルツマン定数)
c = 2.998e+8  # m/s

# COFとCOF2の物理定数 (単位変換も含む)
mass_COF = 1.42969e-25  # COFの質量 (kg)
rotational_constants_COF = [144.15084869,  12.1339279,  11.1918515]  # g·cm² -> kg·m²
nist_moment_COF = 2.199751e-116
frequencies_COF = [631.25, 1028.16, 1927.96]  # 振動周波数 (cm⁻¹)
nist_frequencies_COF =[626,1018,1855]
symmetry_COF = 1  # COFの対称性数

mass_COF2 = 1.658e-25  # COF2の質量 (kg)
rotational_constants_COF2 = [22.2132154,  8.8175810,  6.6956056]  # g·cm² -> kg·m²
nist_moment_COF2 = 7.2256671e-115
frequencies_COF2_low = [578.99, 619.55, 775.47]  # 低周波振動周波数 (cm⁻¹)
frequencies_COF2_high = [964.09, 1229.25, 1992.11]  # 高周波振動周波数 (cm⁻¹)
nist_frequencies_COF2 = [584, 626, 774,965,1249,1928]  
symmetry_COF2 = 1  # COF2の対称性数

# 並進分配関数の計算
def calculate_translational_partition_function(mass, T):
    return (k_B * T / P0) * (2 * np.pi * mass * k_B * T / hp**2)**(3/2)

# 回転分配関数の計算
def calculate_rotational_partition_function(rotational_constants, T, symmetry_number):
    theta_A, theta_B, theta_C = rotational_constants
    #print((hp/(8*np.pi**2*c))**3/(theta_A*theta_B*theta_C))
    Z_r = (np.pi / (theta_A * theta_B * theta_C*10**27))**(1/2) * ((T * k_B) / (hp * c))**(3/2) / symmetry_number
    return Z_r

def calculate_rotational_partition_function2(moment, T, symmetry_number):
    Z_r = (np.pi **7 * moment*10**-120* (8 * k_B * T / hp**3 /c)**3)**(1/2) / symmetry_number
    return Z_r

# 振動分配関数の計算
def calculate_vibrational_partition_function(frequencies, T):
    Z_v = 1.0
    for k in frequencies:
        exponent = hp * c * k * 100 / (k_B * T)  # 周波数がcm⁻¹なので100を掛ける
        Z_v *= (1 - np.exp(-exponent))**-1
    return Z_v

# 温度範囲の設定
T_min = 300  # 最小温度
T_max = 30000  # 最大温度
num_points = 100  # 温度の点の数
temperature_range = np.linspace(T_min, T_max, num_points)

# COFの内部状態和を計算
Z_int_COF = []
for T in temperature_range:
    Z_tr = calculate_translational_partition_function(mass_COF, T)
    Z_r = calculate_rotational_partition_function(rotational_constants_COF, T, symmetry_COF)
    Z_v = calculate_vibrational_partition_function(frequencies_COF, T)
    Z_int = Z_tr * Z_r * Z_v
    Z_int_COF.append(Z_int)

# COF2の内部状態和を計算 (低周波と高周波を考慮)
Z_int_COF2 = []
frequencies_COF2 = frequencies_COF2_low + frequencies_COF2_high
for T in temperature_range:
    Z_tr = calculate_translational_partition_function(mass_COF2, T)
    Z_r = calculate_rotational_partition_function(rotational_constants_COF2, T, symmetry_COF2)
    Z_v = calculate_vibrational_partition_function(frequencies_COF2, T)
    Z_int = Z_tr * Z_r * Z_v
    Z_int_COF2.append(Z_int)

# 仮のNIST文献値（実際の値に置き換えてください）
nist_Z_int_COF = []  # COFの文献値
for T in temperature_range:
    Z_tr = calculate_translational_partition_function(mass_COF, T)
    Z_r = calculate_rotational_partition_function2(nist_moment_COF, T, symmetry_COF)
    Z_v = calculate_vibrational_partition_function(nist_frequencies_COF, T)
    Z_int = Z_tr * Z_r * Z_v
    nist_Z_int_COF.append(Z_int)
    
nist_Z_int_COF2 = []  # COF2の文献値
for T in temperature_range:
    Z_tr = calculate_translational_partition_function(mass_COF2, T)
    Z_r = calculate_rotational_partition_function2(nist_moment_COF2, T, symmetry_COF2)
    Z_v = calculate_vibrational_partition_function(frequencies_COF2, T)
    Z_int = Z_tr * Z_r * Z_v
    nist_Z_int_COF2.append(Z_int)

# 結果のプロット
plt.plot(temperature_range, Z_int_COF,'b', label="COF Internal Partition Function")
plt.plot(temperature_range, Z_int_COF2,'r', label="COF2 Internal Partition Function")
plt.plot(temperature_range, nist_Z_int_COF, 'b--', label="NIST-JANAF COF", linewidth=1.5)
plt.plot(temperature_range, nist_Z_int_COF2, 'r--', label="NIST-JANAF COF2", linewidth=1.5)

# グラフの設定
plt.yscale('log')
plt.xlabel('Temperature (K)')
plt.ylabel('Internal Partition Function')
plt.title('Internal Partition Function of COF and COF2 vs Temperature')
plt.grid(True)
plt.legend()

# グラフの表示
plt.show()