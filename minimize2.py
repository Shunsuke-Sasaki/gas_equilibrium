import numpy as np
from scipy.optimize import minimize
import re
import matplotlib.pyplot as plt
from cycler import cycler
import mplcursors  

def print_start():
    red_color = "\033[91m"  # 赤色のANSIエスケープコード
    reset_color = "\033[0m"  # 色リセットのANSIエスケープコード
    border = """
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
----------------------------------------------------------------------------
    """
    start_art = red_color + border + reset_color  # 80個の「ー」文字を赤色で連結して表示
    print(start_art)

print_start()

# 定数の定義
R = 8.314*10**-5
P0 = 1e5  # Pa
hp = 6.62607015e-34  # J*s (プランク定数)
k_B = 1.380649e-23  # J/K (ボルツマン定数)
c = 2.998*10**8 

def parse_particle_data(file_path):
    particles_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Skip empty lines
        if line.strip():
            # Split line into components
            components = line.split(',')
            if len(components) >= 5:
                try:
                    particle = components[0].strip()
                    initial_amount = float(components[1].strip())
                    mass = float(components[2].strip())

                    # Parse rotational constants if available
                    rotational_constants = []
                    if components[3].strip():
                        rotational_constants = [float(x.strip()) for x in components[3].split()]

                    # Parse frequencies if available
                    frequencies = []
                    if components[4].strip():
                        frequencies = [float(x.strip()) for x in components[4].split()]
                        
                    if components[5].strip():
                        symmetry = float(components[5].strip())
                    
                    if components[6].strip():
                        enthalpy = float(components[6].strip())

                    # Append to particles data
                    if symmetry is not None:
                        particles_data.append({
                            'name': particle,
                            'initial_amount': initial_amount,
                            'm': mass,
                            'rotational_constants': rotational_constants,
                            'frequencies': frequencies,
                            'symmetry': symmetry,
                            'H' : enthalpy
                        })
                except ValueError as e:
                    print(f"Error processing line: {line.strip()}")
                    print(f"Error details: {e}")

    return particles_data

# Example usage
file_path = 'minimize_inputs.txt'
particles_data = parse_particle_data(file_path)

# 各粒子の名前から組成を生成する関数
def parse_composition(name):
    composition = {'C': 0, 'F': 0, 'O': 0, 'e': 0}
    matches = re.findall(r'([A-Z][a-z]?|[-+e])(\d*)', name)
    for (element, count) in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        if element == '-':
            element = 'e'
        elif element == '+':
            element = 'e'
            count = -count
        composition[element] += count
    return [composition['C'], composition['F'], composition['O'], composition['e']]

# 各粒子の組成を生成し、リストを更新
for particle in particles_data:
    particle['composition'] = parse_composition(particle['name'])

# 各粒子が含む原子の数を行列Aとして定義[C, F, O, e]
A = np.array([p['composition'] for p in particles_data])

# 基準化学ポテンシャルと初期物質量のリストを作成
initial_amounts = np.array([p['initial_amount'] for p in particles_data])
m = np.array([p['m'] for p in particles_data])
n_sum = A.T @ initial_amounts

# 回転分配関数の計算
def calculate_rotational_partition_function(rotational_constants, T, symmetry_number):
    if rotational_constants[0] < 10**-3:
        B = (rotational_constants[1] + rotational_constants[2]) / 2
        Z_r = k_B * T / (hp * symmetry_number * B)
    else:
        theta_A, theta_B, theta_C = rotational_constants
        Z_r = (np.pi / (theta_A * theta_B * theta_C * 10**27))**(1/2) * ((T * k_B) / (hp * c))**(3/2) / symmetry_number
    return Z_r

# 振動分配関数の計算
def calculate_vibrational_partition_function(frequencies, T):
    Z_v = 1.0
    for k in frequencies:
        exponent1 = hp * 100 * c * k / (k_B * T)
        Z_v *= 1 / (1 - np.exp(-exponent1))
    return Z_v

# 並進分配関数の計算
def calculate_translational_partition_function(mass, T):
    return (k_B * T / P0) * (2 * np.pi * mass * k_B * T / hp**2)**(3/2)

# ギブスの自由エネルギーの目的関数の定義
def gibbs_free_energy(n, T, mu0):
    total_n = np.sum(n)
    x = n / total_n  # モル分率
    x = np.clip(x, 1e-10, None)  # 0以下の値を避けるためのクリッピング
    mu = mu0 + R * T * np.log(x)
    return np.dot(mu, n)

constraints = [{"type": "eq", "fun": lambda n, j=j: np.dot(A[:, j], n) - n_sum[j]} for j in range(A.shape[1])]
bounds = [(0, None) for _ in particles_data]

# 初期推定値の設定（各粒子の初期物質量）
n = initial_amounts

# temperature_rangeをlogスケールで生成
T_min = 1*10**2  # 最小温度
T_max = 1 * 10**3  # 最大温度
num_points = 100  # 温度の点の数
# logspaceを使ってlogスケールで温度範囲を生成
temperature_range = np.logspace(np.log10(T_min), np.log10(T_max), num_points)

species_num = []

for T in temperature_range:
    T = np.clip(T, 1e-10, None)

    mu0_list = []
    for particle in particles_data:
        mass = particle['m']
        rotational_constants = particle['rotational_constants']
        frequencies = particle['frequencies']
        symmetry = particle['symmetry']
        H = particle['H']
        composition = particle['composition']
        is_monoatomic = 0
        if sum(composition[:3]) == 1:
            is_monoatomic = 1
        elif composition[3] != 0 and sum(composition[:3]) == 0:
            is_monoatomic = 1
        Z_tr = calculate_translational_partition_function(mass, T)
        if is_monoatomic:
            Z_v = 1
            Z_r = 1
        else:
            Z_v = calculate_vibrational_partition_function(frequencies, T)
            Z_r = calculate_rotational_partition_function(rotational_constants, T, symmetry)
        Z = Z_tr * Z_v * Z_r
        mu0 = -R * T * np.log(Z) + H * 10**-3
        mu0_list.append(mu0)
    mu0 = np.array(mu0_list)

    # 最適化の実行
    result = minimize(gibbs_free_energy, n, method='SLSQP', args=(T, mu0), bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    
    # 粒子数が 10^-6 を下回った場合は 0 にする
    optimized_amounts = result.x
    optimized_amounts[optimized_amounts < 1e-8] = 0  # 10^-6未満の粒子数を0に
    species_num.append(optimized_amounts)

    # 結果の表示
    print("T=", T)
    print(result)
    print('Optimal amounts of substances:', result.x)
    print()

colors = plt.cm.tab20(np.linspace(0, 1, len(particles_data)))
linestyles = ['-', '--'] * (len(particles_data) // 2)
if len(linestyles) < len(colors):
    linestyles = (linestyles * ((len(colors) // len(linestyles)) + 1))[:len(colors)]
plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles)))

species_num = np.array(species_num)
lines = []
for i, particle in enumerate(particles_data):
    # モル分率を計算
    total_species_num = np.sum(species_num, axis=1)
    mole_fraction = species_num[:, i] / total_species_num  # モル分率の計算
    line, = plt.plot(temperature_range[:len(mole_fraction)], mole_fraction, label=particle['name'])
    lines.append(line)

# グラフの設定
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-7, 1)  # y軸の範囲を設定（モル分率は0〜1の範囲）
plt.xlabel('Temperature (K)', fontsize=16)  # フォントサイズを14に設定
plt.ylabel('Mole Fraction', fontsize=16)    # フォントサイズを14に設定
plt.title('Mole Fractions vs Temperature', fontsize=16)  # タイトルのフォントサイズを16に設定
plt.legend(fontsize=12)  # 凡例のフォントサイズを12に設定
plt.grid(True)

plt.tick_params(axis='both', which='both', labelsize=16)
# mplcursors を使ってホバー機能を追加
cursor = mplcursors.cursor(lines, hover=True)

# ホバー時に組成を表示する関数
@cursor.connect("add")
def on_add(sel):
    index = lines.index(sel.artist)  # ホバーされたラインのインデックスを取得
    particle = particles_data[index]  # 対応する粒子のデータを取得
    # 組成をツールチップに表示
    sel.annotation.set(text=f"{particle['name']}")

# グラフの表示
plt.show()