import numpy as np
from scipy.optimize import minimize
import re
import matplotlib.pyplot as plt
from cycler import cycler
import re

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
    """
    start_art = red_color + border + reset_color  # 80個の「ー」文字を赤色で連結して表示
    print(start_art)
print_start()


# 定数の定義
R = 8.314*10**-3  # J/(mol*K)
P0 = 1e5  # Pa
hp = 6.62607015e-34  # J*s (プランク定数)
k = 1.380649e-23  # J/K (ボルツマン定数)

# 粒子の種類と基準化学ポテンシャル、初期物質量を定義
def parse_particle_data(file_path):
    particles_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_particle = None
    for line in lines:
        if line.strip():
            if not current_particle:
                # Extract particle name
                current_particle = {'name': line.strip(), 'frequencies': [], 'rotational_constants': [], 'enthalpy': None}
            else:
                if 'Frequencies' in line:
                    freqs = re.findall(r'\d+\.\d+', line)
                    current_particle['frequencies'].extend(freqs)
                elif 'Rotational Constants' in line:
                    constants = re.findall(r'\d+\.\d+', line)
                    current_particle['rotational_constants'].extend(constants)
                elif 'Enthalpy' in line:
                    enthalpy = re.findall(r'-?\d+\.\d+', line)
                    if enthalpy:
                        current_particle['enthalpy'] = float(enthalpy[0])

                # Check if we have completed parsing one particle
                if current_particle['frequencies'] and current_particle['rotational_constants'] and current_particle['enthalpy'] is not None:
                    particles_data.append(current_particle)
                    current_particle = None

    return particles_data

# Example usage
file_path = 'extracted_data.txt'
particles_data2 = parse_particle_data(file_path)

#print(particles_data)

particles_data = [
    {'name': 'C5F10O', 'm': 10**-33,   'Zint': 9*10**6, 'Hf': -1*10**1, 'initial_amount': 1},
    {'name': 'CO2',    'm': 10**-33,   'Zint': 8*10**5, 'Hf': -1*10**1, 'initial_amount': 8},
    {'name': 'O2',     'm': 10**-33,   'Zint': 7*10**4, 'Hf': -1*10**1, 'initial_amount': 1},
    {'name': 'e',      'm': 10**-35,   'Zint': 1*10**0, 'Hf': -1*10**-2, 'initial_amount': 0},
    {'name': 'C+',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**-1, 'initial_amount': 0},
    {'name': 'CF4',    'm': 10**-33,   'Zint': 10**4, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'CF2O',   'm': 10**-33,   'Zint': 9*10**5, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'CF2',    'm': 10**-33,   'Zint': 10**4, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'CF3',    'm': 10**-33,   'Zint': 5*10**4, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'CO',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'C3F',    'm': 10**-33,   'Zint': 4*10**3, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'O+',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'O+2',    'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'OF',     'm': 10**-33,   'Zint': 3*10**4, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'O-',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'F2',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'F+',     'm': 10**-33,   'Zint': 5*10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'F+2',    'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'F-',     'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'C',      'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0},
    {'name': 'O',      'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0}, 
    {'name': 'F',      'm': 10**-33,   'Zint': 10**2, 'Hf': -1*10**1, 'initial_amount': 0}
]

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
Zint = np.array([p['Zint'] for p in particles_data])
Hf = np.array([p['Hf'] for p in particles_data])
n_sum = A.T @ initial_amounts

# 目的関数の定義（ギブスの自由エネルギー）
def gibbs_free_energy(n,T,mu0):
    total_n = np.sum(n)
    x = n / total_n  # モル分率
    x = np.clip(x, 1e-10, None)  # 0以下の値を避けるためのクリッピング
    mu = mu0 + R * T * np.log(x)
    return np.dot(mu, n)

constraints = [{"type": "eq", "fun": lambda n, j=j: np.dot(A[:, j], n) - n_sum[j]} for j in range(A.shape[1])]
bounds = [(0, None) for _ in particles_data]

# 初期推定値の設定（各粒子の初期物質量）
n = initial_amounts
temperature_range = np.arange(10, 3100, 1000)
species_num = []
for T in temperature_range:
    T = np.clip(T, 1e-10, None)
    #Zの定義を更新してください
    Zv=100
    Zr=100
    Zint=Zv*Zr
    Ztr = (k * T / P0 )* (2 * np.pi * m * k * T / hp**2)**1.5
    mu0 = -(R * T * np.log(Ztr * Zint)) 
    print(mu0)

    # 最適化の実行
    result = minimize(gibbs_free_energy, n, method='SLSQP', args=(T, mu0), bounds=bounds, constraints=constraints,options={'maxiter': 1000, 'ftol': 1e-9})
    #if result.success:
    species_num.append(result.x)

    # 結果の表示
    print("T=",T)
    print(result)
    print('Optimal amounts of substances:', result.x)
    #print(mu0)
    print()

# プロット用の色のサイクルを設定
colors = plt.cm.tab20(np.linspace(0, 1, len(particles_data)))
linestyles = ['-', '--'] * (len(particles_data) // 2)
plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles)))
species_num = np.array(species_num)
for i, particle in enumerate(particles_data):
    plt.plot(temperature_range[:len(species_num)], species_num[:, i], label=particle['name'])

# グラフの設定
plt.xlabel('Temperature (K)')
plt.ylabel('Amount of Substance (mol)')
plt.title('Amounts of Substances vs Temperature')
plt.legend()
plt.grid(True)

# グラフの表示
plt.show()
