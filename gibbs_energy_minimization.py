import numpy as np
from scipy.optimize import minimize
import re

# 定数の定義
R = 8.314  # J/(mol*K)
T = 298.15  # 温度（K）

# 粒子の種類と基準化学ポテンシャル、初期物質量を定義
particles_data = [
    {'name': 'C5F10O', 'mu0': 10, 'initial_amount': 1},
    {'name': 'CO2',    'mu0': 50, 'initial_amount': 8},
    {'name': 'O2',     'mu0': 3,  'initial_amount': 1},
    {'name': 'C+',     'mu0': 1,  'initial_amount': 0},
    {'name': 'CF4',      'mu0': 1,  'initial_amount': 0},
    {'name': 'CF2O',      'mu0': 1,  'initial_amount': 0},
    {'name': 'CF2',      'mu0': 1,  'initial_amount': 0},
    {'name': 'CF3',      'mu0': 1,  'initial_amount': 0},
    {'name': 'CO',      'mu0': 1,  'initial_amount': 0},
    {'name': 'C3F',      'mu0': 1,  'initial_amount': 0},
    {'name': 'C',      'mu0': 1,  'initial_amount': 0},
    {'name': 'O',      'mu0': 1,  'initial_amount': 0},
    {'name': 'O+',      'mu0': 1,  'initial_amount': 0},
    {'name': 'O+2',      'mu0': 1,  'initial_amount': 0},
    {'name': 'OF',      'mu0': 1,  'initial_amount': 0},
    {'name': 'O-',      'mu0': 1,  'initial_amount': 0},
    {'name': 'F',      'mu0': 1,  'initial_amount': 0},
    {'name': 'F2',      'mu0': 1,  'initial_amount': 0},
    {'name': 'F+',      'mu0': 1,  'initial_amount': 0},
    {'name': 'F+2',      'mu0': 1,  'initial_amount': 0},
    {'name': 'F-',      'mu0': 1,  'initial_amount': 0}
]

# 各粒子の名前から組成を生成する関数
def parse_composition(name):
    composition = {'C': 0, 'F': 0, 'O': 0, 'e': 0}
    matches = re.findall(r'([A-Z][a-z]?|[-+])(\d*)', name)
    for (element, count) in matches:
        if count == '':
            count = 1
        else:
            count = int(count)
        if (element=='-'):
            element='e'
        elif(element=='+'):
            element='e'
            count=(-count)
        composition[element] += count
    return [composition['C'], composition['F'], composition['O'], composition['e']]

# 各粒子の組成を生成し、リストを更新
for particle in particles_data:
    particle['composition'] = parse_composition(particle['name'])
#print(particles_data)

# 各粒子が含む原子の数を行列Aとして定義[C, F, O, e]
A = np.array([p['composition'] for p in particles_data])

# 基準化学ポテンシャルと初期物質量のリストを作成
mu0 = np.array([p['mu0'] for p in particles_data])
initial_amounts = np.array([p['initial_amount'] for p in particles_data])

# 各原子の総物質量を計算
n_j = A.T @ initial_amounts

# 目的関数の定義（ギブスの自由エネルギー）
def gibbs_free_energy(n):
    total_n = np.sum(n)
    x = n / total_n  # モル分率
    x = np.clip(x, 1e-10, None)  # 0以下の値を避けるためのクリッピング
    mu = mu0 + R * T * np.log(x)
    return np.dot(mu, n)

# 制約条件の定義
constraints = [{'type': 'eq', 'fun': lambda n, j=j: np.dot(A[:, j], n) - n_j[j]} for j in range(A.shape[1])]

# 変数の範囲（境界条件）の定義（0以上の範囲とする）
bounds = [(0, None) for _ in particles_data]

# 初期推定値の設定（各粒子の初期物質量）
n0 = initial_amounts

# 最適化の実行

result = minimize(gibbs_free_energy, n0, bounds=bounds, constraints=constraints)
for j in range(A.shape[1]):
    print(f'Constraint {j}: {np.dot(A[:, j], result.x) - n_j[j]}')

# 結果の表示
print('Optimal Gibbs free energy:', result.fun)
print('Optimal amounts of substances:', result.x)
