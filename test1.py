import re
import numpy as np
import os

# 定数の定義
Na = 6.02214076e23  # アボガドロ定数
hartree_to_kjmol = 2625.49962

# 各原子の質量 (kg)
atomic_masses = {
    'C': 12.011e-3 / Na,  # kg
    'F': 18.998e-3 / Na,  # kg
    'O': 15.999e-3 / Na,  # kg
    'e': 9.10938356e-31  # kg (電子)
}

# 各原子の標準生成エンタルピー [kJ/mol] (NISTデータ)
standard_enthalpy_atoms = {
    'C': 716.68,  # kJ/mol
    'F': 78.99,   # kJ/mol
    'O': 249.19   # kJ/mol
}

# 各原子の全電子エネルギー [kJ/mol] (NISTデータ)
electronic_energy_atoms = {
    'C': -37.789 * hartree_to_kjmol,  # kJ/mol
    'F': -99.726 * hartree_to_kjmol,  # kJ/mol
    'O': -75.06 * hartree_to_kjmol    # kJ/mol
}

# 各粒子の組成を計算する関数
def parse_composition(name):
    composition = {'C': 0, 'O': 0, 'F': 0, 'e': 0}
    matches = re.findall(r'([A-Z][a-z]?|[-+e])(\d*)', name)
    for (element, count) in matches:
        if count == '':
            count = 1
        else:
            try:
                count = int(count)
            except ValueError:
                count = 1  # デフォルト値を設定
        if element == '-':
            element = 'e'
        elif element == '+':
            element = 'e'
            count = -count
        composition[element] += count
    return composition

# ガウシアンの出力ファイルからエネルギー値と振動周波数、回転定数を抽出する関数
def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = {}
    current_molecule = None
    n = 0
    m = 0

    for line in lines:
        # Extract molecule name
        if '%Chk=' in line:
            chk_file = line.split('=')[1].strip()
            # Remove .chk extension to match particle_name
            current_molecule = chk_file.replace('.chk', '')
            data[current_molecule] = {'E0': None, 'EZPE': None, 'frequencies': [], 'rotational_constants': [], 'symmetry': None}
            n = 0
            m = 0
        # Extract electronic energy
        if 'SCF Done:' in line and current_molecule:
            match = re.search(r'[-]?\d+\.\d+', line)
            if match:
                E0 = float(match.group())
                data[current_molecule]['E0'] = E0 * hartree_to_kjmol
        # Extract zero-point energy
        if 'Zero-point correction=' in line and current_molecule:
            match = re.search(r'[-]?\d+\.\d+', line)
            if match:
                EZPE = float(match.group())
                data[current_molecule]['EZPE'] = EZPE * hartree_to_kjmol
        # Extract vibrational frequencies
        if 'Frequencies --' in line and current_molecule:
            freqs = re.findall(r'\d+\.\d+', line)
            data[current_molecule]['frequencies'].extend(freqs)
        # Extract rotational constants
        if 'Rotational constants (GHZ)' in line and n == 0 and current_molecule:
            constants = re.findall(r'\d+\.\d+', line)
            data[current_molecule]['rotational_constants'].append(constants)
            n += 1
        # Extract symmetry
        if 'Full point group' in line and m == 0 and current_molecule:
            symmetry = line.split()[-1]
            data[current_molecule]['symmetry'] = symmetry
            m += 1

    return data

def calculate_standard_enthalpy(data, composition, molecule_name):
    try:
        E0_M = data['E0']
        EZPE_M = data['EZPE']
        E0_X = sum(composition[element] * electronic_energy_atoms[element] for element in composition if element in electronic_energy_atoms)
        Hf_X = sum(composition[element] * standard_enthalpy_atoms[element] for element in composition if element in standard_enthalpy_atoms)
        Hf_M = Hf_X - E0_X + E0_M - EZPE_M
        return Hf_M
    except KeyError:
        return None

def calculate_and_write(input_file_path, output_file_path, freq_file_path):
    # ファイル存在の確認
    if not os.path.exists(freq_file_path):
        print(f"Error: The file {freq_file_path} does not exist.")
        return
    
    if not os.path.exists(input_file_path):
        print(f"Error: The file {input_file_path} does not exist.")
        return

    data = extract_data(freq_file_path)
    
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
    
    with open(output_file_path, 'a') as file:
        # ファイルヘッダーの書き込み（必要に応じてコメントアウトを解除）
        #file.write("particle,initial amount,m,Rot,Freq,Symmetry,Standard Enthalpy \n")
        for line in lines:
            particle_name = line.strip()
            composition = parse_composition(particle_name)
            mass = sum(composition[element] * atomic_masses[element] for element in composition)
            
            # `.chk` を取り除いたキーを使用
            freq_data = data.get(particle_name, {})
            freq_list = freq_data.get('frequencies', [])
            freq_str = "  ".join(freq_list)
            
            rota_list = freq_data.get('rotational_constants', [])
            rota_str_list = ["  ".join(constants) for constants in rota_list]
            rota_str = " ; ".join(rota_str_list)
            
            symmetry = freq_data.get('symmetry', "None")
            
            standard_enthalpy = calculate_standard_enthalpy(freq_data, composition, particle_name)
            
            if standard_enthalpy is not None:
                file.write(f"{particle_name}, 0 ,{mass} ,{rota_str},{freq_str},{symmetry},{standard_enthalpy}\n")
            else:
                print(f"Warning: Could not calculate standard enthalpy for {particle_name}")
                file.write(f"{particle_name}, 0 ,{mass} ,{rota_str},{freq_str},{symmetry},{standard_enthalpy}\n")
                
# ファイルパスを指定して質量を計算し、結果を書き込む
input_file_path = 'particles.txt'
output_file_path = 'minimize_inputs.txt'
freq_file_path = 'particle_results/C5F10O_result'  # ガウシアンの出力ファイルのパス
calculate_and_write(input_file_path, output_file_path, freq_file_path)