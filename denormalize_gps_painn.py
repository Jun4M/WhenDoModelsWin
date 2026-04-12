"""
denormalize_gps_painn.py
GPS와 PaiNN 파일만 raw_data_normalized/ → raw_data/로 denormalize해서 복사.
다른 모델(GCN, RF 등)은 건드리지 않음.
"""

import os
import pandas as pd

RESULTS_DIR = './results'
STD_LOOKUP  = './std_lookup.csv'

FOLDER_TO_DATASET = {
    '01_QM9':  'qm9',
    '02_ESOL': 'esol',
    '03_Lipo': 'lipo',
    '04_BACE': 'bace',
}

DATASET_TARGETS = {
    'qm9':  ['homo', 'lumo', 'gap'],
    'esol': ['measured log solubility in mols per litre'],
    'lipo': ['exp'],
    'bace': ['pIC50'],
}

TARGET_MODELS = ['gps', 'painn']

lookup_df = pd.read_csv(STD_LOOKUP)

for folder, dataset in FOLDER_TO_DATASET.items():
    base_dir  = os.path.join(RESULTS_DIR, folder)
    norm_dir  = os.path.join(base_dir, 'raw_data_normalized')
    raw_dir   = os.path.join(base_dir, 'raw_data')
    targets   = DATASET_TARGETS[dataset]

    if not os.path.exists(norm_dir):
        print(f'[SKIP] {folder}: raw_data_normalized 없음')
        continue

    n_ok = 0
    for fname in sorted(os.listdir(norm_dir)):
        # 타겟 모델인지 확인
        if not any(fname.startswith(m + '_') for m in TARGET_MODELS):
            continue

        # target, seed 파싱
        seed, target = None, None
        for t in targets:
            suffix = f'_{t}.csv'
            if fname.endswith(suffix):
                prefix = fname[:-len(suffix)]
                parts = prefix.split('_')
                try:
                    seed = int(parts[-1])
                    target = t
                except ValueError:
                    pass
                break
        if seed is None:
            print(f'  [WARN] 파싱 실패: {fname}')
            continue

        src = os.path.join(norm_dir, fname)
        dst = os.path.join(raw_dir, fname)

        df = pd.read_csv(src)

        denormed_rows = []
        for _, row in df.iterrows():
            train_size = int(row['train_size'])
            std_row = lookup_df[
                (lookup_df['dataset']    == dataset) &
                (lookup_df['target']     == target) &
                (lookup_df['train_size'] == train_size) &
                (lookup_df['seed']       == seed)
            ]
            if std_row.empty:
                print(f'  [WARN] std 없음: {dataset}/{target}/size={train_size}/seed={seed}')
                std = 1.0
            else:
                std = float(std_row['std_train'].iloc[0])

            denormed_rows.append({
                'train_size': train_size,
                'RMSE':       row['RMSE'] * std,
                'MAE':        row['MAE']  * std,
                'Pearson_R':  row['Pearson_R'],
                'R2':         row['R2'],
                'n_test':     row['n_test'],
            })

        out_df = pd.DataFrame(denormed_rows)
        out_df.to_csv(dst, index=False)
        n_ok += 1

    print(f'[{folder}] {n_ok}개 파일 denormalize 완료 → raw_data/')

print('\nDone.')
