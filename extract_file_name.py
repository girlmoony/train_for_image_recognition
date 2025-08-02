import pandas as pd
import json

# 判定結果シート読み込み（例）
excel_file = '判定結果まとめ.xlsx'
ipro_df = pd.read_excel(excel_file, sheet_name='判定結果')

# 属性情報のJSON
attr_json = {
    "1302@10": ["現行2", "旧型", "上", "左"],
    "1303@5": ["新型", "旧型", "下", "右"]
    # 必要な情報を追加
}
# JSONファイルのパス
json_file = 'attribute_info.json'

# JSON読み込み
with open(json_file, 'r', encoding='utf-8') as f:
    attr_json = json.load(f)

# 確認
print(attr_json)


# shopコードとseat番号の抽出関数
def extract_shop_seat(filename):
    parts = filename.split('_')
    shop = parts[1]
    seat_full = parts[2]
    seat = seat_full.split('-')[1]
    return shop, seat

def extract_shop_seat(filename):
    parts = filename.split('_')  
    # parts[0]から']'の位置を探して、その後ろを取得
    bracket_idx = parts[0].find(']')
    shop = parts[0][bracket_idx + 1:]  # ]の次の文字以降がshopコード
    
    # parts[1]のハイフン区切りの後ろがseat番号
    seat_full = parts[1]
    seat = seat_full.split('-')[1]
    
    return shop, seat

# shopとseat列を追加
ipro_df['shop'] = ipro_df['画像名'].apply(lambda x: extract_shop_seat(x)[0])
ipro_df['seat'] = ipro_df['画像名'].apply(lambda x: extract_shop_seat(x)[1])

# 属性列の追加
def get_attributes(row):
    key = f"{row['shop']}@{row['seat']}"
    return attr_json.get(key, ["不明", "不明", "不明", "不明"])

# 4つの属性列を分解して追加
ipro_df[['属性1', '属性2', '属性3', '属性4']] = ipro_df.apply(get_attributes, axis=1, result_type='expand')

# 結果確認
print(ipro_df[['画像名', 'shop', 'seat', '属性1', '属性2', '属性3', '属性4']].head())

# 必要なら新しいシートに保存
with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    ipro_df.to_excel(writer, sheet_name='判定結果_属性付加', index=False)

print('判定結果にshop, seat, 属性4列を追加し、新しいシートに保存しました。')

# 画像名の次に追加したい列
new_columns = ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']

# 既存列のリスト取得
cols = list(ipro_df.columns)

# 画像名の位置を取得
img_idx = cols.index('画像名')

# 新しい列順序を作成
reordered_cols = cols[:img_idx + 1] + new_columns + [col for col in cols if col not in new_columns and col != '画像名']

# 列順序を並び替え
ipro_df = ipro_df[reordered_cols]

# 元シートを上書き（シート名は同じまま）
with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    ipro_df.to_excel(writer, sheet_name='判定結果', index=False)

print('元シートにshop, seat, 属性4列を画像名の次に追加し、上書き保存しました。')

import pandas as pd
from scipy.stats import chi2_contingency

# データ読み込み（シートから読み込んだ前提）
df = pd.read_excel('判定結果まとめ.xlsx', sheet_name='判定結果')

# 判定結果から「IPROだけ間違い」= × / × もしくは 差分対象 のものだけを見る場合は、事前にフィルタしてください

# 属性ごとにクロス集計＆カイ二乗検定
for col in ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']:
    print(f"\n--- {col} と判定結果の関係 ---")
    
    # クロス集計
    ct = pd.crosstab(df[col], df['判定結果'])
    print(ct)
    
    # カイ二乗検定
    chi2, p, _, _ = chi2_contingency(ct)
    print(f"カイ二乗統計量: {chi2:.2f}, p値: {p:.4f}")
    
    if p < 0.05:
        print("→ 有意な関係あり（この属性と判定結果に強い関係がある可能性）")
    else:
        print("→ 有意な関係なし（この属性と判定結果に強い関係は見つからず）")
import pandas as pd
from scipy.stats import chi2_contingency
from openpyxl import load_workbook

# データ読み込み
excel_file = '判定結果まとめ.xlsx'
df = pd.read_excel(excel_file, sheet_name='判定結果')

# 結果を蓄積するリスト
results = []

# -------- 全体集計 --------
for col in ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']:
    ct = pd.crosstab(df[col], df['判定結果'])
    
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2, p, _, _ = chi2_contingency(ct)
        results.append({
            '集計対象': '全体',
            'クラス': '全体',
            '属性': col,
            'カイ二乗統計量': chi2,
            'p値': p
        })

# -------- クラス毎の集計 --------
for true_class in df['true_label'].unique():
    df_class = df[df['true_label'] == true_class]
    
    for col in ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']:
        ct = pd.crosstab(df_class[col], df_class['判定結果'])
        
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(ct)
            results.append({
                '集計対象': 'クラス別',
                'クラス': true_class,
                '属性': col,
                'カイ二乗統計量': chi2,
                'p値': p
            })

# 結果をDataFrame化
result_df = pd.DataFrame(results)

# Excelファイルに新しいシートとして追加
with pd.ExcelWriter(excel_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    result_df.to_excel(writer, sheet_name='カイ二乗検定結果', index=False)

print('カイ二乗検定結果を新しいシート「カイ二乗検定結果」に保存しました。')


import pandas as pd

# データ読み込み
df = pd.read_excel('判定結果まとめ.xlsx', sheet_name='判定結果')

# 認識結果フラグ（〇 / 〇 を正解とする）
df['正解フラグ'] = df['判定結果'].apply(lambda x: 1 if x.strip() == '〇 / 〇' else 0)

結果_全体 = []
結果_クラス別 = []

# -------- 全体認識率集計 --------
for col in ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']:
    集計 = df.groupby(col)['正解フラグ'].agg(['sum', 'count']).reset_index()
    集計['認識率(%)'] = (集計['sum'] / 集計['count']) * 100
    集計.insert(0, '分類単位', '全体')
    集計.rename(columns={col: '属性値', 'sum': '正解件数', 'count': '総件数'}, inplace=True)
    集計['属性名'] = col
    結果_全体.append(集計[['分類単位', '属性名', '属性値', '正解件数', '総件数', '認識率(%)']])

df_全体 = pd.concat(結果_全体, ignore_index=True)

# -------- クラス毎認識率集計 --------
for true_class in df['true_label'].unique():
    df_class = df[df['true_label'] == true_class]
    
    for col in ['shop', 'seat', '属性1', '属性2', '属性3', '属性4']:
        集計 = df_class.groupby(col)['正解フラグ'].agg(['sum', 'count']).reset_index()
        集計['認識率(%)'] = (集計['sum'] / 集計['count']) * 100
        集計.insert(0, '分類単位', true_class)
        集計.rename(columns={col: '属性値', 'sum': '正解件数', 'count': '総件数'}, inplace=True)
        集計['属性名'] = col
        結果_クラス別.append(集計[['分類単位', '属性名', '属性値', '正解件数', '総件数', '認識率(%)']])

df_クラス別 = pd.concat(結果_クラス別, ignore_index=True)

# -------- エクセルへ追記保存 --------
with pd.ExcelWriter('判定結果まとめ.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df_全体.to_excel(writer, sheet_name='認識率_全体', index=False)
    df_クラス別.to_excel(writer, sheet_name='認識率_クラス別', index=False)

print("認識率一覧をエクセルに保存しました。")


