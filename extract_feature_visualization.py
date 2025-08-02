import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# エクセルから画像パスとラベルを取得
labels_df = pd.read_excel("labels.xlsx", sheet_name="image_details")

# 結果を格納
data = []

# 特徴量計算関数
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像読み込み失敗: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 彩度
    saturation = np.mean(img_hsv[:, :, 1])

    # ノイズレベル（ラプラシアンの分散）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

    # テクスチャ複雑度（GLCMエネルギー）
    gray_8bit = cv2.resize(gray, (64, 64))
    glcm = feature.greycomatrix(gray_8bit, distances=[1], angles=[0], symmetric=True, normed=True)
    texture_energy = feature.greycoprops(glcm, 'energy')[0, 0]

    # 色温度（青の割合）
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    b_mean = np.mean(img_rgb[:, :, 2])
    color_temp = b_mean / (r_mean + 1e-5)

    return saturation, noise_level, texture_energy, color_temp

# データ収集
for idx, row in labels_df.iterrows():
    img_path = row["image_path"]
    label = row["label"]

# for img_path in glob.glob(f"{image_folder}/*.jpg"):  # 拡張子変更可能
#     fname = img_path.split("/")[-1]
#     label_row = labels_df[labels_df["filename"] == fname]
#     if label_row.empty:
#         continue  # ラベルなしはスキップ

    # label = label_row["label"].values[0]  # 正確 or 不正確
    # ラベルをエンコード（4クラス）
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    
    print(le.classes_)  # どの順番で数字に対応してるか確認
    
    if not os.path.exists(img_path):
        print(f"ファイルが存在しません: {img_path}")
        continue

    features = extract_features(img_path)
    if features is None:
        continue

    saturation, noise_level, texture_energy, color_temp = features

    data.append({
        "image_path": img_path,
        "saturation": saturation,
        "noise_level": noise_level,
        "texture_energy": texture_energy,
        "color_temp": color_temp,
        "label": label
    })

# DataFrame化
df = pd.DataFrame(data)
print(df.head())

# 可視化例
sns.boxplot(data=df, x="label", y="saturation")
plt.title("Saturation by Label")
plt.show()

sns.boxplot(data=df, x="label", y="noise_level")
plt.title("Noise Level by Label")
plt.show()

sns.boxplot(data=df, x="label", y="texture_energy")
# plt.title("Texture Energy by Label")
plt.title("Saturation by Label (4クラス)")
plt.show()

sns.boxplot(data=df, x="label", y="color_temp")
plt.title("Color Temperature by Label")
plt.show()

# モデル構築
X = df[["saturation", "noise_level", "texture_energy", "color_temp"]]
y = df["label"].map({"正確": 0, "不正確": 1})
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 重要特徴量
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance")
plt.show()

# BGR
import cv2
import numpy as np

def extract_features_bgr_safe(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像読み込み失敗: {image_path}")
        return None

    # imgはBGR形式のまま

    # 彩度 (BGRから直接HSV変換可能)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(img_hsv[:, :, 1])

    # ノイズレベル（ラプラシアンの分散）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

    # テクスチャ複雑度（GLCMエネルギー）
    gray_8bit = cv2.resize(gray, (64, 64))
    from skimage import feature
    glcm = feature.greycomatrix(gray_8bit, distances=[1], angles=[0], symmetric=True, normed=True)
    texture_energy = feature.greycoprops(glcm, 'energy')[0, 0]

    # 色温度（青と赤の割合：BGR順なので注意）
    b_mean = np.mean(img[:, :, 0])  # B
    g_mean = np.mean(img[:, :, 1])  # G
    r_mean = np.mean(img[:, :, 2])  # R

    color_temp = b_mean / (r_mean + 1e-5)

    return saturation, noise_level, texture_energy, color_temp

import seaborn as sns
import matplotlib.pyplot as plt

# 散布図1：彩度 × ノイズレベル
sns.scatterplot(data=df, x="saturation", y="noise_level", hue="label")
plt.title("Saturation vs Noise Level by Label")
plt.show()

# 散布図2：彩度 × テクスチャ複雑度
sns.scatterplot(data=df, x="saturation", y="texture_energy", hue="label")
plt.title("Saturation vs Texture Energy by Label")
plt.show()

# 散布図3：ノイズレベル × テクスチャ複雑度
sns.scatterplot(data=df, x="noise_level", y="texture_energy", hue="label")
plt.title("Noise Level vs Texture Energy by Label")
plt.show()

# 単純ルールの例（しきい値は散布図を見て調整）

# iPro弱い画像と判定するルール例（仮定）
# 〇/× または ×/× を「iPro弱い」とする
df["iPro_weak"] = df["label"].isin(["〇/×", "×/×"])

# 例：彩度50以下 & ノイズレベル500以下を危険判定
df["simple_rule_flag"] = (df["saturation"] <= 50) & (df["noise_level"] <= 500)

# ルールの精度確認
from sklearn.metrics import confusion_matrix, classification_report

y_true = df["iPro_weak"]
y_pred = df["simple_rule_flag"]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))



import cv2
import numpy as np
from skimage import feature

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像読み込み失敗: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 彩度
    saturation = np.mean(img_hsv[:, :, 1])

    # ノイズレベル（ラプラシアンの分散）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

    # テクスチャ複雑度（GLCMエネルギー）
    gray_8bit = cv2.resize(gray, (64, 64))
    glcm = feature.greycomatrix(gray_8bit, distances=[1], angles=[0], symmetric=True, normed=True)
    texture_energy = feature.greycoprops(glcm, 'energy')[0, 0]

    # 色温度（青の割合）
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    b_mean = np.mean(img_rgb[:, :, 2])
    color_temp = b_mean / (r_mean + 1e-5)

    # BGR平均
    b_mean_bgr = np.mean(img[:, :, 0])
    g_mean_bgr = np.mean(img[:, :, 1])
    r_mean_bgr = np.mean(img[:, :, 2])

    # brightness（明るさ）
    brightness = np.mean(gray)

    # sharpness（鮮鋭度、ノイズレベルと同様）
    sharpness = noise_level  # ラプラシアン分散と同じ指標

    # edge_density（エッジ密度）
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # contrast（コントラスト）
    contrast = np.std(gray)

    return (saturation, noise_level, texture_energy, color_temp,
            b_mean_bgr, g_mean_bgr, r_mean_bgr,
            brightness, sharpness, edge_density, contrast)
features = extract_features("sample.jpg")
if features:
    (saturation, noise_level, texture_energy, color_temp,
     b_mean_bgr, g_mean_bgr, r_mean_bgr,
     brightness, sharpness, edge_density, contrast) = features



#クラス毎に出す
import seaborn as sns
import matplotlib.pyplot as plt

# 例の特徴量リスト
feature_list = ["saturation", "noise_level", "texture_energy", "color_temp"]

# すべてのtrue_label（20クラス）のループ
for true_label in df["true_label"].unique():

    # 該当クラスのデータだけ抽出
    sub_df = df[df["true_label"] == true_label]

    for feature in feature_list:
        plt.figure(figsize=(6, 4))

        # xは判定ラベル（〇/〇、×/×など）、yは特徴量
        sns.boxplot(data=sub_df, x="label", y=feature)

        plt.title(f"{true_label} - {feature}")
        plt.show()

