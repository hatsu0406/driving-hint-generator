"""
運転免許試験 補足説明生成システム
① 知識ベース構築スクリプト

このスクリプトは最初に1回だけ実行する（Colab等で実行することを想定）。
エクセルファイル（問題文・解答・解説）を読み込み、
・欠損値の除去
・文字列型への変換
・埋め込みベクトル化
・faissインデックスの構築
を行い、Streamlitアプリ側で使えるようにファイルとして保存する。

事前に以下をインストールしておくこと：
    !pip install sentence-transformers faiss-cpu pandas openpyxl
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# ------------------------------------------------------------
# ① エクセルファイルの読み込み
# ------------------------------------------------------------
# 列名は実際のファイルに合わせて変更すること
# 想定している列: 問題文, 解答（正しい/誤り）, 解説
EXCEL_PATH = "kaisetsu.xlsx"

df = pd.read_excel(EXCEL_PATH)

# ------------------------------------------------------------
# ② 欠損値・型の整形
# ------------------------------------------------------------
# 解説が空欄の行は知識ベースとして使えないため除外する
df = df.dropna(subset=["解説"]).reset_index(drop=True)

# 念のため、数値だけが入っているセルなどを全て文字列型に統一する
df["問題文"] = df["問題文"].astype(str)
df["解答"] = df["解答"].astype(str)
df["解説"] = df["解説"].astype(str)

print(f"知識ベースとして使用する件数: {len(df)}件")

# ------------------------------------------------------------
# ③ 埋め込みモデルの読み込み・ベクトル化
# ------------------------------------------------------------
# 日本語に対応した多言語埋め込みモデルを使用
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

model = SentenceTransformer(EMBED_MODEL_NAME)

# 「解説」文をベクトル化する
# （検索は問題文の意味的な近さで行うが、知識として渡すのは解説なので
#   ここでは問題文をベクトル化対象にする）
texts_for_embedding = df["問題文"].tolist()

embeddings = model.encode(
    texts_for_embedding,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# ------------------------------------------------------------
# ④ faissインデックスの構築
# ------------------------------------------------------------
# normalize_embeddings=True にしているため、内積(IP)がコサイン類似度に相当する
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# ------------------------------------------------------------
# ⑤ 保存
# ------------------------------------------------------------
# Streamlitアプリ側では、このインデックスとdfを読み込むだけで検索できるようにする
faiss.write_index(index, "kaisetsu_index.faiss")
df.to_pickle("kaisetsu_df.pkl")

print("知識ベースの構築が完了しました。")
print("生成されたファイル: kaisetsu_index.faiss, kaisetsu_df.pkl")
print("この2つのファイルを、Streamlitアプリと同じフォルダに配置してください。")
