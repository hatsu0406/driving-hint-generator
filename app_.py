"""
運転免許試験 補足説明生成システム
② Streamlitアプリ本体（ヒント生成版）
 
事前に build_knowledge_base.py を実行し、
kaisetsu_index.faiss と kaisetsu_df.pkl を
このファイルと同じフォルダに用意しておくこと。
 
APIキーは .streamlit/secrets.toml に以下の形式で設定する：
    GOOGLE_API_KEY = "ここにGemini APIキー"
 
事前に以下をインストールしておくこと：
    pip install streamlit sentence-transformers faiss-cpu pandas google-genai
実行方法：
    streamlit run app_.py
"""
 
import os
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
 
# ------------------------------------------------------------
# 初期設定
# ------------------------------------------------------------
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
INDEX_PATH = "kaisetsu_index.faiss"
DF_PATH = "kaisetsu_df.pkl"
 
st.set_page_config(page_title="運転免許試験 ヒントジェネレーター")
 
# APIキーの設定（.streamlit/secrets.toml から読み込む）
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
 
 
# ------------------------------------------------------------
# リソースの読み込み（重い処理はキャッシュして使い回す）
# ------------------------------------------------------------
@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_pickle(DF_PATH)
    client = genai.Client()
    return embed_model, index, df, client
 
 
embed_model, index, df, client = load_resources()
 
 
# ------------------------------------------------------------
# ① 検索（Retrieval）
# ------------------------------------------------------------
def search(query_text: str, top_k: int = 3) -> pd.DataFrame:
    query_vec = embed_model.encode([query_text], normalize_embeddings=True)
    scores, idxs = index.search(query_vec, top_k)
    return df.iloc[idxs[0]]
 
 
# ------------------------------------------------------------
# ② プロンプト作成（ヒント生成版）
# ------------------------------------------------------------
def build_prompt(question: str, retrieved_rows: pd.DataFrame) -> str:
    context = "\n".join(
        f"- 類似問題:{row['問題文']} / 解答:{row['解答']} / 解説:{row['解説']}"
        for _, row in retrieved_rows.iterrows()
    )
 
    return f"""あなたは運転免許試験の学習をサポートする先生です。
以下の問題について、参考資料(類似問題の解答・解説)を読んで内容を理解したうえで、
「正しい」か「誤り」かを直接明言せずに、受験者自身が考えられるようなヒントを作成してください。
 
【厳守事項】
- 「正しい」「誤り」という言葉、またはそれを直接示唆する断定的な表現は使わない
- 参考資料の文章をそのまま引用・言い換えコピーしない
- 問題文のどの言葉・どの条件に注目すべきかを示す
- 関連する交通ルールの考え方を、答えを教えない範囲で示す
 
【問題文】{question}
 
【参考資料(類似問題の解答・解説。ヒント作成の参考にするだけで、そのまま出力しないこと)】
{context}
 
【出力形式】
ヒント: (100文字程度、です・ます調)
"""
 
 
# ------------------------------------------------------------
# ③ 生成（Generation）
# ------------------------------------------------------------
def generate_explanation(question: str, top_k: int = 3):
    retrieved_rows = search(question, top_k=top_k)
    prompt = build_prompt(question, retrieved_rows)
 
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
    return response.text, retrieved_rows
 
 
# ------------------------------------------------------------
# ④ 画面（UI）
# ------------------------------------------------------------
st.title("運転免許試験 ヒントジェネレーター")
st.write("問題文を入力すると、AIが答えを明言せずにヒントを生成します。")
 
question = st.text_input("問題文を入力してください")
 
if st.button("ヒントを生成"):
    if not question:
        st.warning("問題文を入力してください")
    else:
        with st.spinner("検索・生成中..."):
            result_text, retrieved_rows = generate_explanation(question)
 
        st.subheader("生成されたヒント")
        st.write(result_text)
 
        with st.expander("参考にした類似問題（検索結果）を見る"):
            st.dataframe(retrieved_rows[["問題文", "解答", "解説"]])
 

