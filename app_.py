"""
運転免許試験 補足説明生成システム
③ Streamlitアプリ本体（キーワードマッチ＋RAGフォールバック＋心理学的バイアス判定版）

事前に必要なもの：
    - kaisetsu_index.faiss / kaisetsu_df.pkl（RAGフォールバック用、build_knowledge_base.pyで作成）
    - Googleスプレッドシート（シート「本体」「候補」の2枚構成）
    - サービスアカウントのJSONキー

.streamlit/secrets.toml に以下を設定する：
    GOOGLE_API_KEY = "Gemini APIキー"
    APPROVAL_PASSWORD = "承認タブ用のパスワード"
    SPREADSHEET_URL = "GoogleスプレッドシートのURL"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "..."
    client_email = "..."
    client_id = "..."
    ...（ダウンロードしたJSONの中身をそのままTOML形式で貼り付け）

事前に以下をインストールしておくこと：
    pip install streamlit sentence-transformers faiss-cpu pandas google-genai gspread google-auth

実行方法：
    streamlit run app_.py
"""

import os
import re
import datetime
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import gspread
from google.oauth2.service_account import Credentials

# ------------------------------------------------------------
# 初期設定
# ------------------------------------------------------------
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
INDEX_PATH = "kaisetsu_index.faiss"
DF_PATH = "kaisetsu_df.pkl"

BIAS_TYPES = {
    1: "①ヒューリスティック型",
    2: "②アンカリング型",
    3: "③処理流暢性型",
    4: "④論理構造負荷型",
    5: "⑤絶対表現型",
}

st.set_page_config(page_title="運転免許試験 ヒントジェネレーター", layout="wide")

# APIキーの設定
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

    # Googleスプレッドシート接続
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=scopes
    )
    gc = gspread.authorize(creds)
    spreadsheet = gc.open_by_url(st.secrets["SPREADSHEET_URL"])
    sheet_main = spreadsheet.worksheet("本体")
    sheet_candidate = spreadsheet.worksheet("候補")

    return embed_model, index, df, client, sheet_main, sheet_candidate


embed_model, index, df, client, sheet_main, sheet_candidate = load_resources()


@st.cache_data(ttl=60)  # 1分キャッシュ（承認直後の反映を考慮）
def load_keyword_table():
    records = sheet_main.get_all_records()
    return pd.DataFrame(records)


# ------------------------------------------------------------
# ① キーワードマッチ検索
# ------------------------------------------------------------
def keyword_match(question_text: str, keyword_df: pd.DataFrame):
    """問題文にキーワード表のkeyword1/keyword2が含まれるか調べる。一致した行を返す。"""
    for _, row in keyword_df.iterrows():
        kw1 = str(row.get("keyword1", "")).strip()
        kw2 = str(row.get("keyword2", "")).strip()
        if kw1 and kw1 in question_text:
            return row
        if kw2 and kw2 != "nan" and kw2 in question_text:
            return row
    return None


# ------------------------------------------------------------
# ② RAGフォールバック検索（埋め込み類似度）
# ------------------------------------------------------------
def rag_search(query_text: str, top_k: int = 3) -> pd.DataFrame:
    query_vec = embed_model.encode([query_text], normalize_embeddings=True)
    scores, idxs = index.search(query_vec, top_k)
    return df.iloc[idxs[0]]


# ------------------------------------------------------------
# ③ バイアスタイプ判定（ルールベース＋AI判定のハイブリッド）
# ------------------------------------------------------------
def detect_bias_type_rule_based(question_text: str):
    absolute_words = ["必ず", "絶対に", "常に", "一切", "決して", "例外なく"]
    if any(word in question_text for word in absolute_words):
        return 5

    numbers = re.findall(r"\d+", question_text)
    if len(numbers) >= 2:
        return 2

    return None


# ------------------------------------------------------------
# ④ 候補シートへの自動記録
# ------------------------------------------------------------
def append_candidate(question_text: str, rag_context: str, bias_type: int):
    try:
        sheet_candidate.append_row([
            question_text,
            rag_context,
            BIAS_TYPES.get(bias_type, ""),
            "未確認",
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ])
    except Exception as e:
        # 記録に失敗してもヒント生成自体は止めない
        st.warning(f"候補シートへの記録に失敗しました: {e}")


# ------------------------------------------------------------
# ⑤ プロンプト作成（バイアスタイプ別に誘導文を変える）
# ------------------------------------------------------------
BIAS_GUIDANCE = {
    1: "受験者が既有知識だけで即答しがちな問題です。「その知識は正しいですが、今回の問題文に例外条件が隠れていないか」を確認させるヒントにしてください。",
    2: "数字や基準点に判断が引っ張られやすい問題です。「最初に出てきた数字を一旦忘れて、文章全体の条件を確認する」よう促すヒントにしてください。",
    3: "文章が自然で読みやすく、正しいと感じやすい問題です。「読みやすい文章ほど、一文ずつ区切って条件を確認する」よう促すヒントにしてください。",
    4: "複数条件や否定表現が絡み合い、論理構造が複雑な問題です。「条件を一つずつ整理して、何が対象で何が対象外かを分けて考える」よう促すヒントにしてください。",
    5: "「必ず」「絶対に」といった断定表現を含む問題です。「その断定に例外がないか、もう一度疑ってみる」よう促すヒントにしてください。",
}


def build_prompt(question: str, context: str, bias_type: int) -> str:
    """バイアスタイプがルールベースで既に確定している場合に使う、ヒント生成専用プロンプト。"""
    guidance = BIAS_GUIDANCE.get(bias_type, "")
    return f"""あなたは運転免許試験の学習をサポートする先生です。
以下の問題について、参考資料を読んで内容を理解したうえで、
「正しい」か「誤り」かを直接明言せずに、受験者自身が考えられるようなヒントを作成してください。

【この問題の誤答傾向】{guidance}

【厳守事項】
- 「正しい」「誤り」という言葉、またはそれを直接示唆する断定的な表現は使わない
- 参考資料の文章をそのまま引用・言い換えコピーしない
- 問題文のどの言葉・どの条件に注目すべきかを示す
- 関連する交通ルールの考え方を、答えを教えない範囲で示す

【問題文】{question}

【参考資料(ヒント作成の参考にするだけで、そのまま出力しないこと)】
{context}

【出力形式】
ヒント: (100文字程度、です・ます調)
"""


def build_prompt_with_bias_detection(question: str, context: str) -> str:
    """
    バイアスタイプがルールベースで判定できなかった場合に使う、
    「バイアス判定」と「ヒント生成」を1回のAPI呼び出しにまとめたプロンプト。
    """
    return f"""あなたは運転免許試験の学習をサポートする先生であり、認知バイアスの研究者でもあります。
以下の問題文について、2つのことを行ってください。

【作業1】この問題が誤答を生みやすい心理学的な理由を、次の3タイプのいずれかに分類してください。
①ヒューリスティック型：関連する知識・スキーマが先に発動し、例外条件を読み飛ばして即座に判断してしまうパターン
③処理流暢性型：文章が自然で読みやすいほど「正しい」と感じられやすいパターン
④論理構造負荷型：二重否定や複数条件の併記など、文の論理構造自体が複雑で誤読を生むパターン

【作業2】分類結果を踏まえて、「正しい」か「誤り」かを直接明言せず、受験者自身が考えられるようなヒントを作成してください。

【厳守事項（ヒント作成時）】
- 「正しい」「誤り」という言葉、またはそれを直接示唆する断定的な表現は使わない
- 参考資料の文章をそのまま引用・言い換えコピーしない
- 問題文のどの言葉・どの条件に注目すべきかを示す
- 関連する交通ルールの考え方を、答えを教えない範囲で示す

【問題文】{question}

【参考資料(ヒント作成の参考にするだけで、そのまま出力しないこと)】
{context}

【出力形式(この形式を厳守し、他の文章は含めないこと)】
バイアスタイプ: (1, 3, 4のいずれか1つの数字のみ)
ヒント: (100文字程度、です・ます調)
"""


# ------------------------------------------------------------
# ⑥ 生成（Generation）メインパイプライン
# ------------------------------------------------------------
def parse_combined_response(raw_text: str):
    """バイアスタイプ＋ヒントを同時に返したレスポンスをパースする。"""
    bias_match = re.search(r"バイアスタイプ[：:]\s*([134])", raw_text)
    bias_type = int(bias_match.group(1)) if bias_match else 1

    hint_match = re.search(r"ヒント[：:]\s*(.+)", raw_text, re.DOTALL)
    hint_text = hint_match.group(1).strip() if hint_match else raw_text.strip()

    return bias_type, hint_text


def generate_explanation(question: str):
    keyword_df = load_keyword_table()
    matched_row = keyword_match(question, keyword_df)

    if matched_row is not None:
        # キーワード一致 → 本体シートの知識を使う
        context = matched_row.get("supplement", "")
        source_info = f"キーワード一致: {matched_row.get('keyword1', '')}"
    else:
        # 未一致 → RAGフォールバック
        retrieved_rows = rag_search(question, top_k=3)
        context = "\n".join(
            f"- 類似問題:{row['問題文']} / 解答:{row['解答']} / 解説:{row['解説']}"
            for _, row in retrieved_rows.iterrows()
        )
        source_info = "RAGフォールバック（キーワード未一致）"

    rule_result = detect_bias_type_rule_based(question)

    if rule_result is not None:
        # ⑤絶対表現・②アンカリングはルールで確定 → API呼び出しは1回（ヒント生成のみ）
        bias_type = rule_result
        prompt = build_prompt(question, context, bias_type)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        hint_text = response.text
    else:
        # ①③④はAPI1回で「判定」と「ヒント生成」を同時に行う
        prompt = build_prompt_with_bias_detection(question, context)
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        bias_type, hint_text = parse_combined_response(response.text)

    if matched_row is None:
        # 未知の問題として候補シートに自動記録（API呼び出し後、判定結果が出てから記録）
        append_candidate(question, context, bias_type)

    return hint_text, source_info, BIAS_TYPES.get(bias_type, "")


# ------------------------------------------------------------
# ⑦ 画面（UI）
# ------------------------------------------------------------
st.title("運転免許試験 ヒントジェネレーター")

tab_hint, tab_admin = st.tabs(["ヒント生成", "ナレッジ管理（研究用）"])

# --- タブ1: ヒント生成 ---
with tab_hint:
    st.write("問題文を入力すると、AIが答えを明言せずにヒントを生成します。")

    question = st.text_input("問題文を入力してください")

    if st.button("ヒントを生成"):
        if not question:
            st.warning("問題文を入力してください")
        else:
            with st.spinner("検索・生成中..."):
                result_text, source_info, bias_label = generate_explanation(question)

            st.subheader("生成されたヒント")
            st.write(result_text)

            with st.expander("デバッグ情報（研究用）"):
                st.write(f"参照元: {source_info}")
                st.write(f"推定バイアスタイプ: {bias_label}")

# --- タブ2: ナレッジ管理（パスワード保護） ---
with tab_admin:
    st.subheader("ナレッジ管理（研究用）")

    password = st.text_input("パスワードを入力してください", type="password")

    if password != st.secrets["APPROVAL_PASSWORD"]:
        if password:
            st.error("パスワードが違います")
        st.stop()

    st.success("認証されました")

    candidate_records = sheet_candidate.get_all_records()
    candidate_df = pd.DataFrame(candidate_records)

    if candidate_df.empty:
        st.info("現在、未確認の候補はありません。")
    else:
        pending = candidate_df[candidate_df["ステータス"] == "未確認"]

        if pending.empty:
            st.info("現在、未確認の候補はありません。")

        for idx, row in pending.iterrows():
            with st.container(border=True):
                st.write(f"**問題文**: {row['問題文']}")
                st.write(f"**RAGの参考情報**: {row['RAGの参考情報']}")
                st.write(f"**推定バイアスタイプ**: {row['推定バイアスタイプ']}")

                col1, col2 = st.columns(2)

                # スプレッドシート上の実際の行番号（ヘッダー行+1オフセット）
                sheet_row_num = idx + 2

                with col1:
                    if st.button("承認して本体に追加", key=f"approve_{idx}"):
                        # 本体シートに追加（キーワードは問題文の先頭部分を仮に使用、要手動調整）
                        sheet_main.append_row([
                            row["問題文"][:15],  # 仮のキーワード（後で手動修正推奨）
                            "",
                            row["RAGの参考情報"],
                        ])
                        sheet_candidate.update_cell(sheet_row_num, 4, "承認済み")
                        st.cache_data.clear()
                        st.rerun()

                with col2:
                    if st.button("却下", key=f"reject_{idx}"):
                        sheet_candidate.update_cell(sheet_row_num, 4, "却下")
                        st.rerun()
