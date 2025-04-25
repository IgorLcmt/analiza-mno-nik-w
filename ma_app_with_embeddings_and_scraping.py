# Streamlit App for M&A Comparable Transactions Finder (cleaned for deployment)
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="CMT analiza mnożników pod wycene 🔍", layout="wide")

# --- Constants ---
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 100
EXCEL_PATH = "app_data/Database.xlsx"

# --- Custom styling ---
st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1n76uvr {
        border-color: #80c7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("CMT analiza mnożników pod wycene 🔍")

# --- Load and preprocess the embedded Excel ---
@st.cache_data
def load_database():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        'Business Description\n(Target/Issuer)': 'Business Description',
        'Primary Industry\n(Target/Issuer)': 'Primary Industry'
    })
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(subset=[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry'
    ])
    return df

# --- Scrape website text or fallback to archive ---
def scrape_text(domain):
    try:
        res = requests.get(f"https://{domain}", timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except:
        pass
    try:
        archive_url = f"http://web.archive.org/web/{domain}"
        res = requests.get(archive_url, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except:
        return ""
    return ""

# --- Batch embedding using OpenAI ---
def get_embeddings(texts, api_key):
    openai.api_key = api_key
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        try:
            response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
            batch_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"OpenAI API error during batch {i // BATCH_SIZE}: {e}")
            raise
        time.sleep(1)
    return embeddings

# --- Embedding logic with composite text ---
def embed_database(df, api_key):
    df["Website Text"] = df["Web page"].fillna("").apply(scrape_text)
    df["Composite"] = df.apply(lambda row: " ".join(filter(None, [
        str(row["Business Description"]),
        str(row["Primary Industry"]),
        str(row["Website Text"])
    ])), axis=1)
    df["embedding"] = get_embeddings(df["Composite"].tolist(), api_key)
    return df

# --- Find top matches ---
def find_top_matches(df, query, api_key, top_n=10):
    query_embedding = get_embeddings([query], api_key)[0]
    emb_matrix = np.vstack(df["embedding"].values)
    emb_matrix_norm = normalize(emb_matrix)
    query_norm = normalize(np.array(query_embedding).reshape(1, -1))
    similarities = cosine_similarity(query_norm, emb_matrix_norm)[0]
    df["Similarity Score"] = similarities
    top = df.sort_values(by="Similarity Score", ascending=False).head(top_n).copy()
    top["Reason for Match"] = "High semantic + content + industry similarity"
    return top[[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry', 'Web page', 'Similarity Score', 'Reason for Match'
    ]]

# --- User Interface ---
api_key = st.secrets["openai"]["api_key"]
query_input = st.sidebar.text_area("✏️ Paste company profile here:", height=200)

if api_key and query_input:
    try:
        df = load_database()
        with st.spinner("Embedding and scraping in progress..."):
            df_prepared = embed_database(df, api_key)
        results = find_top_matches(df_prepared, query_input, api_key)
        st.success("Top matches found:")
        st.dataframe(results, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results.to_excel(writer, index=False, sheet_name="Top Matches")
        st.download_button("📥 Download Excel", data=output.getvalue(),
                           file_name="Top_Matches.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("<---- Enter company profile to begin.")
