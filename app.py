import json

import streamlit as st

st.set_page_config(page_title="Amharic Next-Word Prediction", page_icon="🔤")


@st.cache_resource
def load_model(path="model.json"):
    """Load the pruned n-gram model. Keys are the order ("2", "3", "4");
    each order-n table maps an (n-1)-word context to {next_word: count}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(order): table for order, table in raw.items()}


MODEL = load_model()


def predict(text, order, top_k):
    words = text.strip().split()
    if not words:
        return []
    # Use the longest context available, backing off to shorter ones
    for k in range(min(order - 1, len(words)), 0, -1):
        table = MODEL.get(k + 1, {}).get(" ".join(words[-k:]))
        if table:
            total = sum(table.values())
            ranked = sorted(table.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [(word, count / total) for word, count in ranked]
    return []


st.title("Amharic Next-Word Prediction")
st.write("N-gram language model trained on the GPAC Amharic corpus.")

text = st.text_area("Your sentence", placeholder="ኢትዮጵያ ውስጥ የቡና", height=100)
order = st.slider("N-gram order", min_value=2, max_value=4, value=3,
                  help="Higher orders use more context words.")
top_k = st.slider("Number of suggestions", min_value=1, max_value=5, value=5)

if text.strip():
    results = predict(text, order, top_k)
    if results:
        st.subheader("Suggestions")
        for word, prob in results:
            st.write(f"**{word}**  —  {prob:.1%}")
    else:
        st.info("No suggestion for this input. Try a different or shorter phrase.")