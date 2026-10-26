import json
import random
import gradio as gr

def load_model(path="model/ngram.json"):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    model = {}
    for order, table in raw.items():
        model[int(order)] = {tuple(ctx.split()): nxt for ctx, nxt in table.items()}
    return model

MODEL = load_model()

def predict_next(text, order, top_k):
    words = text.strip().split()
    if not words:
        return "Please enter some Amharic text."
    order = int(order)
    # Back off to shorter contexts if the longer one is unseen
    for n in range(min(order, len(words)), 0, -1):
        ctx = tuple(words[-n:])
        table = MODEL.get(n, {}).get(ctx)
        if table:
            ranked = sorted(table.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)]
            total = sum(table.values())
            return "\n".join(f"{w}  ({c / total:.1%})" for w, c in ranked)
    return "No prediction for this input."

demo = gr.Interface(
    fn=predict_next,
    inputs=[
        gr.Textbox(label="Amharic text", placeholder="ቃሉን እዚህ ይጻፉ", lines=2),
        gr.Slider(1, 4, value=3, step=1, label="N-gram order"),
        gr.Slider(1, 10, value=5, step=1, label="Top-K suggestions"),
    ],
    outputs=gr.Textbox(label="Next-word suggestions"),
    title="Amharic Next-Word Prediction",
    description="N-gram (orders 1-4) model trained on 327,882 words.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)