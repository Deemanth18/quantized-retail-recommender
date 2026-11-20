"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st
import pandas as pd
from app.core import (
    milestone_1,
    milestone_2,
    milestone_5,
    milestone_6,
    predict_next_items,
    FREQUENT_RULES,
)

st.set_page_config(page_title="Quantized Retail Recommender", layout="wide")
st.title("🛒 Quantized Retail Recommender")
st.write(
    "A smart retail recommender that identifies frequently bought items using a small language model, "
    "compressed using post-training quantization for edge device deployment."
)

@st.cache_resource
def get_data_and_models():
    data = milestone_1()
    model, _ = milestone_2(data=data, epochs=2, max_baskets=1500)
    dyn_model, stat_model = milestone_5(model)
    comparison = milestone_6(model=model, data=data)
    return data, model, dyn_model, stat_model, comparison

with st.spinner("Initializing models and running quantization benchmark..."):
    data, model, dyn_model, stat_model, comparison = get_data_and_models()

tab1, tab2, tab3 = st.tabs(["🛍️ Shopping Basket Recommender", "📊 Quantization Comparison", "📜 Frequent Itemsets"])

with tab1:
    st.subheader("Interactive Next-Item Recommendation")
    model_choice = st.radio(
        "Choose Model Version:",
        ["Baseline (FP32)", "Dynamic Quantized (INT8)", "Static Quantized (INT8)"],
        horizontal=True,
    )

    active_model = model
    if "Dynamic" in model_choice:
        active_model = dyn_model
    elif "Static" in model_choice:
        active_model = stat_model

    selected_items = st.multiselect(
        "Current Items in Basket:",
        options=data["catalog"],
        default=["Bread White", "Butter 200g"],
    )

    if selected_items:
        recs = predict_next_items(active_model, data, selected_items, top_k=5)
        st.write("### Recommended Next Items:")
        for idx, (item, prob) in enumerate(recs, 1):
            st.progress(min(prob, 1.0), text=f"**#{idx} {item}** — {prob * 100:.1f}% confidence")
    else:
        st.info("Select items above to see next-item recommendations.")

with tab2:
    st.subheader("Model Quantization & Footprint Benchmark")
    base = comparison["baseline"]
    dyn = comparison["dynamic_quantized"]
    stat = comparison["static_quantized"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline FP32 Size", f"{base['size_mb']} MB")
        st.metric("Baseline Top-5 Accuracy", f"{base['top5_accuracy'] * 100:.1f}%")
        st.metric("Baseline Latency", f"{base['latency_ms']} ms")

    with col2:
        st.metric("Dynamic INT8 Size", f"{dyn['size_mb']} MB", delta=f"-{dyn['size_reduction_pct']}% size")
        st.metric("Dynamic Top-5 Accuracy", f"{dyn['top5_accuracy'] * 100:.1f}%")
        st.metric("Dynamic Latency", f"{dyn['latency_ms']} ms")

    with col3:
        st.metric("Static INT8 Size", f"{stat['size_mb']} MB", delta=f"-{stat['size_reduction_pct']}% size")
        st.metric("Static Top-5 Accuracy", f"{stat['top5_accuracy'] * 100:.1f}%")
        st.metric("Static Latency", f"{stat['latency_ms']} ms")

    df_comp = pd.DataFrame([base, dyn, stat])
    st.dataframe(df_comp, use_container_width=True)

with tab3:
    st.subheader("Planted Association Rules")
    rules_data = [{"Antecedents": ", ".join(v), "Consequent": t} for v, t in FREQUENT_RULES]
    st.table(rules_data)

