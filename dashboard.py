import pandas as pd
import streamlit as st

from app.store import get_store, summarize

st.set_page_config(page_title="LLM Gateway", layout="wide")
st.title("LLM Gateway: cost & routing")

rows = get_store().all()
if not rows:
    st.info("No requests logged yet. Call POST /chat first.")
    st.stop()

s = summarize(rows)
c = st.columns(5)
c[0].metric("Requests", s["requests"])
c[1].metric("Escalation rate", f"{s['escalation_rate']:.0%}")
c[2].metric("Total cost", f"${s['total_cost_usd']:.5f}")
c[3].metric("Always-strong (est.)", f"${s['strong_equiv_cost_usd']:.5f}")
c[4].metric("Est. savings", f"{s['est_savings_pct']:.1f}%")

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])

left, right = st.columns(2)
with left:
    st.subheader("Requests by route")
    st.bar_chart(df["route"].value_counts())
with right:
    st.subheader("Latency (ms) by route")
    st.dataframe(df.groupby("route")["latency_ms"].describe(percentiles=[0.5, 0.95])[["count", "50%", "95%", "max"]])

st.subheader("Cumulative cost vs. always-strong estimate")
cum = df.sort_values("timestamp").reset_index(drop=True)[["cost_usd", "strong_equiv_cost_usd"]].cumsum()
st.line_chart(cum)

st.subheader("Recent requests")
st.dataframe(df.sort_values("timestamp", ascending=False).head(50), width="stretch")
