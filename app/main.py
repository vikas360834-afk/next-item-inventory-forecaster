"""Streamlit UI for next-item retail forecasting and inventory planning."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.core import PRODUCTS,generate_mock_transactions,milestone_2,milestone_5
from app.features import build_inventory,inventory_report,association_report,department_summary,transactions_from_csv
from app.storage import Store
from app.transformer import TransformerForecaster

st.set_page_config(page_title="Next-Item Inventory Forecaster",page_icon="🛒",layout="wide")
st.title("🛒 Next-Item Inventory Forecaster")
st.caption("Predict what shoppers may buy next · Plan replenishment · Understand basket patterns")
store=Store()
@st.cache_resource
def load_data():
    rows=store.get("transactions") or generate_mock_transactions(1200,42)
    model=milestone_2(rows)
    stock=store.get("inventory") or build_inventory(rows)
    store.put("transactions",rows); store.put("inventory",stock)
    return rows,model,stock
transactions,ngram,inventory=load_data()
if "custom_rows" in st.session_state:
    transactions=st.session_state.custom_rows
    ngram=milestone_2(transactions)
    inventory=st.session_state.get("custom_inventory",build_inventory(transactions))
transformer=None
checkpoint=Path("data/sku_transformer.pt")
if checkpoint.exists():
    try: transformer=TransformerForecaster.load(str(checkpoint))
    except Exception as exc: st.warning(f"Could not load Transformer checkpoint: {exc}")
k1,k2,k3,k4=st.columns(4)
k1.metric("Transactions",f"{len(transactions):,}")
k2.metric("SKU catalogue",len(PRODUCTS))
k3.metric("Training tokens",f"{sum(map(len,transactions)):,}")
k4.metric("Reorder alerts",sum(x["status"]=="Reorder" for x in inventory))
forecast,stock,analytics,associations,training=st.tabs(["🔮 Basket forecast","📦 Inventory planner","📊 Sales analytics","🧺 Bought together","🧠 Model & data"])
with forecast:
    st.subheader("Enter an incomplete shopping basket")
    labels={f"{name} ({sku})":sku for sku,name in PRODUCTS.items()}
    selected=st.multiselect("Products already in the basket",list(labels),placeholder="Search for a product")
    basket=[labels[x] for x in selected]
    engine=st.radio("Prediction model",["N-gram baseline"]+(["Causal Transformer"] if transformer else []),horizontal=True)
    top_k=st.select_slider("Number of recommendations",options=[3,5,7,10],value=5)
    exclude=st.checkbox("Hide products already in basket",value=True)
    if st.button("Predict next items",type="primary"):
        if not basket: st.warning("Add at least one product first.")
        else:
            model=transformer if engine=="Causal Transformer" else ngram
            st.session_state.predictions=model.predict_next(basket,top_k=top_k,exclude_seen=exclude)
    if "predictions" in st.session_state:
        preds=st.session_state.predictions
        if preds:
            st.subheader("Recommended next purchases")
            cols=st.columns(min(5,len(preds)))
            for i,p in enumerate(preds):
                with cols[i%len(cols)]: st.metric(p["name"],f'{p["probability"]:.1%}',f'Rank #{i+1}')
            st.dataframe(pd.DataFrame([{"Rank":i+1,"SKU":p["sku"],"Product":p["name"],"Relative score":p["probability"]} for i,p in enumerate(preds)]),hide_index=True,use_container_width=True)
            st.caption("Displayed scores are normalized ranking scores, not calibrated purchase probabilities.")
        else: st.info("No eligible items. Try disabling the hide-seen option.")
with stock:
    st.subheader("Stock health and replenishment")
    report=inventory_report(transactions,inventory)
    st.dataframe(pd.DataFrame(report),hide_index=True,use_container_width=True)
    st.download_button("Download inventory CSV",pd.DataFrame(report).to_csv(index=False),"inventory_report.csv","text/csv")
with analytics:
    st.subheader("Sales analytics")
    counts=pd.Series({PRODUCTS[s]:sum(r.count(s) for r in transactions) for s in PRODUCTS}).sort_values(ascending=False)
    left,right=st.columns(2)
    with left: st.markdown("**Units by product**"); st.bar_chart(counts)
    with right: st.markdown("**Units by department**"); st.bar_chart(pd.DataFrame(department_summary(transactions)).set_index("Department"))
    st.dataframe(counts.rename("Units in baskets").rename_axis("Product").reset_index(),hide_index=True)
with associations:
    st.subheader("Frequently bought together")
    pairs=association_report(transactions,st.slider("Minimum pair count",1,20,3))
    if pairs: st.dataframe(pd.DataFrame(pairs),hide_index=True,use_container_width=True)
    else: st.info("No product pairs meet this threshold.")
with training:
    st.subheader("Train and evaluate models")
    st.write(f"Current data: {len(transactions):,} baskets and {sum(map(len,transactions)):,} SKU tokens.")
    epochs=st.slider("Transformer training epochs",1,50,12)
    if st.button("Train causal Transformer",type="primary"):
        with st.spinner("Training on SKU sequences…"):
            model=TransformerForecaster(PRODUCTS)
            losses=model.fit(transactions,epochs=epochs)
            model.save(str(checkpoint)); transformer=model
        st.success(f"Transformer trained and saved. Final training loss: {losses[-1]:.4f}")
        st.line_chart(pd.DataFrame({"Cross-entropy loss":losses}))
    if st.button("Evaluate n-gram against popularity baseline"):
        st.json(milestone_5(n=len(transactions),seed=42,top_k=5))
    uploaded=st.file_uploader("Import transactions CSV (transaction_id, sku)",type=["csv"])
    if uploaded and st.button("Import and use uploaded data"):
        try:
            rows=transactions_from_csv(uploaded.getvalue())
            st.session_state.custom_rows=rows
            st.session_state.custom_inventory=build_inventory(rows)
            store.put("transactions",rows); store.put("inventory",st.session_state.custom_inventory)
            st.success(f"Imported {len(rows)} baskets. Refresh the page to use them across all tabs.")
        except Exception as exc: st.error(str(exc))
    if Path("data/mock_transactions.json").exists():
        try:
            sample=json.loads(Path("data/mock_transactions.json").read_text())
            st.caption(f"Bundled sample dataset: {len(sample.get('transactions',[]))} transactions")
        except Exception: pass
