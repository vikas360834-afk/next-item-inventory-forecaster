import pytest
torch=pytest.importorskip("torch")
from app.core import PRODUCTS
from app.transformer import TransformerForecaster

def test_transformer_trains_and_predicts():
    rows=[["SKU001","SKU002","SKU003"],["SKU001","SKU002","SKU004"]]*4
    model=TransformerForecaster(PRODUCTS,max_len=8)
    losses=model.fit(rows,epochs=2,batch_size=4)
    assert len(losses)==2 and losses[-1]>0
    preds=model.predict_next(["SKU001","SKU002"],top_k=3)
    assert len(preds)==3
    assert all(0<=p["probability"]<=1 for p in preds)

def test_transformer_rejects_unknown_sku():
    model=TransformerForecaster(PRODUCTS)
    model.fit([["SKU001","SKU002"]],epochs=1)
    with pytest.raises(ValueError): model.predict_next(["NOT_A_SKU"])
