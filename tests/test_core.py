import pytest
from app.core import PRODUCTS,generate_mock_transactions,milestone_2,milestone_5

def test_mock_transactions_are_valid():
    rows=generate_mock_transactions(30,7)
    assert len(rows)==30
    assert all(len(r)>=2 and set(r)<=set(PRODUCTS) for r in rows)

def test_next_item_model_learns_sequence():
    model=milestone_2([["A","B","C"],["A","B","C"],["A","B","D"]])
    assert model.predict_next(["A","B"],3)[0]["sku"]=="C"

def test_evaluation_has_valid_metrics():
    result=milestone_5(n=150,top_k=3)
    assert result["examples"]>0
    assert 0<=result["model_hit_rate"]<=1
    assert 0<=result["popularity_hit_rate"]<=1
    assert 0<=result["model_mrr"]<=1

def test_unknown_sku_is_rejected():
    model=milestone_2([["A","B"],["A","C"]])
    with pytest.raises(ValueError): model.predict_next(["unknown"])
