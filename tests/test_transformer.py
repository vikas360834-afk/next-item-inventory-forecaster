import torch

from app.transformer import SKUTransformer, encode_sequences, predict_skus, train_transformer


def test_transformer_forward_shape():
    model = SKUTransformer(vocab_size=8, d_model=16, nhead=4, num_layers=1)
    x = torch.tensor([[0, 1, 2]])
    assert model(x).shape == (1, 3, 8)


def test_train_and_predict():
    sequences = [["A", "B", "C"], ["A", "B", "D"], ["A", "C", "D"]]
    model, vocab, losses = train_transformer(sequences, epochs=2, d_model=16)
    assert len(losses) == 2
    preds = predict_skus(model, vocab, ["A", "B"], top_k=2)
    assert len(preds) == 2
    assert all("sku" in p and "score" in p for p in preds)
