"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch

from app.core import (
    BasketLanguageModel,
    StaticQuantizedLinear,
    apply_dynamic_quantization,
    apply_static_quantization,
    get_model_size_bytes,
    measure_baseline,
    measure_model_accuracy,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    predict_next_items,
    prepare_synthetic_baskets,
    probe_itemsets,
)


# Milestone 1: Prepare synthetic shopping basket data for training
def test_milestone_1():
    data = milestone_1()

    assert "baskets" in data
    assert "catalog" in data
    assert "vocab" in data
    assert "item_to_id" in data
    assert "verbalized" in data

    baskets = data["baskets"]
    assert len(baskets) >= 1000
    assert len(data["catalog"]) >= 10
    assert len(data["vocab"]) > len(data["catalog"])

    # Check sample basket structure
    first_basket = baskets[0]
    assert isinstance(first_basket, list)
    assert len(first_basket) >= 1
    for item in first_basket:
        assert item in data["catalog"]
        assert item in data["item_to_id"]

    # Check verbalization
    assert data["verbalized"][0].startswith("Basket:")


# Milestone 2: Continue pretraining a small LM on the basket data
def test_milestone_2():
    data = prepare_synthetic_baskets(n_baskets=200, seed=1)
    model, returned_data = milestone_2(data=data, epochs=1, max_baskets=100)

    assert isinstance(model, BasketLanguageModel)
    assert returned_data is data

    # Test forward pass with batch
    dummy_input = torch.tensor([[1, 3, 4], [1, 5, 0]], dtype=torch.long)
    logits, _ = model(dummy_input)

    assert logits.shape == (2, 3, len(data["vocab"]))
    assert not torch.isnan(logits).any()


# Milestone 3: Implement next-item prediction and itemset probing
def test_milestone_3():
    data = prepare_synthetic_baskets(n_baskets=300, seed=42)
    model, _ = milestone_2(data=data, epochs=1, max_baskets=150)

    # Test next-item prediction
    current = ["Bread White", "Butter 200g"]
    preds = predict_next_items(model, data, current, top_k=5)

    assert len(preds) == 5
    for item_name, prob in preds:
        assert isinstance(item_name, str)
        assert item_name in data["catalog"]
        assert item_name not in current  # Already in basket should not be recommended
        assert 0.0 <= prob <= 1.0

    # Test itemset probing
    probe_res = probe_itemsets(model, data)
    assert "top1_accuracy" in probe_res
    assert "top5_accuracy" in probe_res
    assert "mrr" in probe_res
    assert 0.0 <= probe_res["top5_accuracy"] <= 1.0
    assert 0.0 <= probe_res["mrr"] <= 1.0

    # Test milestone_3 wrapper
    m3_res = milestone_3(model=model, data=data)
    assert "sample_predictions" in m3_res
    assert "probe_metrics" in m3_res


# Milestone 4: Measure the baseline size and accuracy of the model
def test_milestone_4():
    data = prepare_synthetic_baskets(n_baskets=200, seed=12)
    model, _ = milestone_2(data=data, epochs=1, max_baskets=100)

    baseline = milestone_4(model=model, data=data)

    assert baseline["model_type"] == "baseline_fp32"
    assert baseline["size_bytes"] > 0
    assert baseline["size_mb"] > 0
    assert baseline["num_parameters"] > 0
    assert 0.0 <= baseline["top5_accuracy"] <= 1.0
    assert 0.0 <= baseline["mrr"] <= 1.0
    assert baseline["latency_ms"] >= 0.0


# Milestone 5: Apply dynamic and static post-training quantization to the model
def test_milestone_5():
    data = prepare_synthetic_baskets(n_baskets=150, seed=5)
    model, _ = milestone_2(data=data, epochs=1, max_baskets=100)

    dyn_model, static_model = milestone_5(model=model)

    assert dyn_model is not None
    assert static_model is not None

    # Test forward pass with dynamic quantized model
    test_input = torch.tensor([[1, 2, 4]], dtype=torch.long)
    dyn_logits, _ = dyn_model(test_input)
    assert dyn_logits.shape == (1, 3, len(data["vocab"]))
    assert not torch.isnan(dyn_logits).any()

    # Test forward pass with static quantized model
    static_logits, _ = static_model(test_input)
    assert static_logits.shape == (1, 3, len(data["vocab"]))
    assert not torch.isnan(static_logits).any()


# Milestone 6: Demonstrate size reduction and evaluate accuracy impact
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

