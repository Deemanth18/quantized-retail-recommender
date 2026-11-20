"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch

from app.core import BasketLanguageModel, milestone_1, milestone_2, prepare_synthetic_baskets


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
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Measure the baseline size and accuracy of the model
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Apply dynamic and static post-training quantization to the model
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Demonstrate size reduction and evaluate accuracy impact
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

