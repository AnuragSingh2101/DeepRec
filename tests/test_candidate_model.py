"""Unit tests for CandidateGenerator model architecture, shapes, loss, and checkpointing."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.deeprec.models.candidate_generation import CandidateGenerator
from src.deeprec.training.losses import SampledSoftmaxLoss, BPRLoss
from src.deeprec.retrieval.retrieve import CandidateRetriever


def test_candidate_generator_shapes_and_forward():
    num_items = 50
    embedding_dim = 16
    hidden_dims = [32, 16]
    batch_size = 4
    max_history_len = 5

    model = CandidateGenerator(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        use_context=False,
    )
    model.eval()

    # Synthetic batch of histories
    history = torch.randint(0, num_items + 1, (batch_size, max_history_len))

    # Test encode output shape
    user_vec = model.encode(history)
    assert user_vec.shape == (batch_size, embedding_dim)

    # Test score with specific candidate IDs
    cands = torch.randint(1, num_items + 1, (batch_size, 10))
    scores = model.score(user_vec, cands)
    assert scores.shape == (batch_size, 10)

    # Test full catalogue score_all
    all_scores = model.score_all(user_vec)
    assert all_scores.shape == (batch_size, num_items + 1)
    # Ensure padding token score is -1e9
    assert torch.all(all_scores[:, 0] <= -1e8)


def test_candidate_generator_with_context():
    num_items = 20
    model = CandidateGenerator(
        num_items=num_items,
        embedding_dim=16,
        hidden_dims=[32, 16],
        use_context=True,
    )
    history = torch.randint(0, num_items + 1, (3, 4))
    context = {
        "gender": torch.tensor([0, 1, 0], dtype=torch.long),
        "occupation": torch.tensor([5, 12, 1], dtype=torch.long),
        "age": torch.tensor([-0.5, 0.2, 1.1], dtype=torch.float32),
    }

    user_vec = model.encode(history, context=context)
    assert user_vec.shape == (3, 16)
    assert not torch.isnan(user_vec).any()


def test_candidate_loss_and_backprop():
    num_items = 30
    model = CandidateGenerator(num_items=num_items, embedding_dim=16, hidden_dims=[32, 16], use_context=False)
    criterion = SampledSoftmaxLoss()

    history = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    positives = torch.tensor([6, 7])
    negatives = torch.tensor([[8, 9, 10], [11, 12, 13]])

    user_vec = model.encode(history)
    pos_scores = model.score(user_vec, positives)
    neg_scores = model.score(user_vec, negatives)

    loss = criterion(pos_scores, neg_scores)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # Backprop
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Grad is None for {name}"


def test_checkpoint_save_and_load(tmp_path):
    num_items = 25
    model = CandidateGenerator(num_items=num_items, embedding_dim=16, hidden_dims=[32, 16], use_context=False)
    model.eval()
    chk_path = tmp_path / "cand_test.pt"

    torch.save({
        "num_items": num_items,
        "embedding_dim": 16,
        "hidden_dims": [32, 16],
        "use_context": False,
        "model_state_dict": model.state_dict()
    }, chk_path)

    loaded_chk = torch.load(chk_path, map_location="cpu", weights_only=False)
    loaded_model = CandidateGenerator(
        num_items=loaded_chk["num_items"],
        embedding_dim=loaded_chk["embedding_dim"],
        hidden_dims=loaded_chk["hidden_dims"],
        use_context=loaded_chk.get("use_context", False),
    )
    loaded_model.load_state_dict(loaded_chk["model_state_dict"])
    loaded_model.eval()

    # Compare predictions
    dummy_hist = torch.tensor([[1, 2, 3, 0]])
    with torch.no_grad():
        out_orig = model.encode(dummy_hist)
        out_loaded = loaded_model.encode(dummy_hist)
    torch.testing.assert_close(out_orig, out_loaded)



def test_retriever_cold_start_fallback():
    num_items = 20
    model = CandidateGenerator(num_items=num_items, embedding_dim=8, hidden_dims=[16, 8])
    popular = [3, 7, 2, 9, 1]
    retriever = CandidateRetriever(
        model=model,
        num_items=num_items,
        device=torch.device("cpu"),
        popular_items=popular,
    )

    # Empty history (all zeros)
    empty_hist = [0, 0, 0, 0]
    cands, scores = retriever.retrieve(empty_hist, k=3)
    assert cands == [3, 7, 2]
    assert len(scores) == 3
