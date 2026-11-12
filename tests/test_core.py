import numpy as np
import torch

from ple_lab import GPT, GPTConfig, load_run, load_table_memmap, save_run


def tiny(ple=16):
    return GPTConfig(vocab_size=97, block_size=16, n_layer=3, n_head=2, d_model=32, ple_dim=ple)


def test_shapes_and_loss():
    for ple in (0, 16):
        m = GPT(tiny(ple))
        ids = torch.randint(0, 97, (2, 10))
        logits, loss = m(ids, ids)
        assert logits.shape == (2, 10, 97) and torch.isfinite(loss)


def test_ple_mask_zero_equals_ctx_only():
    m = GPT(tiny()).eval()
    ids = torch.randint(0, 97, (1, 8))
    x0 = m.wte(ids) + m.wpe(torch.arange(8))
    u = m.per_layer_inputs(ids, x0, ple_mask=torch.zeros(1, 1, 3, 1))
    assert torch.allclose(u, m.ple_ctx_part(x0) * 2 ** -0.5)


def test_roundtrip(tmp_path):
    m = GPT(tiny()).eval()
    save_run(m, tmp_path, {"val_loss": 1.0})
    m2 = load_run(tmp_path)
    ids = torch.randint(0, 97, (2, 10))
    a, _ = m(ids)
    b, _ = m2(ids)
    assert torch.allclose(a, b, atol=1e-2)  # fp16 table storage
    rows, scale, meta = load_table_memmap(tmp_path, "i8")
    deq = rows.astype(np.float32) * scale[:, None]
    ref = m.ple_table.table.weight.detach().numpy()
    assert np.abs(deq - ref).max() < 0.01 and meta["row_width"] == 48


def test_load_without_table_then_attach(tmp_path):
    m = GPT(tiny()).eval()
    save_run(m, tmp_path)
    m2 = load_run(tmp_path, load_table=False)

    class Src(torch.nn.Module):
        def __init__(self):
            super().__init__()
            rows, _, _ = load_table_memmap(tmp_path)
            self.rows = torch.from_numpy(np.asarray(rows, dtype=np.float32))

        def forward(self, ids):
            return self.rows[ids]

    m2.ple_table = Src()
    ids = torch.randint(0, 97, (1, 6))
    assert torch.allclose(m(ids)[0], m2(ids)[0], atol=1e-2)
