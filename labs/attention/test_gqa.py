"""CPU correctness checks; run: python -m unittest discover -s labs/attention -v."""

import unittest

import torch
from torch.nn import functional as F

from gqa import GQA


class GQATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(2, 7, 12, dtype=torch.float64)

    def reference(self, model, x, causal):
        b, s, _ = x.shape
        q = model.w_q(x).view(b, s, model.num_q_heads, model.head_dim).transpose(1, 2)
        k = model.w_k(x).view(b, s, model.num_kv_heads, model.head_dim).transpose(1, 2)
        v = model.w_v(x).view(b, s, model.num_kv_heads, model.head_dim).transpose(1, 2)
        # Each query head uses the corresponding compact KV head directly.
        heads = [F.scaled_dot_product_attention(
            q[:, i:i+1], k[:, i//model.group_size:i//model.group_size+1],
            v[:, i//model.group_size:i//model.group_size+1],
            dropout_p=0.0, is_causal=causal,
        ) for i in range(model.num_q_heads)]
        out = torch.cat(heads, dim=1).transpose(1, 2).reshape(b, s, -1)
        return model.w_o(out)

    def test_against_sdpa_including_mha_and_mqa(self):
        for kv_heads in (1, 2, 4):
            for causal in (False, True):
                with self.subTest(kv_heads=kv_heads, causal=causal):
                    model = GQA(12, 4, kv_heads, 3).double()
                    torch.testing.assert_close(model(self.x, is_causal=causal),
                                               self.reference(model, self.x, causal))

    def test_future_tokens_do_not_change_prefix(self):
        model = GQA(12, 4, 2, 3).double()
        changed = self.x.clone()
        changed[:, 3:] += 20
        torch.testing.assert_close(model(self.x)[:, :3], model(changed)[:, :3])

    def test_token_and_chunk_cache_match_full_causal_output(self):
        model = GQA(12, 4, 2, 3).double().eval()
        with torch.no_grad():
            expected = model(self.x)
            for chunks in ((1, 1, 1, 1, 1, 1, 1), (3, 2, 2)):
                outputs, cache, start = [], None, 0
                for length in chunks:
                    out, cache = model(self.x[:, start:start+length],
                                       past_key_value=cache, use_cache=True)
                    outputs.append(out)
                    start += length
                    self.assertEqual(cache[0].shape, (2, 2, start, 3))
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected)

    def test_input_and_parameter_gradients_match_sdpa(self):
        model = GQA(12, 4, 2, 3).double()
        x = self.x.clone().requires_grad_()
        targets = [x, *model.parameters()]
        actual = torch.autograd.grad(model(x).square().sum(), targets)
        expected = torch.autograd.grad(self.reference(model, x, True).square().sum(), targets)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_invalid_head_counts(self):
        for args in ((12, 3, 2, 4), (12, 4, 0, 3), (12, 4, 2, 0)):
            with self.assertRaises(ValueError):
                GQA(*args)


if __name__ == "__main__":
    unittest.main()
