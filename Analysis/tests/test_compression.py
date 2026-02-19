import base64
import numpy as np
import pytest


# Mock implementation of what we will add to static_report.py
class TokenDictionary:
    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.tokens: list[str] = []

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.tokens)
            self.tokens.append(token)
        return self.token_to_id[token]

    def encode(self, tokens: list[str]) -> str:
        indices = [self.add(t) for t in tokens]
        # Use uint32 to be safe for now, or uint16 if we want max consistency
        arr = np.array(indices, dtype=np.uint32)
        return base64.b64encode(arr.tobytes()).decode("ascii")


def encode_float_array(values: list[float]) -> str:
    arr = np.array(values, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def test_encoders():
    # Float encoding
    floats = [0.1, 0.2, 0.3, 100.5]
    encoded_floats = encode_float_array(floats)
    decoded_bytes = base64.b64decode(encoded_floats)
    decoded_arr = np.frombuffer(decoded_bytes, dtype=np.float32)
    assert np.allclose(floats, decoded_arr)

    # Token encoding
    d = TokenDictionary()
    tokens = ["hello", "world", "hello"]
    encoded_tokens = d.encode(tokens)

    # Check dictionary state
    assert d.tokens == ["hello", "world"]
    assert d.token_to_id["hello"] == 0
    assert d.token_to_id["world"] == 1

    # Check encoded indices
    decoded_idx_bytes = base64.b64decode(encoded_tokens)
    decoded_indices = np.frombuffer(decoded_idx_bytes, dtype=np.uint32)
    expected_indices = [0, 1, 0]
    assert np.array_equal(decoded_indices, expected_indices)
