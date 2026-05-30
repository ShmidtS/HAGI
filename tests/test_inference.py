import pytest


torch = pytest.importorskip("torch")

from hagi.inference import ChatSession, generate


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
        logits[:, -1, 1] = 1.0
        return logits


class MockTokenizer:
    def encode(self, text: str):
        return [ord(char) % 8 for char in text]

    def decode(self, ids):
        return "".join(str(int(token)) for token in ids)


def test_generate_produces_expected_shape_with_greedy_sampling():
    prompt = torch.tensor([[2, 3, 4]], dtype=torch.long)

    generated = generate(TinyModel(), prompt, max_new_tokens=5, temperature=0)

    assert generated.shape == (1, 8)
    assert torch.equal(generated[:, :3], prompt)


def test_chat_session_basic_flow_with_mock_tokenizer():
    session = ChatSession(TinyModel(), MockTokenizer(), max_new_tokens=2, temperature=0)

    session.add_user_message("hi")
    response = session.generate_response()

    assert response == "11"
    assert session.history[0] == ("user", "hi")
    assert session.history[-1] == ("assistant", "11")
