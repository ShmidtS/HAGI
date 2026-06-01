import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hagi.data.sft_dataset import SFTDataset, _encode_conversation, _sft_collate


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"<{msg['role']}>: {msg['content']}")
        return "\n".join(parts)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_encode_conversation_masks_assistant_only():
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    example = _encode_conversation(messages, tokenizer, max_seq_len=256)
    # system + user should be -100, assistant should have real token ids
    # compute lengths matching _encode_conversation formatting
    def _fmt(msg):
        return f"<{msg['role']}>{msg['content']}</{msg['role']}>"

    system_text = _fmt(messages[0])
    user_text = _fmt(messages[1])
    assistant_text = _fmt(messages[2])
    sys_len = len(tokenizer.encode(system_text))
    user_len = len(tokenizer.encode(user_text))
    assistant_len = len(tokenizer.encode(assistant_text))
    # full_text concatenates turns without separators
    full_text = "".join([system_text, user_text, assistant_text])
    full_len = len(tokenizer.encode(full_text))
    # labels for system and user should be -100
    assert all(label == -100 for label in example.labels[:sys_len])
    assert all(label == -100 for label in example.labels[sys_len : sys_len + user_len])
    # labels for assistant should match input_ids
    assistant_start = sys_len + user_len
    assistant_end = assistant_start + assistant_len
    assert example.labels[assistant_start:assistant_end] == example.input_ids[assistant_start:assistant_end]
    assert len(example.input_ids) == 256  # padded to max_seq_len
    assert all(t == 0 for t in example.input_ids[full_len:])


def test_encode_conversation_truncates_and_pads():
    tokenizer = _FakeTokenizer()
    messages = [
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
    ]
    max_seq_len = 32
    example = _encode_conversation(messages, tokenizer, max_seq_len=max_seq_len)
    assert len(example.input_ids) == max_seq_len
    assert len(example.labels) == max_seq_len


def test_sft_collate_builds_tensors():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 5]},
        {"input_ids": [4, 5, 6], "labels": [-100, 7, 8]},
    ]
    x, y = _sft_collate(batch)
    assert tuple(x.shape) == (2, 3)
    assert tuple(y.shape) == (2, 3)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_sft_dataset_item_structure():
    class _FakeRow:
        def __init__(self, messages):
            self._messages = messages

        def __getitem__(self, key):
            if key == "messages":
                return self._messages
            raise KeyError(key)

    class _FakeDataset:
        def __init__(self, rows):
            self._rows = rows

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, index):
            return self._rows[index]

    rows = [
        _FakeRow([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]),
    ]
    ds = SFTDataset(_FakeDataset(rows), tokenizer=_FakeTokenizer(), max_seq_len=64)
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert item["input_ids"].shape == (64,)
    assert item["labels"].shape == (64,)
