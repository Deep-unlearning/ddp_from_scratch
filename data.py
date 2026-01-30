"""Data loading utilities."""

from typing import List, Dict, Any


def make_text(example: Dict[str, Any]) -> str:
    """Convert a dataset example to a text string for training."""
    parts = []
    for k in ["context", "question", "answer"]:
        if k in example and example[k] is not None:
            parts.append(str(example[k]))
    return "\n".join(parts)


class CausalLMCollator:
    """Collator for causal language modeling that handles tokenization and padding."""
    
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [make_text(ex) for ex in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc
