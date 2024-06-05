from typing import Dict, List, Any
from pydantic import BaseModel

class PreferencePair(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    margin: float = 1.0

class PreferenceDataCollator:
    def __init__(self, pad_token_id: int = 0, max_length: int = 1024):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def collate_batch(self, batch: List[PreferencePair]) -> Dict[str, Any]:
        return {
            "prompts": [item.prompt for item in batch],
            "chosen_responses": [item.chosen for item in batch],
            "rejected_responses": [item.rejected for item in batch],
            "batch_size": len(batch)
        }

# FlashAttention-2 alignment
