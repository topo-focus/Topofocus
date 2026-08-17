from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class SeparatorStyle(Enum):
    TWO = "two"


@dataclass
class Conversation:
    system: str = ""
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    sep: str = "###"
    sep2: str = "\n"
    version: str = "v1"
    sep_style: SeparatorStyle = SeparatorStyle.TWO
    messages: List[Tuple[str, str]] = field(default_factory=list)

    def copy(self) -> "Conversation":
        return copy.deepcopy(self)

    def append_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))

    def get_prompt(self) -> str:
        chunks: List[str] = []
        for role, text in self.messages:
            chunks.append(f"{self.sep}{role}: {text}{self.sep2}")
        return "".join(chunks)


conv_templates: Dict[str, Conversation] = {
    "vicuna_v1": Conversation(version="v1", roles=("human", "gpt")),
    "v1": Conversation(version="v1", roles=("human", "gpt")),
}

default_conversation: Conversation = conv_templates["vicuna_v1"].copy()
