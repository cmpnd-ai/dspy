import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import orjson

import dspy
from dspy.utils.caching import DSPY_CACHEDIR

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter


class TrainingStatus(str, Enum):
    not_started = "not_started"
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainDataFormat(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    GRPO_CHAT = "grpo_chat"


class Message(TypedDict):
    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class MessageAssistant(TypedDict):
    role: Literal["assistant"]
    content: str


class GRPOChatData(TypedDict):
    messages: list[Message]
    completion: MessageAssistant
    reward: float


class GRPOGroup(TypedDict):
    batch_id: int | None
    group: list[GRPOChatData]

class GRPOStatus(TypedDict):
    job_id: str
    status: str | None = None
    current_model: str
    checkpoints: dict[str, str]
    last_checkpoint: str | None = None
    pending_batch_ids: list[int] = []


def infer_data_format(adapter: "Adapter") -> str:
    if isinstance(adapter, dspy.ChatAdapter):
        return TrainDataFormat.CHAT
    raise ValueError(f"Could not infer the data format for: {adapter}")


def get_finetune_directory() -> str:
    default_finetunedir = os.path.join(DSPY_CACHEDIR, "finetune")
    finetune_dir = os.environ.get("DSPY_FINETUNEDIR") or default_finetunedir
    finetune_dir = os.path.abspath(finetune_dir)
    os.makedirs(finetune_dir, exist_ok=True)
    return finetune_dir


def save_data(
    data: list[dict[str, Any]],
) -> str:
    from dspy.utils.hasher import Hasher

    # Assign a unique name to the file based on the data hash
    hash = Hasher.hash(data)
    file_name = f"{hash}.jsonl"

    finetune_dir = get_finetune_directory()
    file_path = os.path.join(finetune_dir, file_name)
    file_path = os.path.abspath(file_path)
    with open(file_path, "wb") as f:
        for item in data:
            f.write(orjson.dumps(item) + b"\n")
    return file_path
