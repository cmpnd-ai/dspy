from __future__ import annotations

import random
import uuid
from collections.abc import Iterable
from typing import Any

from dspy import Example


class Dataset:
    def __init__(
        self,
        train_seed: int = 0,
        train_size: int | None = None,
        eval_seed: int = 0,
        dev_size: int | None = None,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        self.train_size = train_size
        self.train_seed = train_seed
        self.dev_size = dev_size
        self.dev_seed = eval_seed
        self.test_size = test_size
        self.test_seed = eval_seed
        self.input_keys = input_keys or []

        self.do_shuffle = True

        self.name = self.__class__.__name__

    def reset_seeds(
        self,
        train_seed: int | None = None,
        train_size: int | None = None,
        eval_seed: int | None = None,
        dev_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        self.train_size = train_size if train_size is not None else self.train_size
        self.train_seed = train_seed if train_seed is not None else self.train_seed
        self.dev_size = dev_size if dev_size is not None else self.dev_size
        self.dev_seed = eval_seed if eval_seed is not None else self.dev_seed
        self.test_size = test_size if test_size is not None else self.test_size
        self.test_seed = eval_seed if eval_seed is not None else self.test_seed

        if hasattr(self, "_train_"):
            del self._train_

        if hasattr(self, "_dev_"):
            del self._dev_

        if hasattr(self, "_test_"):
            del self._test_

    @property
    def train(self) -> list[Example]:
        if not hasattr(self, "_train_"):
            self._train_ = self._shuffle_and_sample("train", self._train, self.train_size, self.train_seed)

        return self._train_

    @property
    def dev(self) -> list[Example]:
        if not hasattr(self, "_dev_"):
            self._dev_ = self._shuffle_and_sample("dev", self._dev, self.dev_size, self.dev_seed)

        return self._dev_

    @property
    def test(self) -> list[Example]:
        if not hasattr(self, "_test_"):
            self._test_ = self._shuffle_and_sample("test", self._test, self.test_size, self.test_seed)

        return self._test_

    def _shuffle_and_sample(
        self, split: str, data: Iterable[dict[str, Any]], size: int | None, seed: int = 0
    ) -> list[Example]:
        data_list = list(data)

        # Shuffle the data irrespective of the requested size.
        base_rng = random.Random(seed)

        if self.do_shuffle:
            base_rng.shuffle(data_list)

        data_list = data_list[:size]
        output: list[Example] = []

        for example in data_list:
            example_obj = Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split)
            if self.input_keys:
                example_obj = example_obj.with_inputs(*self.input_keys)
            output.append(example_obj)
        # TODO: NOTE: Ideally we use these uuids for dedup internally, for demos and internal train/val splits.
        # Now, some tasks (like convQA and Colors) have overlapping examples. Here, we should allow the user to give us
        # a uuid field that would respect this in some way. This means that we need a more refined concept that
        # uuid (each example is unique) and more like a group_uuid.

        return output
