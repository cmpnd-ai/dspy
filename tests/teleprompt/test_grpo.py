import dspy
from dspy.teleprompt.grpo import GRPO
from dspy.utils.dummies import DummyLM


def test_grpo_dataset_shuffler():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=3,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(4):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 3
        assert set(trainset_instances[-1]) == set(dataset)


def test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=2,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(15):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 2

    from collections import Counter

    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)

    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10


def test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=5,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(6):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 5

    from collections import Counter

    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)

    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10


def test_grpo_keeps_same_signature_components_and_repeated_calls_separate(monkeypatch):
    class Program(dspy.Module):
        def __init__(self):
            signature = dspy.Signature("q -> a")
            self.first = dspy.Predict(signature)
            self.second = dspy.Predict(signature)

        def forward(self, q):
            self.second(q="second")
            self.first(q="first-1")
            return self.first(q="first-2")

    submitted = []

    class Job:
        def get_status(self):
            return {"pending_batch_ids": [0, 1, 2]}

        def step(self, train_data, train_data_format):
            submitted.extend(train_data)

        def terminate(self):
            pass

    labels = ["first-1", "first-2", "second"]
    lm = DummyLM({label: {"a": label} for label in labels})
    monkeypatch.setattr(lm, "reinforce", lambda **kwargs: Job(), raising=False)  # provider boundary only
    student = Program()
    student.set_lm(lm)
    GRPO(
        metric=lambda gold, pred, trace=None: 0.75, exclude_demos=True,
        num_threads=1, num_train_steps=1, num_rollouts_per_grpo_step=2,
        adapter=dspy.ChatAdapter(),
    ).compile(student, trainset=[dspy.Example(q="root").with_inputs("q")])

    observed = []
    for batch in submitted:
        group = batch["group"]
        assert len(group) == 2
        names = [next(label for label in labels if label in item["messages"][-1]["content"]) for item in group]
        assert names[0] == names[1]
        assert all(names[0] in item["completion"]["content"] and item["reward"] == 0.75 for item in group)
        observed.append(names[0])
    assert sorted(observed) == labels


if __name__ == "__main__":
    test_grpo_dataset_shuffler()
    test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset()
    test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset()
    print("All tests passed!")
