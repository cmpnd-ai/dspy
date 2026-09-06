from dspy.teleprompt.grpo import GRPO


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


def test_grpo_dataset_shuffler_no_padding_when_divisible():
    dataset = [1, 2, 3, 4, 5, 6]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=3,
        exclude_demos=True,
    )

    grpo.select_training_sample_and_update_shuffled_trainset(dataset, 0)

    assert len(grpo.shuffled_trainset_ids) == len(dataset)
    assert len(grpo.shuffled_trainset_ids) % grpo.num_dspy_examples_per_grpo_step == 0
    assert grpo.id_freqs.total() == len(dataset)
    assert all(v == 1 for v in grpo.id_freqs.values())


def test_grpo_dataset_shuffler_across_epoch_boundary_divisible():
    from collections import Counter

    dataset = [1, 2, 3, 4, 5, 6]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=3,
        exclude_demos=True,
    )

    batches, epochs = [], []
    for i in range(6):
        batch = grpo.select_training_sample_and_update_shuffled_trainset(dataset, i)
        batches.append(batch)
        epochs.append(grpo.epoch)

    assert epochs == [0, 0, 1, 1, 2, 2]
    assert len(grpo.shuffled_trainset_ids) == len(dataset)

    epoch_to_ids = {}
    for batch, epoch in zip(batches, epochs, strict=True):
        epoch_to_ids.setdefault(epoch, []).extend(batch)

    for epoch, ids in epoch_to_ids.items():
        assert len(ids) == len(set(ids)), f"epoch {epoch} contains duplicate ids: {Counter(ids)}"
        assert set(ids) == set(dataset), f"epoch {epoch} contains unexpected ids: {set(ids)}"

    assert Counter(epoch_to_ids[0]) == Counter(dataset)
    assert Counter(epoch_to_ids[1]) == Counter(dataset)


if __name__ == "__main__":
    test_grpo_dataset_shuffler()
    test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset()
    test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset()
    test_grpo_dataset_shuffler_no_padding_when_divisible()
    test_grpo_dataset_shuffler_across_epoch_boundary_divisible()
    print("All tests passed!")
