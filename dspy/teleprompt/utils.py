import inspect
import logging
import os
import random
import sys

try:
    from IPython.core.magics.code import extract_symbols
except ImportError:
    # Won't be able to read code from jupyter notebooks
    extract_symbols = None

import dspy
from dspy.teleprompt.bootstrap import BootstrapFewShot, LabeledFewShot

"""
This file consists of helper functions for our variety of optimizers.
"""

### OPTIMIZER TRAINING UTILS ###

logger = logging.getLogger(__name__)


def create_minibatch(trainset, batch_size=50, rng=None):
    """Create a minibatch from the trainset."""

    # Ensure batch_size isn't larger than the size of the dataset
    batch_size = min(batch_size, len(trainset))

    # If no RNG is provided, fall back to the global random instance
    rng = rng or random

    # Randomly sample indices for the mini-batch using the provided rng
    sampled_indices = rng.sample(range(len(trainset)), batch_size)

    # Create the mini-batch using the sampled indices
    minibatch = [trainset[i] for i in sampled_indices]

    return minibatch


def eval_candidate_program(batch_size, trainset, candidate_program, evaluate, rng=None):
    """Evaluate a candidate program on the trainset, using the specified batch size."""

    try:
        # Evaluate on the full trainset
        if batch_size >= len(trainset):
            return evaluate(candidate_program, devset=trainset, callback_metadata={"metric_key": "eval_full"})
        # Or evaluate on a minibatch
        else:
            return evaluate(
                candidate_program,
                devset=create_minibatch(trainset, batch_size, rng),
                callback_metadata={"metric_key": "eval_minibatch"}
            )
    except Exception:
        logger.error("An exception occurred during evaluation", exc_info=True)
        # TODO: Handle this better, as -ve scores are possible
        return dspy.Prediction(score=0.0, results=[])


def get_program_with_highest_avg_score(param_score_dict, fully_evaled_param_combos):
    """Used as a helper function for bayesian + minibatching optimizers. Returns the program with the highest average score from the batches evaluated so far."""

    # Calculate the mean for each combination of categorical parameters, based on past trials
    results = []
    for key, values in param_score_dict.items():
        scores = [v[0] for v in values]
        mean = sum(scores) / len(scores)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean, program, params))

    # Sort results by the mean
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find the combination with the highest mean, skip fully evaluated ones
    for combination in sorted_results:
        key, mean, program, params = combination

        if key in fully_evaled_param_combos:
            continue

        return program, mean, key, params

    raise ValueError("No valid program found in param_score_dict")


### LOGGING UTILS ###


def print_full_program(program):
    """Print out the program's instructions & prefixes for each module."""
    for i, predictor in enumerate(program.predictors()):
        print(f"Predictor {i}")
        print(f"i: {get_signature(predictor).instructions}")
        *_, last_field = get_signature(predictor).fields.values()
        print(f"p: {last_field.json_schema_extra['prefix']}")
    print("\n")


def save_candidate_program(program, log_dir, trial_num, note=None):
    """Save the candidate program to the log directory."""

    if log_dir is None:
        return None

    # Ensure the directory exists
    eval_programs_dir = os.path.join(log_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)

    # Define the save path for the program
    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}.json")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}.json")

    # Save the program
    program.save(save_path)

    return save_path


### OTHER UTILS ###


def get_prompt_model(prompt_model):
    if prompt_model:
        return prompt_model
    else:
        return dspy.settings.lm


def get_signature(predictor):
    assert hasattr(predictor, "signature")
    return predictor.signature


def set_signature(predictor, updated_signature):
    assert hasattr(predictor, "signature")
    predictor.signature = updated_signature


def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets,
    trainset,
    max_labeled_demos,
    max_bootstrapped_demos,
    metric,
    teacher_settings,
    max_errors=None,
    max_rounds=1,
    labeled_sample=True,
    min_num_samples=1,
    metric_threshold=None,
    teacher=None,
    include_non_bootstrapped=True,
    seed=0,
    rng=None,
):
    """
    This function is copied from random_search.py, and creates fewshot examples in the same way that random search does.
    This allows us to take advantage of using the same fewshot examples when we use the same random seed in our optimizers.
    """
    max_errors = dspy.settings.max_errors if max_errors is None else max_errors
    demo_candidates = {}

    # Account for confusing way this is set up, where we add in 3 more candidate sets to the N specified
    num_candidate_sets -= 3

    # Initialize demo_candidates dictionary
    for i, _ in enumerate(student.predictors()):
        demo_candidates[i] = []

    rng = rng or random.Random(seed)

    # Go through and create each candidate set
    for seed in range(-3, num_candidate_sets):
        print(f"Bootstrapping set {seed + 4}/{num_candidate_sets + 3}")

        trainset_copy = list(trainset)

        if seed == -3 and include_non_bootstrapped:
            # zero-shot
            program2 = student.reset_copy()

        elif seed == -2 and max_labeled_demos > 0 and include_non_bootstrapped:
            # labels only
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = teleprompter.compile(
                student,
                trainset=trainset_copy,
                sample=labeled_sample,
            )

        elif seed == -1:
            # unshuffled few-shot
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = program.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            # shuffled few-shot
            rng.shuffle(trainset_copy)
            size = rng.randint(min_num_samples, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )

            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
            )

        for i, _ in enumerate(student.predictors()):
            demo_candidates[i].append(program2.predictors()[i].demos)

    return demo_candidates


def old_getfile(object):
    """Work out which source or compiled file an object was defined in."""
    if inspect.ismodule(object):
        if getattr(object, "__file__", None):
            return object.__file__
        raise TypeError(f"{object!r} is a built-in module")
    if inspect.isclass(object):
        if hasattr(object, "__module__"):
            module = sys.modules.get(object.__module__)
            if getattr(module, "__file__", None):
                return module.__file__
            if object.__module__ == "__main__":
                raise OSError("source code not available")
        raise TypeError(f"{object!r} is a built-in class")
    if inspect.ismethod(object):
        object = object.__func__
    if inspect.isfunction(object):
        object = object.__code__
    if inspect.istraceback(object):
        object = object.tb_frame
    if inspect.isframe(object):
        object = object.f_code
    if inspect.iscode(object):
        return object.co_filename
    raise TypeError(
        f"module, class, method, function, traceback, frame, or code object was expected, got {type(object).__name__}"
    )


def new_getfile(object):
    if not inspect.isclass(object):
        return old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for _, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + "." + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    raise TypeError(f"Source for {object!r} not found")


inspect.getfile = new_getfile
