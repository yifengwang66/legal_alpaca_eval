import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

import fire
import pandas as pd

from alpaca_eval import analyze, annotators, constants, decoders, metrics, utils
from alpaca_eval.types import AnyData, AnyPath
import json

CUR_DIR = Path(__file__).parent
DEFAULT_CONFIGS = "alpaca_eval_gpt4"

__all__ = [
    "evaluate",
    "evaluate_from_model",
    "analyze_evaluators",
    "make_leaderboard",
    "multiwise_evaluate",
    "make_multi_compare_leaderboard"
]


def evaluate(
        model_outputs: Optional[Union[AnyPath, AnyData, Callable]] = None,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        name: Optional[str] = None,
        output_path: Optional[Union[AnyPath, str]] = "auto",
        precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
        is_overwrite_leaderboard: bool = False,
        leaderboard_mode_to_print: Optional[str] = "minimal",
        current_leaderboard_mode: str = "community",
        is_return_instead_of_print: bool = False,
        fn_metric: Union[str, callable] = "pairwise_to_winrate",
        sort_by: str = "win_rate",
        is_cache_leaderboard: Optional[bool] = None,
        max_instances: Optional[int] = None,
        annotation_kwargs: Optional[dict[str, Any]] = None,
        Annotator=annotators.PairwiseAnnotator,
        **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
        003 outputs on the AlpacaEval set.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not
        we use "Current model".

    output_path : path, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if the model is already in it.

    leaderboard_mode_to_print : {"minimal", "verified", "community", None}, optional
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case
        it will filter the leaderboard by this mode. If None keeps all.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard for the current method.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    fn_metric : str or callable, optional
        The function or function name in `metrics.py` that will be used to convert preference to metrics. The function
        should take a sequence of preferences (0 for draw, 1 for base win, 2 when the model to compare wins) and return
        a dictionary of metrics and the key by which to sort the leaderboard.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    is_cache_leaderboard : bool, optional
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances
        not None. A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the
        previously saved leaderboard at `<output_path>/leaderboard.csv`.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    Annotator : class, optional
        The annotator class to use.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    if (
            isinstance(current_leaderboard_mode, str)
            and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    leaderboard, precomputed_leaderboard = utils.get_precomputed_leaderboard(
        precomputed_leaderboard, reference_outputs, annotators_config
    )
    annotations = None

    if model_outputs is not None:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        name = utils.get_generator_name(name, model_outputs)

        if (name not in leaderboard) or is_overwrite_leaderboard:
            logging.info(f"Evaluating the {name} outputs.")

            if max_instances is not None:
                model_outputs = model_outputs[:max_instances]
                reference_outputs = reference_outputs[:max_instances]

            annotator = Annotator(annotators_config=annotators_config, **annotator_kwargs)
            annotations = annotator.annotate_head2head(
                outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
            )

            print("evaluate annotations:", annotations)
            if isinstance(fn_metric, str):
                fn_metric = getattr(metrics, fn_metric)

            leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotations])
            leaderboard[name]["mode"] = current_leaderboard_mode
            leaderboard[name]["avg_length"] = int(model_outputs["output"].str.len().mean())
        else:
            logging.info(f"Skipping evaluation of {name} as it is already in the precomputed leaderboard.")

    output_path = utils.get_output_path(output_path, model_outputs, name)

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(by=sort_by, ascending=False)
    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), ["win_rate", "standard_error"])
    ]

    if output_path is not None:
        logging.info(f"Saving all results to {output_path}")
        df_leaderboard.to_csv(output_path / "leaderboard.csv")
        if annotations is not None:
            utils.convert_to_dataframe(annotations).to_json(
                output_path / "annotations.json", orient="records", indent=2
            )

    if is_cache_leaderboard is None:
        is_cache_leaderboard = max_instances is None

    if is_cache_leaderboard:
        if isinstance(precomputed_leaderboard, AnyPath):
            logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
            df_leaderboard.to_csv(precomputed_leaderboard)
        else:
            logging.info(
                f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
                f"path but {type(precomputed_leaderboard)}."
            )

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        utils.print_leaderboard(
            df_leaderboard,
            leaderboard_mode_to_print,
            current_name=name,
            cols_to_print=["win_rate", "standard_error", "n_total", "avg_length"],  #
        )


def evaluate_from_model(
        model_configs: Union[AnyPath, dict],
        reference_model_configs: Optional[Union[AnyPath, dict]] = None,
        evaluation_dataset: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        output_path: AnyPath = "auto",
        max_instances: int = None,
        is_strip_output: bool = True,
        is_load_outputs: bool = True,
        chunksize: int = 64,
        **kwargs,
):
    """Evaluate a model from HuggingFace or an API provider. This is a wrapper around `evaluate` which includes
    generating from
    a desired model.

    Parameters
    ----------
    model_configs : path or dict
        A dictionary or path (relative to `models_configs`) to a yaml file containing the configuration of the model to
        decode from. If a directory,we search for 'configs.yaml' in it. The keys in the first dictionary should be the
        generator's name, and the value should be a dictionary of the generator's configuration which should have the
        following keys:
        - prompt_template (str): a prompt template or path to one. Each template should contain placeholders for
        keys in the data dictionary, typically {instruction} and {output}.
        - fn_completions (str): function in `alpaca_farm.decoders` for completions. Needs to accept as first argument
            `prompts` which is a list of string.
        - completions_kwargs (dict): kwargs for fn_completions. E.g. model_name, max_tokens, temperature...

    reference_model_configs : path or dict, optional
        Same as in `model_configs` but for the reference model. If None, we use the default Davinci003 outputs.

    evaluation_dataset : path or callable, optional
        Path to the evaluation dataset or a function that returns a dataframe. If None, we use the default evaluation

    annotators_config : path or dict, optional
        Path to the annotators configuration or a dictionary. If None, we use the default annotators configuration.

    output_path : path, optional
        Path to save the generations, annotations and leaderboard. If auto saves at `results/<model_name>`

    max_instances : int, optional
        Maximum number of instances to generate and evaluate. If None, we evaluate all instances.

    is_strip_output : bool, optional
        Whether to strip trailing and leading whitespaces from the outputs.

    is_load_outputs : bool, optional
        Whether to try to load outputs from the output path. If True and outputs exist we only generate outputs for
        instructions that don't have outputs yet.

    chunksize : int, optional
        Number of instances to generate before saving. If None, we save after all generations.

    kwargs:
        Other kwargs to `evaluate`
    """
    df_dataset = utils.load_or_convert_to_dataframe(evaluation_dataset)

    if chunksize is not None and not is_load_outputs:
        logging.info("`is_load_outputs` has to be true to use chunksize. Setting it to True.")
        is_load_outputs = True

    if chunksize is not None and max_instances is not None:
        logging.info("cannot use `chunksize` with max_instances. Setting `chunksize` to None.")
        chunksize = None

    model_configs = utils.load_configs(model_configs, relative_to=constants.MODELS_CONFIG_DIR)
    if reference_model_configs is not None:
        reference_model_configs = utils.load_configs(reference_model_configs, relative_to=constants.MODELS_CONFIG_DIR)

    if output_path == "auto":
        output_path = Path("results") / list(model_configs.keys())[0]
    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    def get_completions(configs, df: pd.DataFrame, old_output_path: Optional[Path] = None):
        columns_to_keep = ["dataset", "instruction", "output", "generator"]
        columns_to_keep = [c for c in columns_to_keep if c in df.columns]
        curr_outputs = df[columns_to_keep].copy()
        is_loading_old_outputs = old_output_path is not None and old_output_path.exists()
        assert len(configs) == 1
        generator = list(configs.keys())[0]
        configs = list(configs.values())[0]

        if is_loading_old_outputs:
            logging.info(f"Loading outputs from {old_output_path}")
            old_outputs = utils.load_or_convert_to_dataframe(old_output_path)
            # select only rows in curr_outputs that have "instruction" that are not in old_outputs
            idx_found_old_outputs = curr_outputs["instruction"].isin(old_outputs["instruction"])
            curr_outputs = curr_outputs[~idx_found_old_outputs]
            assert (old_outputs["generator"] == generator).all()

        if max_instances is not None:
            curr_outputs = curr_outputs.iloc[:max_instances]

        if len(curr_outputs) > 0:
            prompts, _ = utils.make_prompts(
                curr_outputs,
                template=utils.read_or_return(constants.MODELS_CONFIG_DIR / configs["prompt_template"]),
            )
            fn_completions = decoders.get_fn_completions(configs["fn_completions"])
            completions = fn_completions(prompts=prompts, **configs["completions_kwargs"])["completions"]
            if is_strip_output:
                completions = [c.strip() for c in completions]
            curr_outputs["output"] = completions
            curr_outputs["generator"] = generator

        if is_loading_old_outputs:
            curr_outputs = pd.concat([old_outputs, curr_outputs], axis=0)

        return curr_outputs

    for df_chunk in utils.dataframe_chunk_generator(
            df_dataset, chunksize=chunksize, tqdm_desc="Chunking for generation"
    ):
        if is_load_outputs and output_path is not None:
            model_outputs = get_completions(
                model_configs, df=df_chunk, old_output_path=output_path / "model_outputs.json"
            )
        else:
            model_outputs = get_completions(model_configs, df=df_chunk)

        if reference_model_configs is None:
            if "output" not in df_chunk.columns:
                raise ValueError("evaluation_dataset should have a column 'output' containing references outputs")
            reference_outputs = df_dataset.copy()
        else:
            reference_outputs = get_completions(
                reference_model_configs,
                df=df_chunk,
                old_output_path=output_path / "reference_outputs.json",
            )

        if output_path is not None:
            model_outputs.to_json(output_path / "model_outputs.json", orient="records", indent=2)
            reference_outputs.to_json(output_path / "reference_outputs.json", orient="records", indent=2)

    if reference_model_configs is None:
        # using a default reference outputs => uses the right leaderboard
        if evaluation_dataset in [constants.ALPACAEVAL_REFERENCE_OUTPUTS]:
            reference_outputs = evaluation_dataset

    return evaluate(
        model_outputs=model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        output_path=output_path,
        max_instances=max_instances,
        **kwargs,
    )


def make_leaderboard(
        leaderboard_path: AnyPath,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        all_model_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        fn_add_to_leaderboard: Callable = "evaluate",
        leaderboard_mode: str = "verified",
        is_return_instead_of_print: bool = False,
        **kwargs,
):
    """Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

    Parameters
    ----------
    leaderboard_path : path
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
        append

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    all_model_outputs : path or data or callable, optional
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate
        those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate
        them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by
        default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the
        name of the current model.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By
        default,
        the reference outputs are the 003 outputs on AlpacaEval set.

    fn_add_to_leaderboard : callable or str, optional
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
        `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
        `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.

    leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    kwargs :
        Additional arguments to pass to `fn_add_to_leaderboard`.
    """
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
    if "generator" not in all_model_outputs.columns:
        raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")

    all_annotations = []
    # 根据模型类别，一类一类处理
    for model in all_model_outputs["generator"].unique():
        model_outputs = all_model_outputs[all_model_outputs["generator"] == model]
        df_leaderboard, annotations = fn_add_to_leaderboard(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annotators_config,
            name=model,
            precomputed_leaderboard=leaderboard_path,
            is_return_instead_of_print=True,
            current_leaderboard_mode=leaderboard_mode,
            **kwargs,
        )
        if annotations is not None:
            all_annotations += annotations
        df_leaderboard.to_csv(leaderboard_path)

    leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    df_leaderboard = pd.DataFrame(leaderboard)

    print('all_annotations:', all_annotations)
    if is_return_instead_of_print:
        return df_leaderboard, all_annotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode=None, cols_to_print=["win_rate", "standard_error", "n_total"]
        )


def multiwise_evaluate(
        model_outputs: Optional[Union[AnyPath, AnyData, Callable]] = None,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        name: Optional[str] = None,
        output_path: Optional[Union[AnyPath, str]] = "auto",
        precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
        is_overwrite_leaderboard: bool = False,
        leaderboard_mode_to_print: Optional[str] = "minimal",
        current_leaderboard_mode: str = "community",
        is_return_instead_of_print: bool = False,
        fn_metric: Union[str, callable] = "multiwise_to_avg_rank",
        sort_by: str = "avg_rank",
        is_cache_leaderboard: Optional[bool] = None,
        max_instances: Optional[int] = None,
        annotation_kwargs: Optional[dict[str, Any]] = None,
        Annotator=annotators.MultiwiseAnnotator,
        **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are the
        003 outputs on the AlpacaEval set.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not
        we use "Current model".

    output_path : path, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if the model is already in it.

    leaderboard_mode_to_print : {"minimal", "verified", "community", None}, optional
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case
        it will filter the leaderboard by this mode. If None keeps all.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard for the current method.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    fn_metric : str or callable, optional
        The function or function name in `metrics.py` that will be used to convert preference to metrics. The function
        should take a sequence of preferences (0 for draw, 1 for base win, 2 when the model to compare wins) and return
        a dictionary of metrics and the key by which to sort the leaderboard.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    is_cache_leaderboard : bool, optional
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances
        not None. A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the
        previously saved leaderboard at `<output_path>/leaderboard.csv`.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    Annotator : class, optional
        The annotator class to use.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    if (
            isinstance(current_leaderboard_mode, str)
            and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    annotations = None

    if model_outputs is not None:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)

        if max_instances is not None:
            model_outputs = model_outputs[:max_instances]
            reference_outputs = reference_outputs[:max_instances]

        annotator = Annotator(annotators_config=annotators_config, annotation_type=str, **annotator_kwargs)
        annotations = annotator.annotate_reference2multi(
            reference_outputs=reference_outputs, models_outputs=model_outputs, **annotation_kwargs
        )

        print("evaluate annotations:", annotations)
        extract_multi_compare_per_instruction(annotations)
        if isinstance(fn_metric, str):
            fn_metric = getattr(metrics, fn_metric)

        leaderboard = fn_metric(
            rank_obj_list=[json.loads(a["multi_rank_per_instruction"].replace("'", '"')) for a in annotations])

    print("leaderboard:", leaderboard)
    output_path = utils.get_output_path(output_path, model_outputs, name)

    df_leaderboard = leaderboard.sort_values(by=sort_by, ascending=True)
    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), ["avg_rank"])
    ]

    if output_path is not None:
        logging.info(f"Saving all results to {output_path}")
        df_leaderboard.to_csv(output_path / "leaderboard.csv")
        if annotations is not None:
            utils.convert_to_dataframe(annotations).to_json(
                output_path / "annotations.json", orient="records", indent=2
            )

    if is_cache_leaderboard is None:
        is_cache_leaderboard = max_instances is None

    if is_cache_leaderboard:
        if isinstance(precomputed_leaderboard, AnyPath):
            logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
            df_leaderboard.to_csv(precomputed_leaderboard)
        else:
            logging.info(
                f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
                f"path but {type(precomputed_leaderboard)}."
            )

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        utils.print_leaderboard(
            df_leaderboard,
            leaderboard_mode_to_print,
            current_name=name,
            cols_to_print=["win_rate", "standard_error", "n_total", "avg_length"],  #
        )


def make_multi_compare_leaderboard(
        leaderboard_path: AnyPath,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        all_model_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        fn_add_to_leaderboard: Callable = "multiwise_evaluate",
        leaderboard_mode: str = "verified",
        is_return_instead_of_print: bool = False,
        **kwargs,
):
    """Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.

    Parameters
    ----------
    leaderboard_path : path
        The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
        append

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    all_model_outputs : path or data or callable, optional
        The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate
        those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate
        them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by
        default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the
        name of the current model.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By
        default,
        the reference outputs are the 003 outputs on AlpacaEval set.

    fn_add_to_leaderboard : callable or str, optional
        The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
        `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
        `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.

    leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    kwargs :
        Additional arguments to pass to `fn_add_to_leaderboard`.
    """
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
    if "generator" not in all_model_outputs.columns:
        raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")

    all_annotations = []
    df_leaderboard, annotations = fn_add_to_leaderboard(
        model_outputs=all_model_outputs,
        reference_outputs=reference_outputs,
        annotators_config=annotators_config,
        precomputed_leaderboard=leaderboard_path,
        is_return_instead_of_print=True,
        current_leaderboard_mode=leaderboard_mode,
        **kwargs,
    )
    if annotations is not None:
        all_annotations += annotations
    df_leaderboard.to_csv(leaderboard_path)

    leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    df_leaderboard = pd.DataFrame(leaderboard)

    print('all_annotations:', all_annotations)
    if is_return_instead_of_print:
        return df_leaderboard, all_annotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode=None, cols_to_print=["model", "avg_rank"]
        )


# def evaluate_round_robin(
#         outer_model_outputs: Optional[Union[AnyPath, AnyData, Callable]] = None,
#         inner_model_outputs: Optional[Union[AnyPath, AnyData, Callable]] = None,
#         reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
#         annotators_config: AnyPath = DEFAULT_CONFIGS,
#         name: Optional[str] = None,
#         output_path: Optional[Union[AnyPath, str]] = "auto",
#         precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
#         is_overwrite_leaderboard: bool = False,
#         leaderboard_mode_to_print: Optional[str] = "minimal",
#         current_leaderboard_mode: str = "community",
#         is_return_instead_of_print: bool = False,
#         fn_metric: Union[str, callable] = "pairwise_to_winrate",
#         sort_by: str = "win_rate",
#         is_cache_leaderboard: Optional[bool] = None,
#         max_instances: Optional[int] = None,
#         annotation_kwargs: Optional[dict[str, Any]] = None,
#         Annotator=annotators.RoundRobinAnnotator,
#         **annotator_kwargs,
# ):
#     if (
#             isinstance(current_leaderboard_mode, str)
#             and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
#     ):
#         raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")
#
#     annotation_kwargs = annotation_kwargs or dict()
#
#     leaderboard, precomputed_leaderboard = utils.get_precomputed_leaderboard(
#         precomputed_leaderboard, reference_outputs, annotators_config
#     )
#     annotations = None
#
#     if outer_model_outputs is not None and inner_model_outputs is not None:
#         outer_model_outputs = utils.load_or_convert_to_dataframe(outer_model_outputs)
#         inner_model_outputs = utils.load_or_convert_to_dataframe(inner_model_outputs)
#         reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
#         outer_name = utils.get_generator_name(name, outer_model_outputs)
#         inner_name = utils.get_generator_name(name, inner_model_outputs)
#
#         if (name not in leaderboard) or is_overwrite_leaderboard:
#             logging.info(f"Evaluating the {outer_name} and {inner_name} outputs.")
#
#             if max_instances is not None:
#                 outer_outputs = outer_model_outputs[:max_instances]
#                 inner_outputs = inner_model_outputs[:max_instances]
#                 reference_outputs = reference_outputs[:max_instances]
#
#             annotator = Annotator(annotators_config=annotators_config, **annotator_kwargs)
#             annotations = annotator.annotate_round_robin(
#                 reference_output=reference_outputs ,outputs_1=outer_outputs, outputs_2=inner_outputs, **annotation_kwargs
#             )
#
#             print("evaluate annotations:", annotations)
#             if isinstance(fn_metric, str):
#                 fn_metric = getattr(metrics, fn_metric)
#
#             leaderboard[name] = fn_metric(preferences=[a["preference"] for a in annotations])
#             leaderboard[name]["mode"] = current_leaderboard_mode
#             leaderboard[name]["avg_length"] = int(model_outputs["output"].str.len().mean())
#         else:
#             logging.info(f"Skipping evaluation of {name} as it is already in the precomputed leaderboard.")
#
#     output_path = utils.get_output_path(output_path, model_outputs, name)
#
#     df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(by=sort_by, ascending=False)
#     df_leaderboard = df_leaderboard[
#         utils.prioritize_elements(list(df_leaderboard.columns), ["win_rate", "standard_error"])
#     ]
#
#     if output_path is not None:
#         logging.info(f"Saving all results to {output_path}")
#         df_leaderboard.to_csv(output_path / "leaderboard.csv")
#         if annotations is not None:
#             utils.convert_to_dataframe(annotations).to_json(
#                 output_path / "annotations.json", orient="records", indent=2
#             )
#
#     if is_cache_leaderboard is None:
#         is_cache_leaderboard = max_instances is None
#
#     if is_cache_leaderboard:
#         if isinstance(precomputed_leaderboard, AnyPath):
#             logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
#             df_leaderboard.to_csv(precomputed_leaderboard)
#         else:
#             logging.info(
#                 f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
#                 f"path but {type(precomputed_leaderboard)}."
#             )
#
#     if is_return_instead_of_print:
#         return df_leaderboard, annotations
#     else:
#         utils.print_leaderboard(
#             df_leaderboard,
#             leaderboard_mode_to_print,
#             current_name=name,
#             cols_to_print=["win_rate", "standard_error", "n_total", "avg_length"],  #
#         )
#
#
# def make_multi_compare_leaderboard_round_robin(
#         leaderboard_path: AnyPath,
#         annotators_config: AnyPath = DEFAULT_CONFIGS,
#         all_model_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
#         reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
#         fn_add_to_leaderboard: Callable = "evaluate_round_robin",
#         leaderboard_mode: str = "verified",
#         is_return_instead_of_print: bool = False,
#         **kwargs,
# ):
#     """Precompute and save an entire leaderboard for a given dataset / evaluator / set of models generations.
#
#     Parameters
#     ----------
#     leaderboard_path : path
#         The path to save the leaderboard to. The leaderboard will be saved as a csv file, if it already exists it will
#         append
#
#     annotators_config : path or list of dict, optional
#         The path the (or list of dict of) the annotator's config file.
#
#     all_model_outputs : path or data or callable, optional
#         The outputs of all models to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
#         datasets.Dataset) or a path to read those (json, csv, tsv potentially with globbing) or a function to generate
#         those. If the path contains a globbing pattern, we will read all files matching the pattern and concatenate
#         them. Each dictionary (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by
#         default `instruction` and `output` with optional `input`. It should also contain a column `generator` with the
#         name of the current model.
#
#     reference_outputs : path or data, optional
#         The outputs of the reference model. Same format as `all_model_outputs` but without needing `generator`. By
#         default,
#         the reference outputs are the 003 outputs on AlpacaEval set.
#
#     fn_add_to_leaderboard : callable or str, optional
#         The function to use to add a model to the leaderboard. If a string, it should be the name of a function in
#         `main.py`. The function should take the arguments: `model_outputs`, `annotators_config`, `name`,
#         `precomputed_leaderboard`, `is_return_instead_of_print`, `reference_outputs`.
#
#     leaderboard_mode : {"minimal", "verified", "community"}, optional
#         The mode of the leaderboard to save all new entries with.
#
#     is_return_instead_of_print : bool, optional
#         Whether to return the metrics instead of printing the results.
#
#     kwargs :
#         Additional arguments to pass to `fn_add_to_leaderboard`.
#     """
#     if isinstance(fn_add_to_leaderboard, str):
#         fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]
#
#     all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
#     if "generator" not in all_model_outputs.columns:
#         raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")
#
#     all_annotations = []
#     generator_list = all_model_outputs["generator"].unique()
#     # 根据模型类别，一类一类处理
#     for ind_outer, model in enumerate(generator_list):
#         outer_model_outputs = all_model_outputs[all_model_outputs["generator"] == model]
#         for ind_inner in range(ind_outer + 1, len(generator_list)):
#             inner_model_outputs = all_model_outputs[all_model_outputs["generator"] == generator_list[ind_inner]]
#             df_leaderboard, annotations = fn_add_to_leaderboard(
#                 outer_model_outputs=outer_model_outputs,
#                 inner_model_outputs=inner_model_outputs,
#                 reference_outputs=reference_outputs,
#                 annotators_config=annotators_config,
#                 name=model,
#                 precomputed_leaderboard=leaderboard_path,
#                 is_return_instead_of_print=True,
#                 current_leaderboard_mode=leaderboard_mode,
#                 **kwargs,
#             )
#             if annotations is not None:
#                 all_annotations += annotations
#             df_leaderboard.to_csv(leaderboard_path)
#
#         leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
#         df_leaderboard = pd.DataFrame(leaderboard)`
#
#     print('all_annotations:', all_annotations)
#     if is_return_instead_of_print:
#         return df_leaderboard, all_annotations
#     else:
#         utils.print_leaderboard(
#             df_leaderboard, leaderboard_mode=None, cols_to_print=["win_rate", "standard_error", "n_total"]
#         )


def analyze_evaluators(
        annotators_config: Optional[AnyPath] = DEFAULT_CONFIGS,
        Annotator=annotators.PairwiseAnnotator,
        analyzer_kwargs=None,
        precomputed_leaderboard: Optional[Union[AnyPath, AnyData]] = CUR_DIR
                                                                     / "leaderboards/evaluators/evaluators_leaderboard.csv",
        is_save_leaderboard: bool = False,
        is_return_instead_of_print: bool = False,
        is_overwrite_leaderboard: bool = False,
        max_instances: Optional[int] = None,
        is_single_annotator: bool = False,
        leaderboard_mode_to_print: str = "minimal",
        current_leaderboard_mode: str = "minimal",
):
    """Analyze an evaluator and populates the evaluators leaderboard (agreement with human, speed, price,...).

    Parameters
    ----------
    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file.

    Annotator : class, optional
        The annotator class to use.

    analyzer_kwargs : dict, optional
        Additional arguments to pass to the analyzer.

    precomputed_leaderboard : path or data, optional
        The precomputed (meta)leaderboard of annotators or a path to it (json, csv, or tsv).

    is_save_leaderboard : bool, optional
        Whether to save the leaderboard (ie analyzed results).

    is_return_instead_of_print : bool, optional
        Whether to return the leaderboard (ie analyzed results). If True, it will not print the results.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if it already exists.

    max_instances : int, optional
        The maximum number of instances to analyze.

    is_single_annotator : bool, optional
        Whether to analyze a single annotator. If True, will not be able to estimate the annotator's bias.

    leaderboard_mode_to_print : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to print.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard to save all new entries with.
    """

    leaderboard = dict()
    if precomputed_leaderboard is not None:
        try:
            leaderboard = utils.load_or_convert_to_dataframe(precomputed_leaderboard).to_dict(orient="index")
        except FileNotFoundError:
            logging.warning(
                f"Could not find precomputed leaderboard at {precomputed_leaderboard}. Starting from " f"scratch."
            )

    analyzer_kwargs = analyzer_kwargs or {}

    all_crossannotations = dict()
    if annotators_config is not None:
        key = annotators_config.replace("/", "_").replace("_configs.yaml", "")
        if key not in leaderboard or is_overwrite_leaderboard:
            analyzer = analyze.Analyzer(**analyzer_kwargs)

            if key == "humans":
                df_crossannotations = analyzer.df_gold_crossannotations
            elif key == "longest":
                df_crossannotations = analyze._get_longest_predictor(analyzer.df_gold_crossannotations)
            else:
                df_crossannotations = analyze.get_crossannotations(
                    analyzer=analyzer,
                    Annotator=Annotator,
                    max_instances=max_instances,
                    annotators_config=annotators_config,
                    is_single_annotator=is_single_annotator,
                )

            leaderboard[key] = analyze.get_metrics_evaluator(analyzer, df_crossannotations, evaluator_name=key)
            leaderboard[key]["mode"] = current_leaderboard_mode
            all_crossannotations[key] = df_crossannotations

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(
        by="Human agreement [%]", ascending=False
    )

    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), constants.EVALUATORS_LEADERBOARD_COLS_TO_PRIORITIZE)
    ]

    if is_save_leaderboard:
        df_leaderboard.to_csv(precomputed_leaderboard)

    if is_return_instead_of_print:
        return df_leaderboard, all_crossannotations
    else:
        utils.print_leaderboard(
            df_leaderboard, leaderboard_mode_to_print, cols_to_print=constants.EVALUATORS_LEADERBOARD_COLS_TO_PRINT
        )


ALL_FUNCTIONS = {
    "evaluate": evaluate,
    "evaluate_from_model": evaluate_from_model,
    "make_leaderboard": make_leaderboard,
    "analyze_evaluators": analyze_evaluators,
    "make_multi_compare_leaderboard": make_multi_compare_leaderboard,
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


# 对于每个模型下的每个题目提取评测结果, preference 0(平局)/1(模型答案不行)/2(模型答案很行)
def extract_per_instructions_preference():
    preference_result = ['平局', '标答胜', '模型胜']
    df = pd.read_json('legal_reference_outputs.json')
    new_column_names = {
        "instruction": "题目",
        "output": "标准答案",
    }
    df.rename(columns=new_column_names, inplace=True)

    questions = df["题目"]
    question_dic = {}
    for i, question in enumerate(questions):
        question_dic[question] = i

    model_name_list = ['360', 'claude', 'glm', 'gpt_3', 'minimax', 'wenxinyiyan', 'xunfei', '百川', '通义']
    for model_name in model_name_list:
        with open(f'results/{model_name}/annotations.json', 'r') as f:
            results = json.load(f)
        model_answer = [None] * len(results)
        model_preference = [None] * len(results)
        for result in results:
            question_index = question_dic[result['instruction']]
            model_answer[question_index] = result['output_2']
            model_preference[question_index] = preference_result[int(result['preference'] or 1)]
        df[model_name] = model_answer
        df[f"{model_name}_result"] = model_preference

    df.to_csv('legal_model_per_instruction_result.csv', index=False, encoding='utf-8-sig')


def extract_multi_compare_per_instruction(annotations):
    all_df = None
    for annotation in annotations:
        instruction = annotation["instruction"]
        price = annotation["price_per_example"]
        time = annotation["time_per_example"]
        rank = json.loads(annotation["multi_rank_per_instruction"].replace("'", '"'))
        df = pd.DataFrame([
            {
                "instruction": instruction,
                "reference_output": annotation["reference_output"]
            }
        ])
        for per_rank in rank:
            df[f"{per_rank['model']}_output"] = annotation[per_rank['model']]
            df[f"{per_rank['model']}_rank"] = [per_rank['rank']]
        df["price"] = [price]
        df["time"] = [time]
        if all_df is None:
            all_df = df
            continue
        all_df = pd.concat([all_df, df])
    price_sum = all_df["price"].sum()
    time_sum = all_df["time"].sum()
    all_df = all_df.assign(price_sum=price_sum).assign(time_sum=time_sum)
    all_df.to_csv("legal_model_per_instruction_result.csv", encoding="utf-8-sig")


def purify_dataset():
    with open('legal_model_outputs.json', 'r') as f:
        model_outputs = json.load(f)

    with open('legal_reference_outputs.json', 'r') as f:
        reference_outputs = json.load(f)

    model_outputs = [model_output for model_output in model_outputs if model_output["generator"] != "Peking_chatlaw"]
    reference_outputs = [reference_output for reference_output in reference_outputs if reference_output["output"] != ""]

    with open('legal_model_outputs.json', 'w') as f:
        json.dump(model_outputs, f)

    with open('legal_reference_outputs.json', 'w') as f:
        json.dump(reference_outputs, f)


def generate_test_dataset():
    with open('legal_model_outputs.json', 'r') as f:
        model_outputs = json.load(f)

    with open('legal_reference_outputs.json', 'r') as f:
        test_reference_outputs = json.load(f)[:2]

    instruction_list = [reference_output['instruction'] for reference_output in test_reference_outputs]
    test_model_outputs = [model_output for model_output in model_outputs if
                          model_output['instruction'] in instruction_list]

    print('test_reference_output:', test_reference_outputs)
    print('test_model_outputs:', test_model_outputs)

    with open('legal_model_outputs_test.json', 'w') as f:
        json.dump(test_model_outputs, f)
    with open('legal_reference_outputs_test.json', 'w') as f:
        json.dump(test_reference_outputs, f)


def a_test_func():
    data = [
        [
            {
                "model": "model_1",
                "rank": 1,
                "price": 3,
            },
            {
                "model": "model_2",
                "rank": 2,
                "price": 3,
            },
            {
                "model": "model_3",
                "rank": 3,
                "price": 3,
            }
        ],
        [
            {
                "model": "model_1",
                "rank": 2,
                "price": 3,
            },
            {
                "model": "model_2",
                "rank": 1,
                "price": 3,
            },
            {
                "model": "model_3",
                "rank": 3,
                "price": 3,
            },
        ],
        [
            {
                "model": "model_1",
                "rank": 1,
                "price": 3,
            },
            {
                "model": "model_2",
                "rank": 3,
                "price": 3,
            },
            {
                "model": "model_3",
                "rank": 2,
                "price": 3,
            },
        ],
        [
            {
                "model": "model_1",
                "rank": 2,
                "price": 3,
            },
            {
                "model": "model_2",
                "rank": 3,
                "price": 3,
            },
            {
                "model": "model_3",
                "rank": 1,
                "price": 3,
            },
        ],
    ]
    df = pd.DataFrame([item for sublist in data for item in sublist])
    # 计算每个model的平均rank
    average_rank = df.groupby('model')['rank'].mean().rename("avg_rank")

    # 计算每个model获得的每个排名的次数
    rank_counts = df.groupby(['model', 'rank']).size().unstack(fill_value=0)

    leaderboard = pd.merge(
        average_rank,
        rank_counts,
        on="model"
    )

    rank_sum = df["rank"].sum()
    price_sum = df["price"].sum()
    df = df.assign(rank_sum=rank_sum).assign(price_sum=price_sum)
    print(df)

    annotations = [{'instruction': '意思表示（总则编）：民法中“意思表示”的含义？',
                    'reference_output': '向外部表明意欲发生一定私法上效果的意思的行为。',
                    '360': '在民法中，“意思表示”是指一方通过一定的方式表达其真实意愿的意思，另一方据此产生法律效果的行为。具体而言，意思表示必须包含以下要素: \n1. 表示人具有真实的意愿。 \n2. 表示人有能力表达自己的意愿。 \n3. 表示人是自愿的，没有受到任何强制或欺骗的影响。 \n4. 表示的内容是真实的。 \n5. 表示的内容是具体的和明确的。 \n6. 表示人希望以意思表示的形式与他人建立法律关系。 \n总之，意思表示是当事人之间建立法律关系的重要方式，它必须真实、自愿、明确，才能产生法律效果。',
                    'baichuan': '意思表示是指向外部表明意愿的行为，包括明示(直接表达)和默示(间接表达)两种方式。以下是一些关于意思表示的基本要素：\n\n1. 主体资格：发出意思表示的人必须具有相应的行为能力，即具备完全民事行为能力。\n\n2. 意识因素：意思表示必须是真实的、自愿的，而不是被迫的或者是无意识的。\n\n3. 明确性：意思表示必须清晰、准确地传达了发出的意图，否则可能会导致歧义或误解。\n\n4. 合法性：意思表示不得违法，例如不得欺诈、胁迫或恶意串通。\n\n5. 自决性：意思表示必须是自己做出的，而不是被别人操纵或控制的。\n\n6. 可撤回性：如果意思表示还没有到达相对人，那么可以撤回;一旦到达相对人，则无法撤回。\n\n7. 可撤销性：如果意思表示存在瑕疵，如重大误解、显失公平等，那么可以在一定期限内申请撤销。\n\n8. 拘束力：一旦意思表示完成并到达相对人，就会产生一定的法律效果，对双方产生约束力。',
                    'claude': ' 民法中"意思表示"是指表明意思的行为,可以是言语、书面或其他行为。它是民事法律行为的必要组成部分之一。\n\n意思表示需要符合如下条件:\n\n1. 表明意思的意思。行为人必须有表明某种意思的主观意思。\n\n2. 他人能够了解。行为人表明的意思,必须以他人能够理解的方式表达。\n\n3. 有效意思。行为人表明的意思,内容必须合法,不得违反强制性法律规定和社会公序良俗。\n\n4. 自由意思。行为人表明意思时,应当在自由意志的基础上作出。\n\n总之,民法所说的"意思表示"是指通过言语、书面或其他方式,将行为人内心的某种意思以他人能够',
                    'glm': '意思表示是指民事主体通过言语、行为或其他方式表达其内心意愿和意思的过程。',
                    'gpt_3': '在民法中，"意思表示"是指个人或法人通过语言、行为或其他方式表达出的意愿，以达成法律上的效果。它是指个人或法人通过言辞、书面文件、行为等方式，表达出的意愿，以达成法律上的效果。例如，签订合同、提出要约、接受邀约等都属于意思表示的范畴。在民法中，意思表示是一种重要的法律行为，它是民事法律关系形成的基础。根据民法的规定，意思表示应当是真实、明确、自由和合法的。',
                    'minimax': '意思表示是民法中一个重要的概念，是指民事主体通过语言、文字或者行为等方式表达其设立、变更、终止民事法律关系的意图。意思表示是民事法律行为的核心要素，是民事主体参与民事活动的基本方式。\n\n在民法中，意思表示可以分为明示和默示两种。明示的意思表示是指民事主体通过明确的语言、文字或者行为，将其设立、变更、终止民事法律关系的意图表达出来。默示的意思表示是指民事主体通过特定的行为或者事实状态，将其设立、变更、终止民事法律关系的意图表达出来。\n\n意思表示的效力取决于其真实性、合法性和明确性。真实性是指意思表示应当是民事主体真实意愿的反映，不能是虚假的或者受欺诈、胁迫等影响的。合法性是指意思表示的内容应当符合法律的规定和社会公共利益。明确性是指意思表示的内容应当清晰、明确，不能含糊不清或者存在歧义。\n\n意思表示是民事主体参与民事活动的基本方式，也是民事法律行为成立的基本条件。在民法中，意思表示的效力、成立要件、生效时间等问题都是民法研究的重要内容。',
                    'tongyi': '在民法中，“意思表示”是指当事人通过言语、行为或其他方式，将其意思表达出来，从而使他人能够了解其真实意图的行为。意思表示是民事法律行为的基础，是当事人之间设立、变更或终止民事权利义务关系的重要手段。在实践中，意思表示的表达方式可以有多种，包括口头、书面、行为等。',
                    'wenxinyiyan': '民法中的“意思表示”是指，行为人将其发生、变更或终止民事权利、义务的内心意思，以一定的方式加以表达，并希望发生其效果的行为。',
                    'xunfei': '在民法中，“意思表示”是指当事人通过语言、文字、符号、肢体动作等方式表达其意思的行为。 ',
                    'multi_output': '[{"model": "360", "output": "\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u201c\\u610f\\u601d\\u8868\\u793a\\u201d\\u662f\\u6307\\u4e00\\u65b9\\u901a\\u8fc7\\u4e00\\u5b9a\\u7684\\u65b9\\u5f0f\\u8868\\u8fbe\\u5176\\u771f\\u5b9e\\u610f\\u613f\\u7684\\u610f\\u601d\\uff0c\\u53e6\\u4e00\\u65b9\\u636e\\u6b64\\u4ea7\\u751f\\u6cd5\\u5f8b\\u6548\\u679c\\u7684\\u884c\\u4e3a\\u3002\\u5177\\u4f53\\u800c\\u8a00\\uff0c\\u610f\\u601d\\u8868\\u793a\\u5fc5\\u987b\\u5305\\u542b\\u4ee5\\u4e0b\\u8981\\u7d20: \\n1. \\u8868\\u793a\\u4eba\\u5177\\u6709\\u771f\\u5b9e\\u7684\\u610f\\u613f\\u3002 \\n2. \\u8868\\u793a\\u4eba\\u6709\\u80fd\\u529b\\u8868\\u8fbe\\u81ea\\u5df1\\u7684\\u610f\\u613f\\u3002 \\n3. \\u8868\\u793a\\u4eba\\u662f\\u81ea\\u613f\\u7684\\uff0c\\u6ca1\\u6709\\u53d7\\u5230\\u4efb\\u4f55\\u5f3a\\u5236\\u6216\\u6b3a\\u9a97\\u7684\\u5f71\\u54cd\\u3002 \\n4. \\u8868\\u793a\\u7684\\u5185\\u5bb9\\u662f\\u771f\\u5b9e\\u7684\\u3002 \\n5. \\u8868\\u793a\\u7684\\u5185\\u5bb9\\u662f\\u5177\\u4f53\\u7684\\u548c\\u660e\\u786e\\u7684\\u3002 \\n6. \\u8868\\u793a\\u4eba\\u5e0c\\u671b\\u4ee5\\u610f\\u601d\\u8868\\u793a\\u7684\\u5f62\\u5f0f\\u4e0e\\u4ed6\\u4eba\\u5efa\\u7acb\\u6cd5\\u5f8b\\u5173\\u7cfb\\u3002 \\n\\u603b\\u4e4b\\uff0c\\u610f\\u601d\\u8868\\u793a\\u662f\\u5f53\\u4e8b\\u4eba\\u4e4b\\u95f4\\u5efa\\u7acb\\u6cd5\\u5f8b\\u5173\\u7cfb\\u7684\\u91cd\\u8981\\u65b9\\u5f0f\\uff0c\\u5b83\\u5fc5\\u987b\\u771f\\u5b9e\\u3001\\u81ea\\u613f\\u3001\\u660e\\u786e\\uff0c\\u624d\\u80fd\\u4ea7\\u751f\\u6cd5\\u5f8b\\u6548\\u679c\\u3002"}, {"model": "baichuan", "output": "\\u610f\\u601d\\u8868\\u793a\\u662f\\u6307\\u5411\\u5916\\u90e8\\u8868\\u660e\\u610f\\u613f\\u7684\\u884c\\u4e3a\\uff0c\\u5305\\u62ec\\u660e\\u793a(\\u76f4\\u63a5\\u8868\\u8fbe)\\u548c\\u9ed8\\u793a(\\u95f4\\u63a5\\u8868\\u8fbe)\\u4e24\\u79cd\\u65b9\\u5f0f\\u3002\\u4ee5\\u4e0b\\u662f\\u4e00\\u4e9b\\u5173\\u4e8e\\u610f\\u601d\\u8868\\u793a\\u7684\\u57fa\\u672c\\u8981\\u7d20\\uff1a\\n\\n1. \\u4e3b\\u4f53\\u8d44\\u683c\\uff1a\\u53d1\\u51fa\\u610f\\u601d\\u8868\\u793a\\u7684\\u4eba\\u5fc5\\u987b\\u5177\\u6709\\u76f8\\u5e94\\u7684\\u884c\\u4e3a\\u80fd\\u529b\\uff0c\\u5373\\u5177\\u5907\\u5b8c\\u5168\\u6c11\\u4e8b\\u884c\\u4e3a\\u80fd\\u529b\\u3002\\n\\n2. \\u610f\\u8bc6\\u56e0\\u7d20\\uff1a\\u610f\\u601d\\u8868\\u793a\\u5fc5\\u987b\\u662f\\u771f\\u5b9e\\u7684\\u3001\\u81ea\\u613f\\u7684\\uff0c\\u800c\\u4e0d\\u662f\\u88ab\\u8feb\\u7684\\u6216\\u8005\\u662f\\u65e0\\u610f\\u8bc6\\u7684\\u3002\\n\\n3. \\u660e\\u786e\\u6027\\uff1a\\u610f\\u601d\\u8868\\u793a\\u5fc5\\u987b\\u6e05\\u6670\\u3001\\u51c6\\u786e\\u5730\\u4f20\\u8fbe\\u4e86\\u53d1\\u51fa\\u7684\\u610f\\u56fe\\uff0c\\u5426\\u5219\\u53ef\\u80fd\\u4f1a\\u5bfc\\u81f4\\u6b67\\u4e49\\u6216\\u8bef\\u89e3\\u3002\\n\\n4. \\u5408\\u6cd5\\u6027\\uff1a\\u610f\\u601d\\u8868\\u793a\\u4e0d\\u5f97\\u8fdd\\u6cd5\\uff0c\\u4f8b\\u5982\\u4e0d\\u5f97\\u6b3a\\u8bc8\\u3001\\u80c1\\u8feb\\u6216\\u6076\\u610f\\u4e32\\u901a\\u3002\\n\\n5. \\u81ea\\u51b3\\u6027\\uff1a\\u610f\\u601d\\u8868\\u793a\\u5fc5\\u987b\\u662f\\u81ea\\u5df1\\u505a\\u51fa\\u7684\\uff0c\\u800c\\u4e0d\\u662f\\u88ab\\u522b\\u4eba\\u64cd\\u7eb5\\u6216\\u63a7\\u5236\\u7684\\u3002\\n\\n6. \\u53ef\\u64a4\\u56de\\u6027\\uff1a\\u5982\\u679c\\u610f\\u601d\\u8868\\u793a\\u8fd8\\u6ca1\\u6709\\u5230\\u8fbe\\u76f8\\u5bf9\\u4eba\\uff0c\\u90a3\\u4e48\\u53ef\\u4ee5\\u64a4\\u56de;\\u4e00\\u65e6\\u5230\\u8fbe\\u76f8\\u5bf9\\u4eba\\uff0c\\u5219\\u65e0\\u6cd5\\u64a4\\u56de\\u3002\\n\\n7. \\u53ef\\u64a4\\u9500\\u6027\\uff1a\\u5982\\u679c\\u610f\\u601d\\u8868\\u793a\\u5b58\\u5728\\u7455\\u75b5\\uff0c\\u5982\\u91cd\\u5927\\u8bef\\u89e3\\u3001\\u663e\\u5931\\u516c\\u5e73\\u7b49\\uff0c\\u90a3\\u4e48\\u53ef\\u4ee5\\u5728\\u4e00\\u5b9a\\u671f\\u9650\\u5185\\u7533\\u8bf7\\u64a4\\u9500\\u3002\\n\\n8. \\u62d8\\u675f\\u529b\\uff1a\\u4e00\\u65e6\\u610f\\u601d\\u8868\\u793a\\u5b8c\\u6210\\u5e76\\u5230\\u8fbe\\u76f8\\u5bf9\\u4eba\\uff0c\\u5c31\\u4f1a\\u4ea7\\u751f\\u4e00\\u5b9a\\u7684\\u6cd5\\u5f8b\\u6548\\u679c\\uff0c\\u5bf9\\u53cc\\u65b9\\u4ea7\\u751f\\u7ea6\\u675f\\u529b\\u3002"}, {"model": "claude", "output": " \\u6c11\\u6cd5\\u4e2d\\"\\u610f\\u601d\\u8868\\u793a\\"\\u662f\\u6307\\u8868\\u660e\\u610f\\u601d\\u7684\\u884c\\u4e3a,\\u53ef\\u4ee5\\u662f\\u8a00\\u8bed\\u3001\\u4e66\\u9762\\u6216\\u5176\\u4ed6\\u884c\\u4e3a\\u3002\\u5b83\\u662f\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u884c\\u4e3a\\u7684\\u5fc5\\u8981\\u7ec4\\u6210\\u90e8\\u5206\\u4e4b\\u4e00\\u3002\\n\\n\\u610f\\u601d\\u8868\\u793a\\u9700\\u8981\\u7b26\\u5408\\u5982\\u4e0b\\u6761\\u4ef6:\\n\\n1. \\u8868\\u660e\\u610f\\u601d\\u7684\\u610f\\u601d\\u3002\\u884c\\u4e3a\\u4eba\\u5fc5\\u987b\\u6709\\u8868\\u660e\\u67d0\\u79cd\\u610f\\u601d\\u7684\\u4e3b\\u89c2\\u610f\\u601d\\u3002\\n\\n2. \\u4ed6\\u4eba\\u80fd\\u591f\\u4e86\\u89e3\\u3002\\u884c\\u4e3a\\u4eba\\u8868\\u660e\\u7684\\u610f\\u601d,\\u5fc5\\u987b\\u4ee5\\u4ed6\\u4eba\\u80fd\\u591f\\u7406\\u89e3\\u7684\\u65b9\\u5f0f\\u8868\\u8fbe\\u3002\\n\\n3. \\u6709\\u6548\\u610f\\u601d\\u3002\\u884c\\u4e3a\\u4eba\\u8868\\u660e\\u7684\\u610f\\u601d,\\u5185\\u5bb9\\u5fc5\\u987b\\u5408\\u6cd5,\\u4e0d\\u5f97\\u8fdd\\u53cd\\u5f3a\\u5236\\u6027\\u6cd5\\u5f8b\\u89c4\\u5b9a\\u548c\\u793e\\u4f1a\\u516c\\u5e8f\\u826f\\u4fd7\\u3002\\n\\n4. \\u81ea\\u7531\\u610f\\u601d\\u3002\\u884c\\u4e3a\\u4eba\\u8868\\u660e\\u610f\\u601d\\u65f6,\\u5e94\\u5f53\\u5728\\u81ea\\u7531\\u610f\\u5fd7\\u7684\\u57fa\\u7840\\u4e0a\\u4f5c\\u51fa\\u3002\\n\\n\\u603b\\u4e4b,\\u6c11\\u6cd5\\u6240\\u8bf4\\u7684\\"\\u610f\\u601d\\u8868\\u793a\\"\\u662f\\u6307\\u901a\\u8fc7\\u8a00\\u8bed\\u3001\\u4e66\\u9762\\u6216\\u5176\\u4ed6\\u65b9\\u5f0f,\\u5c06\\u884c\\u4e3a\\u4eba\\u5185\\u5fc3\\u7684\\u67d0\\u79cd\\u610f\\u601d\\u4ee5\\u4ed6\\u4eba\\u80fd\\u591f"}, {"model": "glm", "output": "\\u610f\\u601d\\u8868\\u793a\\u662f\\u6307\\u6c11\\u4e8b\\u4e3b\\u4f53\\u901a\\u8fc7\\u8a00\\u8bed\\u3001\\u884c\\u4e3a\\u6216\\u5176\\u4ed6\\u65b9\\u5f0f\\u8868\\u8fbe\\u5176\\u5185\\u5fc3\\u610f\\u613f\\u548c\\u610f\\u601d\\u7684\\u8fc7\\u7a0b\\u3002"}, {"model": "gpt_3", "output": "\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\"\\u610f\\u601d\\u8868\\u793a\\"\\u662f\\u6307\\u4e2a\\u4eba\\u6216\\u6cd5\\u4eba\\u901a\\u8fc7\\u8bed\\u8a00\\u3001\\u884c\\u4e3a\\u6216\\u5176\\u4ed6\\u65b9\\u5f0f\\u8868\\u8fbe\\u51fa\\u7684\\u610f\\u613f\\uff0c\\u4ee5\\u8fbe\\u6210\\u6cd5\\u5f8b\\u4e0a\\u7684\\u6548\\u679c\\u3002\\u5b83\\u662f\\u6307\\u4e2a\\u4eba\\u6216\\u6cd5\\u4eba\\u901a\\u8fc7\\u8a00\\u8f9e\\u3001\\u4e66\\u9762\\u6587\\u4ef6\\u3001\\u884c\\u4e3a\\u7b49\\u65b9\\u5f0f\\uff0c\\u8868\\u8fbe\\u51fa\\u7684\\u610f\\u613f\\uff0c\\u4ee5\\u8fbe\\u6210\\u6cd5\\u5f8b\\u4e0a\\u7684\\u6548\\u679c\\u3002\\u4f8b\\u5982\\uff0c\\u7b7e\\u8ba2\\u5408\\u540c\\u3001\\u63d0\\u51fa\\u8981\\u7ea6\\u3001\\u63a5\\u53d7\\u9080\\u7ea6\\u7b49\\u90fd\\u5c5e\\u4e8e\\u610f\\u601d\\u8868\\u793a\\u7684\\u8303\\u7574\\u3002\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u610f\\u601d\\u8868\\u793a\\u662f\\u4e00\\u79cd\\u91cd\\u8981\\u7684\\u6cd5\\u5f8b\\u884c\\u4e3a\\uff0c\\u5b83\\u662f\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u5173\\u7cfb\\u5f62\\u6210\\u7684\\u57fa\\u7840\\u3002\\u6839\\u636e\\u6c11\\u6cd5\\u7684\\u89c4\\u5b9a\\uff0c\\u610f\\u601d\\u8868\\u793a\\u5e94\\u5f53\\u662f\\u771f\\u5b9e\\u3001\\u660e\\u786e\\u3001\\u81ea\\u7531\\u548c\\u5408\\u6cd5\\u7684\\u3002"}, {"model": "minimax", "output": "\\u610f\\u601d\\u8868\\u793a\\u662f\\u6c11\\u6cd5\\u4e2d\\u4e00\\u4e2a\\u91cd\\u8981\\u7684\\u6982\\u5ff5\\uff0c\\u662f\\u6307\\u6c11\\u4e8b\\u4e3b\\u4f53\\u901a\\u8fc7\\u8bed\\u8a00\\u3001\\u6587\\u5b57\\u6216\\u8005\\u884c\\u4e3a\\u7b49\\u65b9\\u5f0f\\u8868\\u8fbe\\u5176\\u8bbe\\u7acb\\u3001\\u53d8\\u66f4\\u3001\\u7ec8\\u6b62\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u5173\\u7cfb\\u7684\\u610f\\u56fe\\u3002\\u610f\\u601d\\u8868\\u793a\\u662f\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u884c\\u4e3a\\u7684\\u6838\\u5fc3\\u8981\\u7d20\\uff0c\\u662f\\u6c11\\u4e8b\\u4e3b\\u4f53\\u53c2\\u4e0e\\u6c11\\u4e8b\\u6d3b\\u52a8\\u7684\\u57fa\\u672c\\u65b9\\u5f0f\\u3002\\n\\n\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u610f\\u601d\\u8868\\u793a\\u53ef\\u4ee5\\u5206\\u4e3a\\u660e\\u793a\\u548c\\u9ed8\\u793a\\u4e24\\u79cd\\u3002\\u660e\\u793a\\u7684\\u610f\\u601d\\u8868\\u793a\\u662f\\u6307\\u6c11\\u4e8b\\u4e3b\\u4f53\\u901a\\u8fc7\\u660e\\u786e\\u7684\\u8bed\\u8a00\\u3001\\u6587\\u5b57\\u6216\\u8005\\u884c\\u4e3a\\uff0c\\u5c06\\u5176\\u8bbe\\u7acb\\u3001\\u53d8\\u66f4\\u3001\\u7ec8\\u6b62\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u5173\\u7cfb\\u7684\\u610f\\u56fe\\u8868\\u8fbe\\u51fa\\u6765\\u3002\\u9ed8\\u793a\\u7684\\u610f\\u601d\\u8868\\u793a\\u662f\\u6307\\u6c11\\u4e8b\\u4e3b\\u4f53\\u901a\\u8fc7\\u7279\\u5b9a\\u7684\\u884c\\u4e3a\\u6216\\u8005\\u4e8b\\u5b9e\\u72b6\\u6001\\uff0c\\u5c06\\u5176\\u8bbe\\u7acb\\u3001\\u53d8\\u66f4\\u3001\\u7ec8\\u6b62\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u5173\\u7cfb\\u7684\\u610f\\u56fe\\u8868\\u8fbe\\u51fa\\u6765\\u3002\\n\\n\\u610f\\u601d\\u8868\\u793a\\u7684\\u6548\\u529b\\u53d6\\u51b3\\u4e8e\\u5176\\u771f\\u5b9e\\u6027\\u3001\\u5408\\u6cd5\\u6027\\u548c\\u660e\\u786e\\u6027\\u3002\\u771f\\u5b9e\\u6027\\u662f\\u6307\\u610f\\u601d\\u8868\\u793a\\u5e94\\u5f53\\u662f\\u6c11\\u4e8b\\u4e3b\\u4f53\\u771f\\u5b9e\\u610f\\u613f\\u7684\\u53cd\\u6620\\uff0c\\u4e0d\\u80fd\\u662f\\u865a\\u5047\\u7684\\u6216\\u8005\\u53d7\\u6b3a\\u8bc8\\u3001\\u80c1\\u8feb\\u7b49\\u5f71\\u54cd\\u7684\\u3002\\u5408\\u6cd5\\u6027\\u662f\\u6307\\u610f\\u601d\\u8868\\u793a\\u7684\\u5185\\u5bb9\\u5e94\\u5f53\\u7b26\\u5408\\u6cd5\\u5f8b\\u7684\\u89c4\\u5b9a\\u548c\\u793e\\u4f1a\\u516c\\u5171\\u5229\\u76ca\\u3002\\u660e\\u786e\\u6027\\u662f\\u6307\\u610f\\u601d\\u8868\\u793a\\u7684\\u5185\\u5bb9\\u5e94\\u5f53\\u6e05\\u6670\\u3001\\u660e\\u786e\\uff0c\\u4e0d\\u80fd\\u542b\\u7cca\\u4e0d\\u6e05\\u6216\\u8005\\u5b58\\u5728\\u6b67\\u4e49\\u3002\\n\\n\\u610f\\u601d\\u8868\\u793a\\u662f\\u6c11\\u4e8b\\u4e3b\\u4f53\\u53c2\\u4e0e\\u6c11\\u4e8b\\u6d3b\\u52a8\\u7684\\u57fa\\u672c\\u65b9\\u5f0f\\uff0c\\u4e5f\\u662f\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u884c\\u4e3a\\u6210\\u7acb\\u7684\\u57fa\\u672c\\u6761\\u4ef6\\u3002\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u610f\\u601d\\u8868\\u793a\\u7684\\u6548\\u529b\\u3001\\u6210\\u7acb\\u8981\\u4ef6\\u3001\\u751f\\u6548\\u65f6\\u95f4\\u7b49\\u95ee\\u9898\\u90fd\\u662f\\u6c11\\u6cd5\\u7814\\u7a76\\u7684\\u91cd\\u8981\\u5185\\u5bb9\\u3002"}, {"model": "tongyi", "output": "\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u201c\\u610f\\u601d\\u8868\\u793a\\u201d\\u662f\\u6307\\u5f53\\u4e8b\\u4eba\\u901a\\u8fc7\\u8a00\\u8bed\\u3001\\u884c\\u4e3a\\u6216\\u5176\\u4ed6\\u65b9\\u5f0f\\uff0c\\u5c06\\u5176\\u610f\\u601d\\u8868\\u8fbe\\u51fa\\u6765\\uff0c\\u4ece\\u800c\\u4f7f\\u4ed6\\u4eba\\u80fd\\u591f\\u4e86\\u89e3\\u5176\\u771f\\u5b9e\\u610f\\u56fe\\u7684\\u884c\\u4e3a\\u3002\\u610f\\u601d\\u8868\\u793a\\u662f\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u884c\\u4e3a\\u7684\\u57fa\\u7840\\uff0c\\u662f\\u5f53\\u4e8b\\u4eba\\u4e4b\\u95f4\\u8bbe\\u7acb\\u3001\\u53d8\\u66f4\\u6216\\u7ec8\\u6b62\\u6c11\\u4e8b\\u6743\\u5229\\u4e49\\u52a1\\u5173\\u7cfb\\u7684\\u91cd\\u8981\\u624b\\u6bb5\\u3002\\u5728\\u5b9e\\u8df5\\u4e2d\\uff0c\\u610f\\u601d\\u8868\\u793a\\u7684\\u8868\\u8fbe\\u65b9\\u5f0f\\u53ef\\u4ee5\\u6709\\u591a\\u79cd\\uff0c\\u5305\\u62ec\\u53e3\\u5934\\u3001\\u4e66\\u9762\\u3001\\u884c\\u4e3a\\u7b49\\u3002"}, {"model": "wenxinyiyan", "output": "\\u6c11\\u6cd5\\u4e2d\\u7684\\u201c\\u610f\\u601d\\u8868\\u793a\\u201d\\u662f\\u6307\\uff0c\\u884c\\u4e3a\\u4eba\\u5c06\\u5176\\u53d1\\u751f\\u3001\\u53d8\\u66f4\\u6216\\u7ec8\\u6b62\\u6c11\\u4e8b\\u6743\\u5229\\u3001\\u4e49\\u52a1\\u7684\\u5185\\u5fc3\\u610f\\u601d\\uff0c\\u4ee5\\u4e00\\u5b9a\\u7684\\u65b9\\u5f0f\\u52a0\\u4ee5\\u8868\\u8fbe\\uff0c\\u5e76\\u5e0c\\u671b\\u53d1\\u751f\\u5176\\u6548\\u679c\\u7684\\u884c\\u4e3a\\u3002"}, {"model": "xunfei", "output": "\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u201c\\u610f\\u601d\\u8868\\u793a\\u201d\\u662f\\u6307\\u5f53\\u4e8b\\u4eba\\u901a\\u8fc7\\u8bed\\u8a00\\u3001\\u6587\\u5b57\\u3001\\u7b26\\u53f7\\u3001\\u80a2\\u4f53\\u52a8\\u4f5c\\u7b49\\u65b9\\u5f0f\\u8868\\u8fbe\\u5176\\u610f\\u601d\\u7684\\u884c\\u4e3a\\u3002 "}]',
                    'annotator': 'self_check_gpt4',
                    'multi_rank_per_instruction': "[\n    {'model': 'baichuan', 'rank': 1},\n    {'model': 'gpt_3', 'rank': 2},\n    {'model': '360', 'rank': 3},\n    {'model': 'minimax', 'rank': 4},\n    {'model': 'claude', 'rank': 5},\n    {'model': 'wenxinyiyan', 'rank': 6},\n    {'model': 'tongyi', 'rank': 7},\n    {'model': 'glm', 'rank': 8},\n    {'model': 'xunfei', 'rank': 9}\n]",
                    'price_per_example': 0.19065, 'time_per_example': 7.895266056060791},
                   {'instruction': ' 表见代理（总则编）：请解释民法中“表见代理”的含义',
                    'reference_output': '指被代理人的行为足以使第三人相信无权代理人具有代理权，并基于这种信赖而与无权代理人实施法律行为的代理。',
                    '360': ' 在民法中，“表见代理”是指第三人通过自己的行为或者通过其他方式，使得被代理人(即代理人)被视为有代理权的行为，从而使得第三人有权直接向被代理人要求履行一定的义务。表见代理是民法中的一种特殊的代理关系，它基于第三人的行为或者某些其他方式，使得被代理人被错误地认为具有代理权，第三人因此获得了对被代理人的权利。 \n表见代理的构成要件如下: \n1. 代理人与第三人之间存在着一定的法律关系; \n2. 代理人的行为导致了被代理人被错误地认为具有代理权; \n3. 代理人的行为是符合法律规定的; \n4. 第三人善意地相信被代理人具有代理权。 \n在表见代理的情况下，第三人可以要求被代理人履行相应的义务，被代理人则不能以自己不具有代理权为由进行抗辩。因此，表见代理在民事活动中具有重要意义，它有助于保护第三人的合法权益，促进交易的顺利进行。',
                    'baichuan': '表见代理，又称为“表面授权”，是一种特殊的无权代理现象。它发生在一个人(代理人)实际上并没有获得另一个人的(被代理人)授权，但他却能够通过某种方式让其他人(相对人)相信他拥有这种授权。这种情况通常是因为被代理人本身的原因导致的，例如未告知他人代理权已经终止，或者未能及时更正错误的信息。\n\n表见代理的主要特点如下：\n\n1. 代理人实际上没有代理权，但在客观上形成了具有代理权的表象。\n\n2. 相对人有足够的理由相信代理人具有代理权，这可能是基于被代理人的行为、声明或者公开信息等。\n\n3. 被代理人需要对代理人的行为承担法律责任。这是因为，尽管被代理人并未真正授予代理权，但他的行为使得相对人产生了信任，从而与代理人进行了法律行为。为了保护善意第三人的利益，维护交易安全，法律要求被代理人承担这种法律责任。\n\n4. 表见代理并非总是有效的。在某些情况下，被代理人可以证明自己并未授予代理权，或者证明代理人的行为违反了法律规定，那么表见代理就不再成立，被代理人不需要承担法律责任。\n\n需要注意的是，表见代理在不同国家和地区的法律体系中可能有所不同，具体规定也会因情况而异。了解这些差异有助于更好地理解和应对表见代理问题。',
                    'claude': ' 表见代理是民法中的一个概念。\n\n表见代理是指代理人代表被代理人进行民事法律行为,但代理人的身份和权限未向第三人表明的一种代理。\n\n表见代理的特点是:\n\n1. 代理人实际上获得了被代理人进行某项或某些法律行为的授权。\n\n2. 但是代理人未向第三人表明自己是代表被代理人行事。第三人以为代理人是以自己的名义和为自己利益进行行为。\n\n3. 因此,表见代理产生的法律效果,是以代理人为当事人。\n\n4. 但是,被代理人对第三人承担连带责任。如果代理人无法履行或者无法全部履行责任,被代理人应承担责任。\n\n5. 表见代理一旦产生,被代理人',
                    'glm': '表见代理是指行为人虽然没有代理权，但相对人有理由认为行为人有代理权而与其进行法律行为，其行为的法律后果由被代理人承担的代理。表见代理的构成条件包括：行为人没有代理权；有使相对人相信行为人具有代理权的事实或者理由；行为人与相对人之间的民事行为不得违反法律或者社会的公共利益。表见代理制度是为了保护善意第三人的信赖利益和交易的安全而设立的，在被代理人的过失或被代理人与无权代理人之间存在特殊关系的情况下，法律强制被代理人承担其法律后果。在我国，《中华人民共和国民法总则》第一百七十二条规定了表见代理的相关内容。',
                    'gpt_3': '表见代理是指当事人通过一种明示的方式，委托他人代为进行法律行为的一种形式。在民法中，表见代理是指代理人根据当事人的授权，以自己的名义与第三人进行交涉、签订合同或进行其他法律行为。表见代理需要满足一定的条件，包括代理人必须具备完全民事行为能力，代理行为必须符合法律规定的范围和目的，并且代理人必须按照当事人的意思进行代理。通过表见代理，当事人可以通过代理人来实现自己的权益，同时也需要承担代理行为所产生的法律后果。',
                    'minimax': '表见代理，是指行为人没有代理权、超越代理权或者代理权终止后，仍然实施代理行为，相对人有理由相信行为人有代理权的，代理行为有效。表见代理是代理制度的一种例外，旨在保护善意相对人的合法权益，维护交易安全。\n\n根据《中华人民共和国民法典》第一百七十二条规定：“行为人没有代理权、超越代理权或者代理权终止后，仍然实施代理行为，相对人有理由相信行为人有代理权的，代理行为有效。”\n\n表见代理的构成要件包括：\n1. 行为人没有代理权、超越代理权或者代理权终止；\n2. 行为人以被代理人的名义进行民事活动；\n3. 相对人有理由相信行为人有代理权；\n4. 相对人基于这种相信而与行为人进行民事活动。\n\n在表见代理的情况下，被代理人应当对相对人的损失承担民事责任。如果被代理人因此遭受损失，可以向无权代理人追偿。',
                    'tongyi': '在民法中，“表见代理”是指当事人虽然没有代理权，但是由于第三人的误解，而使得第三人相信其有代理权，并与其进行了民事活动。在这种情况下，如果代理人和被代理人之间的行为构成了真实的代理关系，那么表见代理就是有效的，第三人可以依据表见代理关系向被代理人主张权利。',
                    'wenxinyiyan': '民法中的“表见代理”是指行为人虽无代理权，但由于本人的行为，造成了足以使善意第三人相信其有代理权的表象，而与善意第三人进行的、由被代理人承担法律后果的代理行为。\n表见代理实质上是无权代理，即行为人事实上并无代理权。但由于本人的行为，造成了足以使善意第三人相信其有代理权的表象。根据法律的规定，行为人的代理行为有效，第三人与之发生法律关系的后果由被代理人承担。\n根据《民法通则》的规定，如果被代理人的行为不违法，则该行为有效。如果被代理人的行为违法，则该行为无效。如果被代理人的行为不合法，但第三人不知道或者应当知道而没有知道，则该行为有效。如果第三人知道或者应当知道而没有知道，则该行为无效。',
                    'xunfei': '表见代理是指行为人虽无代理权而实施代理行为，如果相对人有理由相信其有代理权，该代理行为有效。《民法典》第一百七十二条规定了表见代理制度。 \n\n在总则编中，表见代理的定义为：所谓表见代理，是指虽然行为人事实上没有代理权，但相对人有理由认为行为人有代理权而与其进行法律行为，其行为的法律后果由被代理人承担的代理。',
                    'multi_output': '[{"model": "360", "output": " \\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u201c\\u8868\\u89c1\\u4ee3\\u7406\\u201d\\u662f\\u6307\\u7b2c\\u4e09\\u4eba\\u901a\\u8fc7\\u81ea\\u5df1\\u7684\\u884c\\u4e3a\\u6216\\u8005\\u901a\\u8fc7\\u5176\\u4ed6\\u65b9\\u5f0f\\uff0c\\u4f7f\\u5f97\\u88ab\\u4ee3\\u7406\\u4eba(\\u5373\\u4ee3\\u7406\\u4eba)\\u88ab\\u89c6\\u4e3a\\u6709\\u4ee3\\u7406\\u6743\\u7684\\u884c\\u4e3a\\uff0c\\u4ece\\u800c\\u4f7f\\u5f97\\u7b2c\\u4e09\\u4eba\\u6709\\u6743\\u76f4\\u63a5\\u5411\\u88ab\\u4ee3\\u7406\\u4eba\\u8981\\u6c42\\u5c65\\u884c\\u4e00\\u5b9a\\u7684\\u4e49\\u52a1\\u3002\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6c11\\u6cd5\\u4e2d\\u7684\\u4e00\\u79cd\\u7279\\u6b8a\\u7684\\u4ee3\\u7406\\u5173\\u7cfb\\uff0c\\u5b83\\u57fa\\u4e8e\\u7b2c\\u4e09\\u4eba\\u7684\\u884c\\u4e3a\\u6216\\u8005\\u67d0\\u4e9b\\u5176\\u4ed6\\u65b9\\u5f0f\\uff0c\\u4f7f\\u5f97\\u88ab\\u4ee3\\u7406\\u4eba\\u88ab\\u9519\\u8bef\\u5730\\u8ba4\\u4e3a\\u5177\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u7b2c\\u4e09\\u4eba\\u56e0\\u6b64\\u83b7\\u5f97\\u4e86\\u5bf9\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u6743\\u5229\\u3002 \\n\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u6784\\u6210\\u8981\\u4ef6\\u5982\\u4e0b: \\n1. \\u4ee3\\u7406\\u4eba\\u4e0e\\u7b2c\\u4e09\\u4eba\\u4e4b\\u95f4\\u5b58\\u5728\\u7740\\u4e00\\u5b9a\\u7684\\u6cd5\\u5f8b\\u5173\\u7cfb; \\n2. \\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u5bfc\\u81f4\\u4e86\\u88ab\\u4ee3\\u7406\\u4eba\\u88ab\\u9519\\u8bef\\u5730\\u8ba4\\u4e3a\\u5177\\u6709\\u4ee3\\u7406\\u6743; \\n3. \\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u662f\\u7b26\\u5408\\u6cd5\\u5f8b\\u89c4\\u5b9a\\u7684; \\n4. \\u7b2c\\u4e09\\u4eba\\u5584\\u610f\\u5730\\u76f8\\u4fe1\\u88ab\\u4ee3\\u7406\\u4eba\\u5177\\u6709\\u4ee3\\u7406\\u6743\\u3002 \\n\\u5728\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u60c5\\u51b5\\u4e0b\\uff0c\\u7b2c\\u4e09\\u4eba\\u53ef\\u4ee5\\u8981\\u6c42\\u88ab\\u4ee3\\u7406\\u4eba\\u5c65\\u884c\\u76f8\\u5e94\\u7684\\u4e49\\u52a1\\uff0c\\u88ab\\u4ee3\\u7406\\u4eba\\u5219\\u4e0d\\u80fd\\u4ee5\\u81ea\\u5df1\\u4e0d\\u5177\\u6709\\u4ee3\\u7406\\u6743\\u4e3a\\u7531\\u8fdb\\u884c\\u6297\\u8fa9\\u3002\\u56e0\\u6b64\\uff0c\\u8868\\u89c1\\u4ee3\\u7406\\u5728\\u6c11\\u4e8b\\u6d3b\\u52a8\\u4e2d\\u5177\\u6709\\u91cd\\u8981\\u610f\\u4e49\\uff0c\\u5b83\\u6709\\u52a9\\u4e8e\\u4fdd\\u62a4\\u7b2c\\u4e09\\u4eba\\u7684\\u5408\\u6cd5\\u6743\\u76ca\\uff0c\\u4fc3\\u8fdb\\u4ea4\\u6613\\u7684\\u987a\\u5229\\u8fdb\\u884c\\u3002"}, {"model": "baichuan", "output": "\\u8868\\u89c1\\u4ee3\\u7406\\uff0c\\u53c8\\u79f0\\u4e3a\\u201c\\u8868\\u9762\\u6388\\u6743\\u201d\\uff0c\\u662f\\u4e00\\u79cd\\u7279\\u6b8a\\u7684\\u65e0\\u6743\\u4ee3\\u7406\\u73b0\\u8c61\\u3002\\u5b83\\u53d1\\u751f\\u5728\\u4e00\\u4e2a\\u4eba(\\u4ee3\\u7406\\u4eba)\\u5b9e\\u9645\\u4e0a\\u5e76\\u6ca1\\u6709\\u83b7\\u5f97\\u53e6\\u4e00\\u4e2a\\u4eba\\u7684(\\u88ab\\u4ee3\\u7406\\u4eba)\\u6388\\u6743\\uff0c\\u4f46\\u4ed6\\u5374\\u80fd\\u591f\\u901a\\u8fc7\\u67d0\\u79cd\\u65b9\\u5f0f\\u8ba9\\u5176\\u4ed6\\u4eba(\\u76f8\\u5bf9\\u4eba)\\u76f8\\u4fe1\\u4ed6\\u62e5\\u6709\\u8fd9\\u79cd\\u6388\\u6743\\u3002\\u8fd9\\u79cd\\u60c5\\u51b5\\u901a\\u5e38\\u662f\\u56e0\\u4e3a\\u88ab\\u4ee3\\u7406\\u4eba\\u672c\\u8eab\\u7684\\u539f\\u56e0\\u5bfc\\u81f4\\u7684\\uff0c\\u4f8b\\u5982\\u672a\\u544a\\u77e5\\u4ed6\\u4eba\\u4ee3\\u7406\\u6743\\u5df2\\u7ecf\\u7ec8\\u6b62\\uff0c\\u6216\\u8005\\u672a\\u80fd\\u53ca\\u65f6\\u66f4\\u6b63\\u9519\\u8bef\\u7684\\u4fe1\\u606f\\u3002\\n\\n\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u4e3b\\u8981\\u7279\\u70b9\\u5982\\u4e0b\\uff1a\\n\\n1. \\u4ee3\\u7406\\u4eba\\u5b9e\\u9645\\u4e0a\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u5728\\u5ba2\\u89c2\\u4e0a\\u5f62\\u6210\\u4e86\\u5177\\u6709\\u4ee3\\u7406\\u6743\\u7684\\u8868\\u8c61\\u3002\\n\\n2. \\u76f8\\u5bf9\\u4eba\\u6709\\u8db3\\u591f\\u7684\\u7406\\u7531\\u76f8\\u4fe1\\u4ee3\\u7406\\u4eba\\u5177\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u8fd9\\u53ef\\u80fd\\u662f\\u57fa\\u4e8e\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u3001\\u58f0\\u660e\\u6216\\u8005\\u516c\\u5f00\\u4fe1\\u606f\\u7b49\\u3002\\n\\n3. \\u88ab\\u4ee3\\u7406\\u4eba\\u9700\\u8981\\u5bf9\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u627f\\u62c5\\u6cd5\\u5f8b\\u8d23\\u4efb\\u3002\\u8fd9\\u662f\\u56e0\\u4e3a\\uff0c\\u5c3d\\u7ba1\\u88ab\\u4ee3\\u7406\\u4eba\\u5e76\\u672a\\u771f\\u6b63\\u6388\\u4e88\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u4ed6\\u7684\\u884c\\u4e3a\\u4f7f\\u5f97\\u76f8\\u5bf9\\u4eba\\u4ea7\\u751f\\u4e86\\u4fe1\\u4efb\\uff0c\\u4ece\\u800c\\u4e0e\\u4ee3\\u7406\\u4eba\\u8fdb\\u884c\\u4e86\\u6cd5\\u5f8b\\u884c\\u4e3a\\u3002\\u4e3a\\u4e86\\u4fdd\\u62a4\\u5584\\u610f\\u7b2c\\u4e09\\u4eba\\u7684\\u5229\\u76ca\\uff0c\\u7ef4\\u62a4\\u4ea4\\u6613\\u5b89\\u5168\\uff0c\\u6cd5\\u5f8b\\u8981\\u6c42\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u8fd9\\u79cd\\u6cd5\\u5f8b\\u8d23\\u4efb\\u3002\\n\\n4. \\u8868\\u89c1\\u4ee3\\u7406\\u5e76\\u975e\\u603b\\u662f\\u6709\\u6548\\u7684\\u3002\\u5728\\u67d0\\u4e9b\\u60c5\\u51b5\\u4e0b\\uff0c\\u88ab\\u4ee3\\u7406\\u4eba\\u53ef\\u4ee5\\u8bc1\\u660e\\u81ea\\u5df1\\u5e76\\u672a\\u6388\\u4e88\\u4ee3\\u7406\\u6743\\uff0c\\u6216\\u8005\\u8bc1\\u660e\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u8fdd\\u53cd\\u4e86\\u6cd5\\u5f8b\\u89c4\\u5b9a\\uff0c\\u90a3\\u4e48\\u8868\\u89c1\\u4ee3\\u7406\\u5c31\\u4e0d\\u518d\\u6210\\u7acb\\uff0c\\u88ab\\u4ee3\\u7406\\u4eba\\u4e0d\\u9700\\u8981\\u627f\\u62c5\\u6cd5\\u5f8b\\u8d23\\u4efb\\u3002\\n\\n\\u9700\\u8981\\u6ce8\\u610f\\u7684\\u662f\\uff0c\\u8868\\u89c1\\u4ee3\\u7406\\u5728\\u4e0d\\u540c\\u56fd\\u5bb6\\u548c\\u5730\\u533a\\u7684\\u6cd5\\u5f8b\\u4f53\\u7cfb\\u4e2d\\u53ef\\u80fd\\u6709\\u6240\\u4e0d\\u540c\\uff0c\\u5177\\u4f53\\u89c4\\u5b9a\\u4e5f\\u4f1a\\u56e0\\u60c5\\u51b5\\u800c\\u5f02\\u3002\\u4e86\\u89e3\\u8fd9\\u4e9b\\u5dee\\u5f02\\u6709\\u52a9\\u4e8e\\u66f4\\u597d\\u5730\\u7406\\u89e3\\u548c\\u5e94\\u5bf9\\u8868\\u89c1\\u4ee3\\u7406\\u95ee\\u9898\\u3002"}, {"model": "claude", "output": " \\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6c11\\u6cd5\\u4e2d\\u7684\\u4e00\\u4e2a\\u6982\\u5ff5\\u3002\\n\\n\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6307\\u4ee3\\u7406\\u4eba\\u4ee3\\u8868\\u88ab\\u4ee3\\u7406\\u4eba\\u8fdb\\u884c\\u6c11\\u4e8b\\u6cd5\\u5f8b\\u884c\\u4e3a,\\u4f46\\u4ee3\\u7406\\u4eba\\u7684\\u8eab\\u4efd\\u548c\\u6743\\u9650\\u672a\\u5411\\u7b2c\\u4e09\\u4eba\\u8868\\u660e\\u7684\\u4e00\\u79cd\\u4ee3\\u7406\\u3002\\n\\n\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u7279\\u70b9\\u662f:\\n\\n1. \\u4ee3\\u7406\\u4eba\\u5b9e\\u9645\\u4e0a\\u83b7\\u5f97\\u4e86\\u88ab\\u4ee3\\u7406\\u4eba\\u8fdb\\u884c\\u67d0\\u9879\\u6216\\u67d0\\u4e9b\\u6cd5\\u5f8b\\u884c\\u4e3a\\u7684\\u6388\\u6743\\u3002\\n\\n2. \\u4f46\\u662f\\u4ee3\\u7406\\u4eba\\u672a\\u5411\\u7b2c\\u4e09\\u4eba\\u8868\\u660e\\u81ea\\u5df1\\u662f\\u4ee3\\u8868\\u88ab\\u4ee3\\u7406\\u4eba\\u884c\\u4e8b\\u3002\\u7b2c\\u4e09\\u4eba\\u4ee5\\u4e3a\\u4ee3\\u7406\\u4eba\\u662f\\u4ee5\\u81ea\\u5df1\\u7684\\u540d\\u4e49\\u548c\\u4e3a\\u81ea\\u5df1\\u5229\\u76ca\\u8fdb\\u884c\\u884c\\u4e3a\\u3002\\n\\n3. \\u56e0\\u6b64,\\u8868\\u89c1\\u4ee3\\u7406\\u4ea7\\u751f\\u7684\\u6cd5\\u5f8b\\u6548\\u679c,\\u662f\\u4ee5\\u4ee3\\u7406\\u4eba\\u4e3a\\u5f53\\u4e8b\\u4eba\\u3002\\n\\n4. \\u4f46\\u662f,\\u88ab\\u4ee3\\u7406\\u4eba\\u5bf9\\u7b2c\\u4e09\\u4eba\\u627f\\u62c5\\u8fde\\u5e26\\u8d23\\u4efb\\u3002\\u5982\\u679c\\u4ee3\\u7406\\u4eba\\u65e0\\u6cd5\\u5c65\\u884c\\u6216\\u8005\\u65e0\\u6cd5\\u5168\\u90e8\\u5c65\\u884c\\u8d23\\u4efb,\\u88ab\\u4ee3\\u7406\\u4eba\\u5e94\\u627f\\u62c5\\u8d23\\u4efb\\u3002\\n\\n5. \\u8868\\u89c1\\u4ee3\\u7406\\u4e00\\u65e6\\u4ea7\\u751f,\\u88ab\\u4ee3\\u7406\\u4eba"}, {"model": "glm", "output": "\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6307\\u884c\\u4e3a\\u4eba\\u867d\\u7136\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u8ba4\\u4e3a\\u884c\\u4e3a\\u4eba\\u6709\\u4ee3\\u7406\\u6743\\u800c\\u4e0e\\u5176\\u8fdb\\u884c\\u6cd5\\u5f8b\\u884c\\u4e3a\\uff0c\\u5176\\u884c\\u4e3a\\u7684\\u6cd5\\u5f8b\\u540e\\u679c\\u7531\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u7684\\u4ee3\\u7406\\u3002\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u6784\\u6210\\u6761\\u4ef6\\u5305\\u62ec\\uff1a\\u884c\\u4e3a\\u4eba\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\uff1b\\u6709\\u4f7f\\u76f8\\u5bf9\\u4eba\\u76f8\\u4fe1\\u884c\\u4e3a\\u4eba\\u5177\\u6709\\u4ee3\\u7406\\u6743\\u7684\\u4e8b\\u5b9e\\u6216\\u8005\\u7406\\u7531\\uff1b\\u884c\\u4e3a\\u4eba\\u4e0e\\u76f8\\u5bf9\\u4eba\\u4e4b\\u95f4\\u7684\\u6c11\\u4e8b\\u884c\\u4e3a\\u4e0d\\u5f97\\u8fdd\\u53cd\\u6cd5\\u5f8b\\u6216\\u8005\\u793e\\u4f1a\\u7684\\u516c\\u5171\\u5229\\u76ca\\u3002\\u8868\\u89c1\\u4ee3\\u7406\\u5236\\u5ea6\\u662f\\u4e3a\\u4e86\\u4fdd\\u62a4\\u5584\\u610f\\u7b2c\\u4e09\\u4eba\\u7684\\u4fe1\\u8d56\\u5229\\u76ca\\u548c\\u4ea4\\u6613\\u7684\\u5b89\\u5168\\u800c\\u8bbe\\u7acb\\u7684\\uff0c\\u5728\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u8fc7\\u5931\\u6216\\u88ab\\u4ee3\\u7406\\u4eba\\u4e0e\\u65e0\\u6743\\u4ee3\\u7406\\u4eba\\u4e4b\\u95f4\\u5b58\\u5728\\u7279\\u6b8a\\u5173\\u7cfb\\u7684\\u60c5\\u51b5\\u4e0b\\uff0c\\u6cd5\\u5f8b\\u5f3a\\u5236\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u5176\\u6cd5\\u5f8b\\u540e\\u679c\\u3002\\u5728\\u6211\\u56fd\\uff0c\\u300a\\u4e2d\\u534e\\u4eba\\u6c11\\u5171\\u548c\\u56fd\\u6c11\\u6cd5\\u603b\\u5219\\u300b\\u7b2c\\u4e00\\u767e\\u4e03\\u5341\\u4e8c\\u6761\\u89c4\\u5b9a\\u4e86\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u76f8\\u5173\\u5185\\u5bb9\\u3002"}, {"model": "gpt_3", "output": "\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6307\\u5f53\\u4e8b\\u4eba\\u901a\\u8fc7\\u4e00\\u79cd\\u660e\\u793a\\u7684\\u65b9\\u5f0f\\uff0c\\u59d4\\u6258\\u4ed6\\u4eba\\u4ee3\\u4e3a\\u8fdb\\u884c\\u6cd5\\u5f8b\\u884c\\u4e3a\\u7684\\u4e00\\u79cd\\u5f62\\u5f0f\\u3002\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6307\\u4ee3\\u7406\\u4eba\\u6839\\u636e\\u5f53\\u4e8b\\u4eba\\u7684\\u6388\\u6743\\uff0c\\u4ee5\\u81ea\\u5df1\\u7684\\u540d\\u4e49\\u4e0e\\u7b2c\\u4e09\\u4eba\\u8fdb\\u884c\\u4ea4\\u6d89\\u3001\\u7b7e\\u8ba2\\u5408\\u540c\\u6216\\u8fdb\\u884c\\u5176\\u4ed6\\u6cd5\\u5f8b\\u884c\\u4e3a\\u3002\\u8868\\u89c1\\u4ee3\\u7406\\u9700\\u8981\\u6ee1\\u8db3\\u4e00\\u5b9a\\u7684\\u6761\\u4ef6\\uff0c\\u5305\\u62ec\\u4ee3\\u7406\\u4eba\\u5fc5\\u987b\\u5177\\u5907\\u5b8c\\u5168\\u6c11\\u4e8b\\u884c\\u4e3a\\u80fd\\u529b\\uff0c\\u4ee3\\u7406\\u884c\\u4e3a\\u5fc5\\u987b\\u7b26\\u5408\\u6cd5\\u5f8b\\u89c4\\u5b9a\\u7684\\u8303\\u56f4\\u548c\\u76ee\\u7684\\uff0c\\u5e76\\u4e14\\u4ee3\\u7406\\u4eba\\u5fc5\\u987b\\u6309\\u7167\\u5f53\\u4e8b\\u4eba\\u7684\\u610f\\u601d\\u8fdb\\u884c\\u4ee3\\u7406\\u3002\\u901a\\u8fc7\\u8868\\u89c1\\u4ee3\\u7406\\uff0c\\u5f53\\u4e8b\\u4eba\\u53ef\\u4ee5\\u901a\\u8fc7\\u4ee3\\u7406\\u4eba\\u6765\\u5b9e\\u73b0\\u81ea\\u5df1\\u7684\\u6743\\u76ca\\uff0c\\u540c\\u65f6\\u4e5f\\u9700\\u8981\\u627f\\u62c5\\u4ee3\\u7406\\u884c\\u4e3a\\u6240\\u4ea7\\u751f\\u7684\\u6cd5\\u5f8b\\u540e\\u679c\\u3002"}, {"model": "minimax", "output": "\\u8868\\u89c1\\u4ee3\\u7406\\uff0c\\u662f\\u6307\\u884c\\u4e3a\\u4eba\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\u3001\\u8d85\\u8d8a\\u4ee3\\u7406\\u6743\\u6216\\u8005\\u4ee3\\u7406\\u6743\\u7ec8\\u6b62\\u540e\\uff0c\\u4ecd\\u7136\\u5b9e\\u65bd\\u4ee3\\u7406\\u884c\\u4e3a\\uff0c\\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u76f8\\u4fe1\\u884c\\u4e3a\\u4eba\\u6709\\u4ee3\\u7406\\u6743\\u7684\\uff0c\\u4ee3\\u7406\\u884c\\u4e3a\\u6709\\u6548\\u3002\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u4ee3\\u7406\\u5236\\u5ea6\\u7684\\u4e00\\u79cd\\u4f8b\\u5916\\uff0c\\u65e8\\u5728\\u4fdd\\u62a4\\u5584\\u610f\\u76f8\\u5bf9\\u4eba\\u7684\\u5408\\u6cd5\\u6743\\u76ca\\uff0c\\u7ef4\\u62a4\\u4ea4\\u6613\\u5b89\\u5168\\u3002\\n\\n\\u6839\\u636e\\u300a\\u4e2d\\u534e\\u4eba\\u6c11\\u5171\\u548c\\u56fd\\u6c11\\u6cd5\\u5178\\u300b\\u7b2c\\u4e00\\u767e\\u4e03\\u5341\\u4e8c\\u6761\\u89c4\\u5b9a\\uff1a\\u201c\\u884c\\u4e3a\\u4eba\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\u3001\\u8d85\\u8d8a\\u4ee3\\u7406\\u6743\\u6216\\u8005\\u4ee3\\u7406\\u6743\\u7ec8\\u6b62\\u540e\\uff0c\\u4ecd\\u7136\\u5b9e\\u65bd\\u4ee3\\u7406\\u884c\\u4e3a\\uff0c\\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u76f8\\u4fe1\\u884c\\u4e3a\\u4eba\\u6709\\u4ee3\\u7406\\u6743\\u7684\\uff0c\\u4ee3\\u7406\\u884c\\u4e3a\\u6709\\u6548\\u3002\\u201d\\n\\n\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u6784\\u6210\\u8981\\u4ef6\\u5305\\u62ec\\uff1a\\n1. \\u884c\\u4e3a\\u4eba\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\u3001\\u8d85\\u8d8a\\u4ee3\\u7406\\u6743\\u6216\\u8005\\u4ee3\\u7406\\u6743\\u7ec8\\u6b62\\uff1b\\n2. \\u884c\\u4e3a\\u4eba\\u4ee5\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u540d\\u4e49\\u8fdb\\u884c\\u6c11\\u4e8b\\u6d3b\\u52a8\\uff1b\\n3. \\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u76f8\\u4fe1\\u884c\\u4e3a\\u4eba\\u6709\\u4ee3\\u7406\\u6743\\uff1b\\n4. \\u76f8\\u5bf9\\u4eba\\u57fa\\u4e8e\\u8fd9\\u79cd\\u76f8\\u4fe1\\u800c\\u4e0e\\u884c\\u4e3a\\u4eba\\u8fdb\\u884c\\u6c11\\u4e8b\\u6d3b\\u52a8\\u3002\\n\\n\\u5728\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u60c5\\u51b5\\u4e0b\\uff0c\\u88ab\\u4ee3\\u7406\\u4eba\\u5e94\\u5f53\\u5bf9\\u76f8\\u5bf9\\u4eba\\u7684\\u635f\\u5931\\u627f\\u62c5\\u6c11\\u4e8b\\u8d23\\u4efb\\u3002\\u5982\\u679c\\u88ab\\u4ee3\\u7406\\u4eba\\u56e0\\u6b64\\u906d\\u53d7\\u635f\\u5931\\uff0c\\u53ef\\u4ee5\\u5411\\u65e0\\u6743\\u4ee3\\u7406\\u4eba\\u8ffd\\u507f\\u3002"}, {"model": "tongyi", "output": "\\u5728\\u6c11\\u6cd5\\u4e2d\\uff0c\\u201c\\u8868\\u89c1\\u4ee3\\u7406\\u201d\\u662f\\u6307\\u5f53\\u4e8b\\u4eba\\u867d\\u7136\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u662f\\u7531\\u4e8e\\u7b2c\\u4e09\\u4eba\\u7684\\u8bef\\u89e3\\uff0c\\u800c\\u4f7f\\u5f97\\u7b2c\\u4e09\\u4eba\\u76f8\\u4fe1\\u5176\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u5e76\\u4e0e\\u5176\\u8fdb\\u884c\\u4e86\\u6c11\\u4e8b\\u6d3b\\u52a8\\u3002\\u5728\\u8fd9\\u79cd\\u60c5\\u51b5\\u4e0b\\uff0c\\u5982\\u679c\\u4ee3\\u7406\\u4eba\\u548c\\u88ab\\u4ee3\\u7406\\u4eba\\u4e4b\\u95f4\\u7684\\u884c\\u4e3a\\u6784\\u6210\\u4e86\\u771f\\u5b9e\\u7684\\u4ee3\\u7406\\u5173\\u7cfb\\uff0c\\u90a3\\u4e48\\u8868\\u89c1\\u4ee3\\u7406\\u5c31\\u662f\\u6709\\u6548\\u7684\\uff0c\\u7b2c\\u4e09\\u4eba\\u53ef\\u4ee5\\u4f9d\\u636e\\u8868\\u89c1\\u4ee3\\u7406\\u5173\\u7cfb\\u5411\\u88ab\\u4ee3\\u7406\\u4eba\\u4e3b\\u5f20\\u6743\\u5229\\u3002"}, {"model": "wenxinyiyan", "output": "\\u6c11\\u6cd5\\u4e2d\\u7684\\u201c\\u8868\\u89c1\\u4ee3\\u7406\\u201d\\u662f\\u6307\\u884c\\u4e3a\\u4eba\\u867d\\u65e0\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u7531\\u4e8e\\u672c\\u4eba\\u7684\\u884c\\u4e3a\\uff0c\\u9020\\u6210\\u4e86\\u8db3\\u4ee5\\u4f7f\\u5584\\u610f\\u7b2c\\u4e09\\u4eba\\u76f8\\u4fe1\\u5176\\u6709\\u4ee3\\u7406\\u6743\\u7684\\u8868\\u8c61\\uff0c\\u800c\\u4e0e\\u5584\\u610f\\u7b2c\\u4e09\\u4eba\\u8fdb\\u884c\\u7684\\u3001\\u7531\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u6cd5\\u5f8b\\u540e\\u679c\\u7684\\u4ee3\\u7406\\u884c\\u4e3a\\u3002\\n\\u8868\\u89c1\\u4ee3\\u7406\\u5b9e\\u8d28\\u4e0a\\u662f\\u65e0\\u6743\\u4ee3\\u7406\\uff0c\\u5373\\u884c\\u4e3a\\u4eba\\u4e8b\\u5b9e\\u4e0a\\u5e76\\u65e0\\u4ee3\\u7406\\u6743\\u3002\\u4f46\\u7531\\u4e8e\\u672c\\u4eba\\u7684\\u884c\\u4e3a\\uff0c\\u9020\\u6210\\u4e86\\u8db3\\u4ee5\\u4f7f\\u5584\\u610f\\u7b2c\\u4e09\\u4eba\\u76f8\\u4fe1\\u5176\\u6709\\u4ee3\\u7406\\u6743\\u7684\\u8868\\u8c61\\u3002\\u6839\\u636e\\u6cd5\\u5f8b\\u7684\\u89c4\\u5b9a\\uff0c\\u884c\\u4e3a\\u4eba\\u7684\\u4ee3\\u7406\\u884c\\u4e3a\\u6709\\u6548\\uff0c\\u7b2c\\u4e09\\u4eba\\u4e0e\\u4e4b\\u53d1\\u751f\\u6cd5\\u5f8b\\u5173\\u7cfb\\u7684\\u540e\\u679c\\u7531\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u3002\\n\\u6839\\u636e\\u300a\\u6c11\\u6cd5\\u901a\\u5219\\u300b\\u7684\\u89c4\\u5b9a\\uff0c\\u5982\\u679c\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u4e0d\\u8fdd\\u6cd5\\uff0c\\u5219\\u8be5\\u884c\\u4e3a\\u6709\\u6548\\u3002\\u5982\\u679c\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u8fdd\\u6cd5\\uff0c\\u5219\\u8be5\\u884c\\u4e3a\\u65e0\\u6548\\u3002\\u5982\\u679c\\u88ab\\u4ee3\\u7406\\u4eba\\u7684\\u884c\\u4e3a\\u4e0d\\u5408\\u6cd5\\uff0c\\u4f46\\u7b2c\\u4e09\\u4eba\\u4e0d\\u77e5\\u9053\\u6216\\u8005\\u5e94\\u5f53\\u77e5\\u9053\\u800c\\u6ca1\\u6709\\u77e5\\u9053\\uff0c\\u5219\\u8be5\\u884c\\u4e3a\\u6709\\u6548\\u3002\\u5982\\u679c\\u7b2c\\u4e09\\u4eba\\u77e5\\u9053\\u6216\\u8005\\u5e94\\u5f53\\u77e5\\u9053\\u800c\\u6ca1\\u6709\\u77e5\\u9053\\uff0c\\u5219\\u8be5\\u884c\\u4e3a\\u65e0\\u6548\\u3002"}, {"model": "xunfei", "output": "\\u8868\\u89c1\\u4ee3\\u7406\\u662f\\u6307\\u884c\\u4e3a\\u4eba\\u867d\\u65e0\\u4ee3\\u7406\\u6743\\u800c\\u5b9e\\u65bd\\u4ee3\\u7406\\u884c\\u4e3a\\uff0c\\u5982\\u679c\\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u76f8\\u4fe1\\u5176\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u8be5\\u4ee3\\u7406\\u884c\\u4e3a\\u6709\\u6548\\u3002\\u300a\\u6c11\\u6cd5\\u5178\\u300b\\u7b2c\\u4e00\\u767e\\u4e03\\u5341\\u4e8c\\u6761\\u89c4\\u5b9a\\u4e86\\u8868\\u89c1\\u4ee3\\u7406\\u5236\\u5ea6\\u3002 \\n\\n\\u5728\\u603b\\u5219\\u7f16\\u4e2d\\uff0c\\u8868\\u89c1\\u4ee3\\u7406\\u7684\\u5b9a\\u4e49\\u4e3a\\uff1a\\u6240\\u8c13\\u8868\\u89c1\\u4ee3\\u7406\\uff0c\\u662f\\u6307\\u867d\\u7136\\u884c\\u4e3a\\u4eba\\u4e8b\\u5b9e\\u4e0a\\u6ca1\\u6709\\u4ee3\\u7406\\u6743\\uff0c\\u4f46\\u76f8\\u5bf9\\u4eba\\u6709\\u7406\\u7531\\u8ba4\\u4e3a\\u884c\\u4e3a\\u4eba\\u6709\\u4ee3\\u7406\\u6743\\u800c\\u4e0e\\u5176\\u8fdb\\u884c\\u6cd5\\u5f8b\\u884c\\u4e3a\\uff0c\\u5176\\u884c\\u4e3a\\u7684\\u6cd5\\u5f8b\\u540e\\u679c\\u7531\\u88ab\\u4ee3\\u7406\\u4eba\\u627f\\u62c5\\u7684\\u4ee3\\u7406\\u3002"}]',
                    'annotator': 'self_check_gpt4',
                    'multi_rank_per_instruction': "[\n    {'model': 'glm', 'rank': 1},\n    {'model': 'minimax', 'rank': 2},\n    {'model': '360', 'rank': 3},\n    {'model': 'baichuan', 'rank': 4},\n    {'model': 'claude', 'rank': 5},\n    {'model': 'gpt_3', 'rank': 6},\n    {'model': 'tongyi', 'rank': 7},\n    {'model': 'wenxinyiyan', 'rank': 8},\n    {'model': 'xunfei', 'rank': 9}\n]",
                    'price_per_example': 0.27636, 'time_per_example': 7.895266056060791}]


if __name__ == "__main__":
    # purify_dataset()
    # fire.Fire(ALL_FUNCTIONS)
    main()
    # a_test_func()
    # extract_per_instructions_preference()
    # generate_test_dataset()
