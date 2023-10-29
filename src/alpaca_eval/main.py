import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Union

import fire
import pandas as pd

from alpaca_eval import analyze, annotators, constants, decoders, metrics, utils
from alpaca_eval.constants import MODEL_NAME_VEIL
from alpaca_eval.types import AnyData, AnyPath
import json

CUR_DIR = Path(__file__).parent
DEFAULT_CONFIGS = "alpaca_eval_gpt4"

__all__ = [
    "evaluate",
    "evaluate_from_model",
    "analyze_evaluators",
    "make_leaderboard",
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

    # 获取每个模型的结果输出路径
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


def evaluate_round_robin(
        model_outputs_1: Optional[Union[AnyPath, AnyData, Callable]] = None,
        model_outputs_2: Optional[Union[AnyPath, AnyData, Callable]] = None,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        name_1: Optional[str] = None,
        name_2: Optional[str] = None,
        round_count: Optional[int] = 1,
        precomputed_round_rank: Optional[Union[str, AnyPath, AnyData]] = "auto",
        current_leaderboard_mode: str = "community",
        fn_metric: Union[str, callable] = "round_rank_evaluate",
        is_cache_leaderboard: Optional[bool] = None,
        max_instances: Optional[int] = None,
        annotation_kwargs: Optional[dict[str, Any]] = None,
        Annotator=annotators.RoundRobinAnnotator,
        **annotator_kwargs,
):
    if (
            isinstance(current_leaderboard_mode, str)
            and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    round_rank, round_rank_veil, precomputed_round_rank = utils.get_precomputed_round_rank(precomputed_round_rank)
    print("round_rank:", round_rank.columns)
    print("round_rank:", round_rank.head())
    annotations = None

    if model_outputs_1 is not None and model_outputs_2 is not None:
        model_outputs_1 = utils.load_or_convert_to_dataframe(model_outputs_1)
        model_outputs_2 = utils.load_or_convert_to_dataframe(model_outputs_2)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        col_name_1 = f"r{round_count}_{name_1}"
        col_name_2 = f"r{round_count}_{name_2}"
        col_name_1_veil = f"r{round_count}_{MODEL_NAME_VEIL[name_1]}"
        col_name_2_veil = f"r{round_count}_{MODEL_NAME_VEIL[name_2]}"

        logging.info(f"Evaluating the {name_1} and {name_2} outputs.")

        if max_instances is not None:
            model_outputs_1 = model_outputs_1[:max_instances]
            model_outputs_2 = model_outputs_2[:max_instances]
            reference_outputs = reference_outputs[:max_instances]

        annotator = Annotator(annotators_config=annotators_config, annotation_type=str, **annotator_kwargs)
        annotations = annotator.annotate_round_robin(
            reference_output=reference_outputs,
            outputs_1=model_outputs_1,
            outputs_2=model_outputs_2,
            **annotation_kwargs
        )

        print("evaluate annotations:", annotations)
        if isinstance(fn_metric, str):
            fn_metric = getattr(metrics, fn_metric)

        instructions = []
        instructions_type = []
        reference_answer = []
        outputs_1 = []
        outputs_2 = []
        preferences = []
        rank_reasons = []
        error_decoded_res = []
        for annotation in annotations:
            if round_rank.empty:
                instructions_type.append(annotation["type"])
                instructions.append(annotation["instruction"])
                reference_answer.append((annotation["reference_output"]))
            outputs_1.append(annotation["output_1"])
            outputs_2.append(annotation["output_2"])
            try:
                rank_res = json.loads(annotation["rank_res"])
            except Exception as e:
                # json 解析出错，通常是由于GPT-4返回的内容不是直接的JSON，当题目不好进行排名时GPT-4会进行说明
                print("json decoded error:", annotation["rank_res"])
                error_decoded_res.append(annotation)
                rank_res = json.loads(
                    annotation["rank_res"][
                    annotation["rank_res"].find("{"): annotation["rank_res"].rfind("}") + 1].replace("\\\n", "\\n"))
            # 给出这样的排名的原因
            rank_reason = rank_res["reason"]
            preferences.append(rank_res["rank_list"])
            rank_reasons.append(rank_reason)

        # 记录发生JSON解析错误的题目
        if len(error_decoded_res) != 0:
            if os.path.exists("error_decoded_res.json"):
                with open("error_decoded_res.json", "r") as f:
                    pre_error_decoded_res = json.load(f)
            else:
                pre_error_decoded_res = []
            error_decoded_res += pre_error_decoded_res
            with open("error_decoded_res.json", "w") as f:
                json.dump(error_decoded_res, f)

        if round_rank.empty:
            round_rank["type"] = instructions_type
            round_rank["instruction"] = instructions
            round_rank["reference_output"] = reference_answer
        if round_rank_veil.empty:
            round_rank_veil["type"] = instructions_type
            round_rank_veil["instruction"] = instructions
            round_rank_veil["reference_output"] = reference_answer
        round_rank[col_name_1] = outputs_1
        round_rank[col_name_2] = outputs_2
        round_rank[f"r{round_count}_{name_1}_{name_2}_rank_reason"] = rank_reasons
        round_rank_veil[col_name_1_veil] = outputs_1
        round_rank_veil[col_name_2_veil] = outputs_2
        win_name_list, result_dic, models_score_record = fn_metric(preferences=preferences)
        print("win_name_list:", win_name_list)
        print("result_idc:", result_dic)
        round_rank[f"r{round_count}_{name_1}_score"] = models_score_record[name_1]
        round_rank[f"r{round_count}_{name_2}_score"] = models_score_record[name_2]
        round_rank[f"r{round_count}_{name_1}_vs_{name_2}"] = win_name_list
        round_rank[f"r{round_count}_{name_1}_vs_{name_2}_human"] = [None] * len(annotations)
        round_rank_veil[f"r{round_count}_{MODEL_NAME_VEIL[name_1]}_vs_{MODEL_NAME_VEIL[name_2]}_human"] = [None] * len(
            annotations)

        # 人工审核时隐藏模型信息以及机器评比结果
        # if round_rank_veil.empty:
        #     round_rank_veil = round_rank.copy()
        # round_rank_veil.drop(f"r{round_count}_{name_1}_vs_{name_2}", axis=1, inplace=True)
        # for real_name, veil_name in MODEL_NAME_VEIL.items():
        #     pattern = re.compile(re.escape(real_name))
        #     new_columns = [re.sub(pattern, veil_name, col) for col in round_rank_veil.columns]
        #     round_rank_veil.rename(columns=dict(zip(round_rank_veil.columns, new_columns)), inplace=True)

    return round_rank, round_rank_veil, annotations, result_dic


def make_leaderboard_round_robin(
        leaderboard_path: AnyPath,
        annotators_config: AnyPath = DEFAULT_CONFIGS,
        all_model_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAFARM_ALL_OUTPUTS,
        reference_outputs: Union[AnyPath, AnyData, Callable] = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
        fn_add_to_leaderboard: Callable = "evaluate_round_robin",
        leaderboard_mode: str = "verified",
        **kwargs,
):
    if isinstance(fn_add_to_leaderboard, str):
        fn_add_to_leaderboard = globals()[fn_add_to_leaderboard]

    all_model_outputs = utils.load_or_convert_to_dataframe(all_model_outputs)
    if "generator" not in all_model_outputs.columns:
        raise ValueError(f"all_model_outputs should have a column 'generator' with the name of the model.")

    # 当前轮次对比的结果存放地址
    round_result_path = (Path(leaderboard_path).parent / "round_result")
    round_result_path.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    round_result = dict(
        win_group=[],
        lose_group=[]
    )
    generator_list = [model_name for model_name in all_model_outputs["generator"].unique() if model_name != "minimax"]

    random.shuffle(generator_list)
    # 随机分组
    # generator_groups = [generator_list[i:i + 2] for i in range(0, len(generator_list), 2)]
    # generator_groups = [['glm', '360'], ['xunfei', 'claude'], ["wenxinyiyan", "baichuan"], ["tongyi", "gpt-3"]] # round_1 group
    # generator_groups = [['glm', 'xunfei'], ['baichuan', 'tongyi']]  # round_2 win_group
    # generator_groups = [['360', 'claude'], ['wenxinyiyan', 'gpt-3']]  # round_2 loss_group
    # generator_groups = [['glm', 'baichuan']]  # round_3 win_win_group 排1,2名
    # generator_groups = [['xunfei', 'tongyi']]  # round_3 win_loss_group 排3,4名
    # generator_groups = [['claude', 'wenxinyiyan']]  # round_3 loss_win_group 排5,6名
    generator_groups = [['360', 'gpt-3']]  # round_3 loss_loss_group 排7,8名

    # generator_groups = [['glm', '360']] # test_group
    print("make_leaderboard_round_robin generator_groups: ", generator_groups)
    round_count = 3
    round_info = f"round_{round_count}_loss_loss_group"
    precomputed_round_rank = f"{leaderboard_path.split('.')[0]}_{round_info}.{leaderboard_path.split('.')[1]}"
    for group in generator_groups:
        if len(group) == 1:
            continue
        model_output_1 = all_model_outputs[all_model_outputs["generator"] == group[0]]
        model_output_2 = all_model_outputs[all_model_outputs["generator"] == group[1]]
        df_round_rank, df_round_rank_veil, annotations, result_dic = fn_add_to_leaderboard(
            model_outputs_1=model_output_1,
            model_outputs_2=model_output_2,
            reference_outputs=reference_outputs,
            annotators_config=annotators_config,
            name_1=group[0],
            name_2=group[1],
            round_count=round_count,
            precomputed_round_rank=precomputed_round_rank,
            current_leaderboard_mode=leaderboard_mode,
            # max_instances=10,
            **kwargs,
        )
        if annotations is not None:
            all_annotations += annotations
        df_round_rank.to_csv(precomputed_round_rank, encoding="utf-8-sig")
        df_round_rank_veil.to_csv(f"{precomputed_round_rank.split('.')[0]}_veil.{precomputed_round_rank.split('.')[1]}",
                                  encoding="utf-8-sig")
        round_result["win_group"].append(result_dic["win_name"])
        round_result["lose_group"].append(result_dic["lose_name"])
        # 每一对比完之后都记录一下，防止报错中断导致前面比过的也没有记录
        with open(round_result_path / f"{round_info}_result.json", "w", encoding="utf-8") as f:
            json.dump(round_result, f, ensure_ascii=False)
        with open(round_result_path / f"{round_info}_{group[0]}_vs_{group[1]}.json", "w", encoding="utf-8") as f:
            json.dump(result_dic, f, ensure_ascii=False)

    # total_round_rank = utils.load_or_convert_to_dataframe(leaderboard_path)
    # total_round_rank = pd.DataFrame(total_round_rank)

    print('all_annotations:', all_annotations)

    return all_annotations


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
    "make_leaderboard_round_robin": make_leaderboard_round_robin
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


# 人工评测结果处理
def extract_round_result(round_result_path, record_output_path="round_result_diff_record.json"):
    """
    return example:
    {
        "r1": {
            "glm_vs_360": {
                "概念": {
                    "diff_count": int,
                    "diff_rate": float,
                    "total": int,
                },
                "法条": {
                    "diff_count": int,
                    "diff_rate": float,
                    "total": int,
                }
            }
            "wenxinyiyan_vs_baichuan": {...}
        }
        "r2": {...}
    }
    """
    round_result = pd.read_csv(round_result_path)
    veil_name_model = {veil_name: real_name for real_name, veil_name in MODEL_NAME_VEIL.items()}
    try:
        with open(record_output_path, "r", encoding="utf-8") as f:
            round_record = json.load(f)
    except FileNotFoundError:
        round_record = {}

    # 转换人工评测结果
    for col in round_result.columns:
        if "human" in col:
            round_result[col] = [veil_name_model[val] if veil_name_model.get(val) is not None else val for val in
                                 round_result[col]]

    question_type_list = [type_name for type_name in round_result["type"].unique()]
    # 统计人工结果和机器结果的不一致情况
    for type_name in question_type_list:
        type_questions = round_result[round_result["type"] == type_name]
        for col in type_questions.columns:
            if "vs" in col and "human" not in col:  # col: r1_glm_vs_360 r1_glm_vs_360_human
                name_chunk = col.split('_')
                round_name = name_chunk[0]
                model_1, model_2 = name_chunk[1], name_chunk[3]
                if round_record.get(round_name) is None:
                    round_record[round_name] = {}
                if round_record.get(round_name).get(f"{model_1}_vs_{model_2}") is None:
                    round_record[round_name][f"{model_1}_vs_{model_2}"] = {
                        "type_count": 0,
                        "type_avg_diff_rate": 0
                    }
                mr_list = type_questions[col]  # machine_result_list
                hr_list = type_questions[f"{col}_human"]  # human_result_list
                diff_count = pd.Series([0 if mr == hr else 1 for mr, hr in zip(mr_list, hr_list)]).sum()
                diff_rate = round(diff_count / len(type_questions), 4) * 100
                round_record[round_name][f"{model_1}_vs_{model_2}"][type_name] = {
                    "diff_count": int(diff_count),
                    "diff_rate": diff_rate,
                    "total": len(type_questions)
                }
                round_record[round_name][f"{model_1}_vs_{model_2}"]["type_count"] += 1
                round_record[round_name][f"{model_1}_vs_{model_2}"]["type_avg_diff_rate"] = (round_record[round_name][
                                                                                                 f"{model_1}_vs_{model_2}"][
                                                                                                 "type_avg_diff_rate"] * (
                                                                                                     round_record[
                                                                                                         round_name][
                                                                                                         f"{model_1}_vs_{model_2}"][
                                                                                                         "type_count"] - 1) + diff_rate) / \
                                                                                            round_record[round_name][
                                                                                                f"{model_1}_vs_{model_2}"][
                                                                                                "type_count"]

    # 统计人工测评的排名结果
    for col in round_result.columns:
        if "vs" in col and "human" in col:
            human_record = {}
            name_chunk = col.split('_')
            round_name = name_chunk[0]
            model_1, model_2 = name_chunk[1], name_chunk[3]
            result_count = round_result[col].value_counts().to_dict()
            model_1_count = result_count.get(model_1) if result_count.get(model_1) is not None else 0
            model_2_count = result_count.get(model_2) if result_count.get(model_2) is not None else 0
            both_wrong_count = result_count.get("both_wrong") if result_count.get("both_wrong") is not None else 0
            both_right_count = result_count.get("both_right") if result_count.get("both_right") is not None else 0
            total = model_1_count + model_2_count + both_wrong_count + both_right_count
            human_record[model_1] = {
                "win_rate": round((model_1_count + both_right_count) / total, 4) * 100,
                "n_wins": model_1_count,
                "n_draws": model_2_count,
                "both_wrong": both_wrong_count,
                "both_right": both_right_count,
                "n_total": total
            }
            human_record[model_2] = {
                "win_rate": round((model_2_count + both_right_count) / total, 4) * 100,
                "n_wins": model_2_count,
                "n_draws": model_1_count,
                "both_wrong": both_wrong_count,
                "both_right": both_right_count,
                "n_total": total
            }
            human_record["win_name"] = model_1 if human_record[model_1]["win_rate"] > human_record[model_2][
                "win_rate"] else model_2
            human_record["loss_name"] = model_1 if human_record["win_name"] == model_2 else model_2
            with open(f"round_result/{round_name}_{model_1}_vs_{model_2}_human_result.json", "w",
                      encoding="utf-8") as f:
                json.dump(human_record, f, ensure_ascii=False)

    print("round_record:", round_record)
    with open(record_output_path, "w", encoding="utf-8") as f:
        json.dump(round_record, f, ensure_ascii=False)

    round_result.to_csv(round_result_path, encoding="utf-8-sig")
    return


def extract_machine_result(round_result_path):
    round_result = pd.read_csv(round_result_path)
    # 统计机器测评的排名结果
    for col in round_result.columns:
        if "vs" in col and "human" not in col:
            machine_record = {}
            name_chunk = col.split('_')
            round_name = name_chunk[0]
            model_1, model_2 = name_chunk[1], name_chunk[3]
            result_count = round_result[col].value_counts().to_dict()
            model_1_count = result_count.get(model_1) if result_count.get(model_1) is not None else 0
            model_2_count = result_count.get(model_2) if result_count.get(model_2) is not None else 0
            both_wrong_count = result_count.get("both_wrong") if result_count.get("both_wrong") is not None else 0
            both_right_count = result_count.get("both_right") if result_count.get("both_right") is not None else 0
            total = model_1_count + model_2_count + both_wrong_count + both_right_count
            machine_record[model_1] = {
                "win_rate": round((model_1_count + both_right_count) / total, 4) * 100,
                "n_wins": model_1_count,
                "n_draws": model_2_count,
                "both_wrong": both_wrong_count,
                "both_right": both_right_count,
                "n_total": total
            }
            machine_record[model_2] = {
                "win_rate": round((model_2_count + both_right_count) / total, 4) * 100,
                "n_wins": model_2_count,
                "n_draws": model_1_count,
                "both_wrong": both_wrong_count,
                "both_right": both_right_count,
                "n_total": total
            }
            machine_record["win_name"] = model_1 if machine_record[model_1]["win_rate"] > machine_record[model_2][
                "win_rate"] else model_2
            machine_record["loss_name"] = model_1 if machine_record["win_name"] == model_2 else model_2
            with open(f"round_result/round_{round_name[1]}_{model_1}_vs_{model_2}.json", "w",
                      encoding="utf-8") as f:
                json.dump(machine_record, f, ensure_ascii=False)


def extract_multi_compare_per_instruction(annotations):
    all_df = None
    all_rank = None
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
        rank_df = pd.DataFrame([
            {
                "instruction": instruction
            }
        ])
        for per_rank in rank:
            df[f"{per_rank['model']}_output"] = annotation[per_rank['model']]
            df[f"{per_rank['model']}_rank"] = [per_rank['rank']]
            rank_df[f"{per_rank['rank']}"] = [per_rank['model']]
        df["price"] = [price]
        df["time"] = [time]
        if all_df is None and all_rank is None:
            all_df = df
            all_rank = rank_df
            continue
        all_df = pd.concat([all_df, df])
        all_rank = pd.concat([all_rank, rank_df])
    price_sum = all_df["price"].sum()
    time_sum = all_df["time"].sum()
    all_df = all_df.assign(price_sum=price_sum).assign(time_sum=time_sum)
    all_df.to_csv("legal_model_per_instruction_result.csv", encoding="utf-8-sig")
    all_rank.to_csv("legal_model_per_rank.csv", encoding="utf-8-sig")


def purify_dataset():
    with open('legal_model_outputs.json', 'r') as f:
        model_outputs = json.load(f)

    with open('legal_reference_outputs.json', 'r') as f:
        reference_outputs = json.load(f)

    model_outputs = [model_output for model_output in model_outputs if
                     model_output["generator"] != "Peking_chatlaw"]
    reference_outputs = [reference_output for reference_output in reference_outputs if
                         reference_output["output"] != ""]

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
    return


if __name__ == "__main__":
    # purify_dataset()
    # fire.Fire(ALL_FUNCTIONS)
    # main()
    # a_test_func()
    # generate_test_dataset()
    extract_round_result('legal_leaderboard_round_1.csv', "round_result/round_result_diff_record.json")
    # extract_machine_result("legal_leaderboard_round_1.csv")
