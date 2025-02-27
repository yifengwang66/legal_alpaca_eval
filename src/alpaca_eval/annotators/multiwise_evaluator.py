import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from .. import utils
from .base import BaseAnnotator, BaseAnnotatorJSON, SingleAnnotator

__all__ = ["MultiwiseAnnotator", "SingleMultiwiseAnnotator"]

PAIRWISE_ADDED_DOCSTRING = """

    p_label_flip : float, optional
        Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
        2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
        for training. You can set this later on using `set_noise`.

    input_keys : sequence of str, optional
        Keys use to distinguish inputs.

    output_keys : sequence of str, optional
        Keys use to distinguish outputs.

    Notes
    -----
    There are three main functions for annotations depending on how the outputs to compare are given:
        - annotate_pairs: annotate a sequence of examples that contain the pair of outputs `"output_1"` and `"output_2"`
        - annotate_samples: annotate a sequence of examples that contain `"output"` from which we will sample a pair of
            outputs. Useful for collecting pairwise preferences for RLHF.
        - annotate_head2head: annotate a pair of sequence of outputs, each containing `"output"` which will be merged
            into a single sequence of paired outputs. Useful for evaluation against a reference.
    """


class MultiwiseAnnotatorLocal(BaseAnnotator):
    __doc__ = BaseAnnotator.__doc__.replace("Base class", "Class") + PAIRWISE_ADDED_DOCSTRING

    def __init__(
            self,
            *args,
            input_keys: Sequence[str] = ("instruction",),
            output_keys: Sequence[str] = ("multi_output", "reference_output"),
            p_label_flip: Optional[float] = None,
            **kwargs,
    ):
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        super().__init__(*args, **kwargs, primary_keys=self.input_keys + self.output_keys)
        self.p_label_flip = p_label_flip

    @property
    def SingleAnnotator(self) -> Type["SingleAnnotator"]:
        return SingleMultiwiseAnnotator

    @property
    def annotation_key(self) -> str:
        return "multi_rank_per_instruction"

    @property
    def random_seed_keys(self) -> list[str]:
        return list(self.input_keys)

    def annotate_samples(
            self,
            all_outputs: utils.AnyData,
            keys_to_sample_output_2: Optional[Sequence] = None,
            is_unique_instructions: bool = True,
            p_label_flip: Optional[float] = None,
            is_multisample_list: bool = True,
            **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Sample pairs of outputs from a sequence of examples and annotate them.

        Parameters
        ----------
        all_outputs : list of dict or pd.DataFrame or datasets.Dataset
            All examples from which we will sample a pair of outputs to annotate. Each dictionary (or row) should
            contain all of `self.input_keys` and `keys_to_sample_output_2` and `"output"`.

        keys_to_sample_output_2 : tuple of str, optional
            Keys to use to sample paired `"output_2"` to compare to the current `"output"` which will become
            `"output_1"`. If `None` it uses `self.input_keys`.

        is_unique_instructions : bool, optional
            Whether to deduplicate the instructions such that there is only one pair per instruction. If False
            there will be as many pairs as there are outputs for each instruction.

        p_label_flip : float, optional
            Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
            2*p_label_flip of independent coin flip). If None, will use `self.p_label_flip`.

        is_multisample_list : bool, optional
            If True `all_outputs` is a list of examples (dictionary) and each example has an `"output"` column
            containing
            a list of all multi samples. If False `"output"` contains a single output but each element in the list is a
            different (instruction, output) pair with potentially the same instruction.

        decoding_kwargs :
            Additional arguments to pass to the decoder.
        """

        all_outputs = utils.convert_to_dataframe(all_outputs)

        if is_multisample_list:
            all_outputs = all_outputs.explode("output").reset_index().rename(columns={"index": "sample_id"})
            all_outputs["sample_id"] = all_outputs.groupby("sample_id").cumcount()

        if keys_to_sample_output_2 is None:
            keys_to_sample_output_2 = self.input_keys
        keys_to_sample_output_2 = list(keys_to_sample_output_2)

        n_pre_drop = len(all_outputs)

        # set output to be unique for each keys_to_sample_output_2
        df_to_annotate = (
            all_outputs.groupby(keys_to_sample_output_2)
            .apply(lambda x: x.drop_duplicates(["output"]))
            .reset_index(drop=True)
            .rename(columns={"output": "output_1"})
        )

        if len(df_to_annotate) != n_pre_drop:
            logging.warning(
                f"Filtered rows because of duplicate outputs for the same keys_to_sample_output_2="
                f"{keys_to_sample_output_2}. {n_pre_drop} -> {len(df_to_annotate)}"
            )

        # sample an output 2 for each output 1 that are different
        df_to_annotate["output_2"] = df_to_annotate.groupby(list(keys_to_sample_output_2))["output_1"].transform(
            lambda x: utils.random_derangement(x.values, seed=self.seed)
        )

        if is_unique_instructions:
            n_pre_dedup = len(df_to_annotate)
            df_to_annotate = df_to_annotate.drop_duplicates(subset=self.input_keys)
            if len(df_to_annotate) != n_pre_dedup:
                logging.info(f"Filtered unique instruction/input pairs: {n_pre_dedup} -> {len(df_to_annotate)}")

        if p_label_flip is not None:
            old_p_label_flip = self.p_label_flip
            self.set_noise(p_label_flip)

        try:
            annotated = self.__call__(df_to_annotate, **decoding_kwargs)
        finally:
            # reset even if there is an error
            if p_label_flip is not None:
                self.set_noise(old_p_label_flip)

        return annotated

    def annotate_reference2multi(
            self,
            reference_outputs: Union[Sequence[dict[str, Any]], pd.DataFrame],
            models_outputs: Union[Sequence[dict[str, Any]], pd.DataFrame],
            keys_to_merge: Optional[Sequence[str]] = None,
            is_ordered: bool = False,
            **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """reference-2-multi comparison between two sequence of outputs.

        Parameters
        ----------
        reference_outputs : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.
            `"output"` will become `"reference_output"`.

        models_outputs : list of dict or dataframe
            Second  to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.

        keys_to_merge : tuple of str, optional
            Keys to use to merge the two sequences of outputs. If None uses `self.input_keys`

        is_ordered : bool, optional
            Whether the two sequences of outputs are in matching order. If not we will be merging based on
            `keys_to_merge`, which means that the outputs can actually be shorter than the inputs (if some outputs
            are not found in the other sequence) or longer (if some outputs are duplicated in both sequences =>
            set cartesian products).

        decoding_kwargs :
            Additional arguments to pass to `fn_completions`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `keys_to_merge`, `"output_1"`, `"output_2"`, and
            `"preference"`. Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2
            is preferred.
        """
        if keys_to_merge is None:
            keys_to_merge = self.input_keys

        keys_to_merge = list(keys_to_merge)

        reference_outputs = utils.convert_to_dataframe(reference_outputs)
        reference_outputs.rename(columns={"output": "reference_output"}, inplace=True)

        models_outputs = utils.convert_to_dataframe(models_outputs)
        new_model_outputs = models_outputs.pivot_table(index='instruction', columns='generator', values='output',
                                  aggfunc=lambda x: x)
        models_outputs = pd.merge(models_outputs[['instruction']].drop_duplicates(), new_model_outputs, on="instruction")
        models_outputs_json = json.loads(models_outputs.to_json(orient='records'))
        multi_outputs = []
        for item in models_outputs_json:
            answer_for_one_instruction = []
            for key, value in item.items():
                if key == 'instruction':
                    continue
                answer_for_one_instruction.append({
                    "model": key,
                    "output": value
                })
            multi_outputs.append(json.dumps(answer_for_one_instruction))
        models_outputs['multi_output'] = multi_outputs

        if is_ordered:
            reference_outputs = reference_outputs.copy()
            models_outputs = models_outputs.copy()
            reference_outputs["tmp_idx"] = range(len(reference_outputs))
            models_outputs["tmp_idx"] = range(len(models_outputs))
            keys_to_merge += ["tmp_idx"]  # add a temporary index to merge on

        # find all the columns that are in both
        other_same_cols = [k for k in reference_outputs.columns if k in models_outputs and k not in keys_to_merge]

        df_to_annotate = pd.merge(
            reference_outputs,
            models_outputs,
            on=keys_to_merge,
        )

        for c in other_same_cols:
            # if the columns are the same, we can drop the _2
            if df_to_annotate[c + "_1"].equals(df_to_annotate[c + "_2"]):
                df_to_annotate = df_to_annotate.drop(columns=c + "_2").rename(columns={c + "_1": c})

        if is_ordered:
            df_to_annotate = df_to_annotate.drop(columns="tmp_idx")
        else:
            # if you are taking the cartesian product, you can have undesired duplicates
            df_to_annotate = df_to_annotate.drop_duplicates()

            if not (len(reference_outputs) == len(models_outputs) == len(df_to_annotate)):
                logging.warning(
                    f"The length of outputs before and after merge are not the same. We have len(reference_outputs)=="
                    f"{len(reference_outputs)}, len(models_outputs)=={len(models_outputs)}, and len(df_annotated)=={len(df_to_annotate)}."
                    f" This means that there are missing examples or duplicates. We are taking a SQL inner join."
                )

        out = self.__call__(df_to_annotate, **decoding_kwargs)

        return out

    def annotate_pairs(
            self,
            to_annotate: Union[Sequence[dict[str, Any]], pd.DataFrame],
            **decoding_kwargs,
    ) -> list[dict[str, Any]]:
        """Annotates the given examples, which contain both `"output_1"` and `"output_2"` keys.

        Parameters
        ----------
        to_annotate : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `self.primary_keys`.

        **decoding_kwargs :
            Additional arguments to pass to `fn_completions`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `self.primary_keys` and `"preference"`.
            Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2 is preferred.
        """
        # `annotate_pairs` is used for backward compatibility
        return self.__call__(to_annotate, **decoding_kwargs)

    def set_noise(self, p_label_flip: float):
        """Set the noise level for the annotators.

        Parameters
        ----------
        p_label_flip : float, optional
            Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
            2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
            for training.
        """
        self.p_label_flip = p_label_flip

    def _preprocess(self, to_annotate: utils.AnyData) -> pd.DataFrame:
        # same as preprocess but with potential random noising and dealing with eauality

        df_to_annotate = super()._preprocess(to_annotate)

        # 1. adds random noise => avoids annotating examples that will be noised out.
        if self.p_label_flip:
            logging.info(f"Adding random noise to the labels p_label_flip={self.p_label_flip}.")
            # if you have 25% change of flipping the label, you have 50% chance of selecting random label
            p_noise = self.p_label_flip * 2
            noisy_preference = df_to_annotate.apply(
                # we add "noisy_label" at the beginning to use ~independent seeds between tasks
                lambda x: utils.random_seeded_choice(  # seed on inputs for reproducibility
                    seed="noisy_preference" + "".join(x[self.random_seed_keys]) + str(self.seed),
                    choices=[np.nan, 1, 2],
                    weights=[1 - p_noise, self.p_label_flip, self.p_label_flip],
                ),
                axis=1,
            )
            df_to_annotate["is_noisy_label"] = ~noisy_preference.isna()
            # keeps previously annotated examples when you did not add noise
            df_to_annotate[self.annotation_key] = np.where(
                df_to_annotate["is_noisy_label"],
                noisy_preference,
                df_to_annotate[self.annotation_key],
            )
        print("MultiwiseAnnotatorLocal._preprocess df_to_annotate columns", df_to_annotate.columns)
        print("MultiwiseAnnotatorLocal._preprocess df_to_annotate", df_to_annotate.head())


        return df_to_annotate

    def _filter_annotations_before_storing(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        if "is_noisy_label" in df_annotated.columns:
            # don't store noisy labels
            df_annotated = df_annotated.query("is_noisy_label == False").drop(columns=["is_noisy_label"])

        df_annotated = super()._filter_annotations_before_storing(df_annotated)
        return df_annotated


# Note: we separate local and json to make it easier to inherit e.g. for having a database version
class MultiwiseAnnotator(MultiwiseAnnotatorLocal, BaseAnnotatorJSON):
    __doc__ = BaseAnnotatorJSON.__doc__.replace("Base class", "Class") + PAIRWISE_ADDED_DOCSTRING


class SingleMultiwiseAnnotator(SingleAnnotator):
    __doc__ = (
            SingleAnnotator.__doc__.replace(
                "A helper class for a single auto annotator.",
                "A helper class for a single pairwise auto annotator.",
            )
            + """
    is_randomize_output_order : bool
        Whether to randomize output_1, output_2 when formatting.

    random_seed_keys : str
        The column to use to seed the randomization of output_1, output_2.
    """
    )

    def __init__(
            self,
            *args,
            annotation_column: str = "preference",
            random_seed_column: Sequence[str] = ("instruction",),
            processors_to_kwargs: Optional[dict[str, dict]] = None,
            **kwargs,
    ):
        processors_to_kwargs = processors_to_kwargs or {}

        super().__init__(
            *args, annotation_column=annotation_column, processors_to_kwargs=processors_to_kwargs, **kwargs
        )
        self.random_seed_column = list(random_seed_column)

    def _postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        df_annotated = super()._postprocess(df_annotated)

        return df_annotated
