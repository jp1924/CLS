import json
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

from datasets import load_dataset
from setproctitle import setproctitle

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.utils import is_sagemaker_mp_enabled


@dataclass
class TrainPipelineArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "어떤 attention 연산 방식을 사용할지 결정하는 값, default가 eager임, eager, flash_attention_2, sdpa중 하나 고르셈."
        },
    )
    packing_max_elem: int = field(
        default=10,
        metadata={"help": ""},
    )
    do_packing: bool = field(
        default=True,
        metadata={"help": ""},
    )
    packing_shuffle: bool = field(
        default=True,
        metadata={"help": "packing shuffle"},
    )
    freeze_named_param: List[str] = field(
        default=None,
        metadata={"help": "freeze_named_param"},
    )
    profiling: bool = field(
        default=False,
        metadata={"help": "profiling"},
    )
    profiling_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "profiling_kwargs"},
    )
    config_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    model_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    processor_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )


@dataclass
class CLSTrainingArguments(Seq2SeqTrainingArguments, TrainPipelineArguments):
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    def __post_init__(self):
        if self.output_dir is None:
            raise ValueError("output_dir은 무조건 설정되어 있어야 한다.")

        super().__post_init__()

    @property
    def is_local_process_zero(self) -> bool:
        return self.local_process_index == 0

    @property
    def is_world_process_zero(self) -> bool:
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp  # type: ignore

            return smp.rank() == 0
        else:
            return self.process_index == 0


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: CLSTrainingArguments) -> None:
    def processor(example):
        finish_data_ls = list()
        for data_row in list(zip(*[example[key] for key in example])):
            data_row = {key: value for key, value in zip(example.keys(), data_row)}  # noqa: C416
            situation, disease, emotion = f"[{data_row['situation']}]", f"[{data_row['disease']}]", data_row["emotion"]
            age, gender = f"[{data_row['metadata']['age']}]", f"[{data_row['metadata']['gender']}]"

            if emotion not in model.config.label2id:
                continue

            text = tokenizer.apply_chat_template(
                # [{"role": "system", "content": f"{situation}{disease}{age}{gender}"}, *data_row["conversations"]],
                data_row["conversations"],
                tokenize=False,
            )

            output = tokenizer(text, return_length=True)
            output["labels"] = model.config.label2id[emotion]

            finish_data_ls.append(output)

        return_dict = dict()
        for res in finish_data_ls:
            for key, value in res.items():
                return_dict.setdefault(key, []).append(value)
        return return_dict

    config = AutoConfig.from_pretrained(train_args.model_name_or_path, _attn_implementation="eager")
    model = AutoModelForSequenceClassification.from_pretrained(
        train_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(train_args.model_name_or_path)
    dataset = load_dataset("jp1924/EmotionalDialogueCorpus")

    with train_args.main_process_first():
        dataset = dataset.map(
            processor,
            num_proc=4,
            batched=True,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()


if "__main__" in __name__:
    parser = HfArgumentParser([CLSTrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and train_args.is_world_process_zero:
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)

{
    "a": [1, 2, 3, 4],
    "b": ["1", "2", "3", "4"],
    "c": [1, 2, 3, 4],
}
