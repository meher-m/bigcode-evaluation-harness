from dataclasses import dataclass
import inspect
from pprint import pprint

from . import (apps, codexglue_code_to_text, codexglue_text_to_text, conala,
               concode, ds1000, gsm, humaneval, humanevalplus, humanevalpack,
               instruct_humaneval, instruct_wizard_humaneval, mbpp, mbppplus,
               multiple, parity, python_bugs, quixbugs, recode, santacoder_fim,
               studenteval, mercury)

TASK_REGISTRY = {
    **apps.create_all_tasks(),
    **codexglue_code_to_text.create_all_tasks(),
    **codexglue_text_to_text.create_all_tasks(),
    **multiple.create_all_tasks(),
    "codexglue_code_to_text-python-left": codexglue_code_to_text.LeftCodeToText,
    "conala": conala.Conala,
    "concode": concode.Concode,
    **ds1000.create_all_tasks(),
    **humaneval.create_all_tasks(),
    **humanevalplus.create_all_tasks(),
    **humanevalpack.create_all_tasks(),
    "mbpp": mbpp.MBPP,
    "mbppplus": mbppplus.MBPPPlus,
    "parity": parity.Parity,
    "python_bugs": python_bugs.PythonBugs,
    "quixbugs": quixbugs.QuixBugs,
    "instruct_wizard_humaneval": instruct_wizard_humaneval.HumanEvalWizardCoder,
    **gsm.create_all_tasks(),
    **instruct_humaneval.create_all_tasks(),
    **recode.create_all_tasks(),
    **santacoder_fim.create_all_tasks(),
    "studenteval": studenteval.StudentEval,
    "mercury": mercury.Mercury,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


@dataclass
class NuggetsConfig():
    prompt_quality: int = 2
    add_context: bool = False
    example_idxs: list = None
    examples_path: str = None


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "prompt" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["prompt"] = args.prompt
        if "load_data_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["load_data_path"] = args.load_data_path

        nuggets_config = NuggetsConfig(
            args.humaneval_prompt_quality,
            args.humaneval_add_context,
            args.humaneval_example_idxs,
            args.humaneval_examples_path
        )
        if task_name == "humaneval":
            kwargs["nuggets_config"] = nuggets_config
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
