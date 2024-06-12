import inspect
import json
import os
import warnings

from typing import List


from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = min(self.args.limit, len(dataset) - self.args.limit_start) if self.args.limit else len(dataset)
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues 
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        gen_dict = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,  # curr_sample_idx will added to limit_start to fix indexing
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_generations_path,
        )

        raw_gens_to_save = gen_dict["raw_generations"]
        if self.args.save_results_path:
            # Dump raw generations to a new json to verify that prompt creation is as expected 
            # Dump of the form {'task_id': --, 'generation': --, 'result': --}
            save_raw_gens_path = f"{os.path.splitext(self.args.save_results_path)[0]}_{task_name}_raw_generations.json"

            with open(save_raw_gens_path, "w") as fp:
                json.dump(raw_gens_to_save, fp)
                print(f"raw generation was saved at {save_raw_gens_path}")

        generations = gen_dict["generations"]
        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references

    def evaluate(self, task_name, intermediate_generations=None):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations, references = self.generate_text(task_name, intermediate_generations=intermediate_generations)
        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}.json"
                self.save_json_files(generations, references, save_generations_path, f"references_{task_name}.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")

            results, fine_grain_results = task.process_results(generations, references)

            # Simplify the fine grain results to save task_id, generation, and whether it passed.
            simplified_fine_grain_res = {}
            for task_id in fine_grain_results:
                # The element at each task_id is a single element list. The element is a tuple (0, dict) where the dict contains
                # 'task_id', 'passed', 'result', and 'completion_id' as its keys. Extracting this dictionary to save. 
                simplified_fine_grain_res[task_id] = fine_grain_results[task_id][0][1]

            if self.args.save_results_dir:
                os.makedirs(self.args.save_results_dir, exist_ok=True)
                save_results_path = f"{self.args.save_results_dir}/{task_name}_results.json"

                # Add the generations to the fine grain results and then dump to a new json
                for task_id in simplified_fine_grain_res:
                    simplified_fine_grain_res[task_id]["generation"] = generations[task_id]

                # Dump to a new json that has {'task_id': --, 'generation': --, 'result': --}
                with open(save_results_path, "w") as fp:
                    json.dump(simplified_fine_grain_res, fp)
                    print(f"results with generation were saved at {save_results_path}")

            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
