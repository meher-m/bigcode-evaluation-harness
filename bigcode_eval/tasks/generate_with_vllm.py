from dataclasses import dataclass
from vllm import LLM, SamplingParams
import humaneval
import json
import time
import s3fs
import aiobotocore.session
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
import os
import argparse

STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]

@dataclass
class NuggetsConfig():
    prompt_quality: int = 2
    add_context: bool = False
    example_idxs: list = None
    examples_path: str = None
    use_chat_template: bool = False


def get_s3fs():
    s3_session = aiobotocore.session.AioSession(profile="ml-worker")
    storage_options = {"session": s3_session}
    fs = s3fs.S3FileSystem(**storage_options)
    return fs


def save_ds_s3(ds, path: str):
    fs = get_s3fs()
    ds.save_to_disk(path, storage_options=fs.storage_options)


def load_ds_s3(path: str):
    fs = get_s3fs()
    dataset = load_from_disk(path, storage_options=fs.storage_options)
    return dataset


def get_dataset():
    
    TASK_REGISTRY = {
        **humaneval.create_all_tasks(),
    }

    kwargs = {}
    nuggets_config = NuggetsConfig(
        prompt_quality=2,
        add_context=False,
        example_idxs=None,
        examples_path="bigcode_eval/tasks/few_shot_examples/public_ots_0.json",
        use_chat_template=False
    )
    kwargs["nuggets_config"] = nuggets_config
    task_name = "humaneval"
    task = TASK_REGISTRY[task_name](**kwargs)

    dataset = task.get_dataset()
    return nuggets_config, dataset


def fewshot_examples(nuggets_config, dataset, tokenizer=None):
    # If arguments passed in as indices, build examples from HumanEval dataset
    # Otherwise, load examples from file
    if nuggets_config.example_idxs:
        examples = ""
        for example_idx in nuggets_config.example_idxs:
            # For now, hard-coding the one shot example to be the last task in the HumanEval dataset
            # Also hard-coding the three sample solutions (good, decent, and bad) here
            sample = dataset["test"][example_idx]
            correct_sol = sample["canonical_solution"]
            really_bad_sol = "    cat: cat cat\n    dog dog dog;\n    return [giraffe if giraffe for giraffe in giraffe]"
            decent_sol = "    lower = 2\n    upper = 8\n    return [i if i % 2 = 0 for i in range(lower, upper)]"
            
            # Create a list of all the different task answers possible for various experiments
            example_task_answers = [really_bad_sol, decent_sol, correct_sol]

            # If add_context is set, add additional context to the prompt. 
            examples += sample["prompt"] + "\n"
            examples += example_task_answers[nuggets_config.prompt_quality] + "\n"

        return examples
    else:
        with open(nuggets_config.examples_path, "r") as file:
            data = json.load(file)
        
        if nuggets_config.use_chat_template:
            examples = ""
            for example in data:
                # Hard coded using only the first turn for now. Should expand to using all the turns. 
                example_to_template = example["messages"][:2]
                examples += tokenizer.apply_chat_template(example_to_template, add_generation_prompt=False, tokenize=False)
        else:
            examples = ""
            # Loop through each few-shot example
            for example in data:
                # Use all turns
                messages = example["messages"]
                for message in messages:
                    examples += message["content"] + "\n"

    return examples


def get_base_prompt(doc, strip_prompt=False):
    # Strip prompt if required
    if strip_prompt:
        return doc['prompt'].strip()
    else:
        return doc['prompt']


def get_prompt(nuggets_config, dataset, doc, tokenizer=None):
    """Builds the prompt for the LM to generate from."""
    base_prompt = get_base_prompt(doc)

    # Cases: few shot with context, few shot without context, zero shot
    use_few_shot = nuggets_config.example_idxs or nuggets_config.examples_path
    start_context = "Implement solutions to the following coding tasks given the function heading:\n"

    prompt = ""
    if nuggets_config.add_context:
        prompt += start_context

    if use_few_shot:
        prompt += fewshot_examples(nuggets_config, dataset, tokenizer)

    # If using chat template, we need to add the user role to the prompt
    if nuggets_config.use_chat_template:
        if not tokenizer:
            raise ValueError("Tokenizer must be provided when using chat template")
        
        base_prompt_to_template = [
            {"role": "user", "content": f"I am a software engineering working in Python. Can you help me implement a function with the following header and description.\n{base_prompt}\n"},
        ]
        base_prompt_templated = tokenizer.apply_chat_template(base_prompt_to_template, add_generation_prompt=True, tokenize=False)

        prompt += base_prompt_templated
    else:
        prompt += base_prompt

    return prompt


def _stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def postprocess_generation(nuggets_config, dataset, generation, idx, tokenizer=None):
    """Defines the postprocessing for a LM generation.
    :param generation: str
        code generation from LM
    :param idx: int
        index of doc in the dataset to which the generation belongs
        (not used for Humaneval-Task)
    """
    # Want to remove one shot example and any context when post processing
    base_prompt = get_base_prompt(dataset[idx])

    ### Templating post_process
    # Find the ``` lines and return what is in between them
    if nuggets_config.use_chat_template:
        try:
            start_idx = generation.index("```")
            new_start_idx = generation[start_idx:].index("\n") + start_idx
            end_idx = generation.find("```", new_start_idx + 1)
            generation = generation[new_start_idx:end_idx].strip()
        except Exception as e:
            raise ValueError("Failed to find '```' in generation") from e
        return generation
    else:
        processed_generation = base_prompt + "\n" + _stop_at_stop_token(generation, STOP_WORDS)
        return processed_generation
        

def generate(nuggets_config, dataset, llm, prompts):
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=16000, stop=STOP_WORDS) #8192)
    
    outputs = llm.generate(prompts, sampling_params)

    processed_results = []
    # Print the outputs.
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        process_gen = postprocess_generation(nuggets_config, dataset, generated_text, idx)
        processed_results.append(process_gen)

    return processed_results


def generate_for_one_shot(args, nuggets_config, llm, dataset):
    tokenizer = None
    if nuggets_config.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            truncation_side="left",
            padding_side="right"
        )

    prompts = []
    i = 0
    for task in dataset:
        prompt = get_prompt(nuggets_config, dataset, task, tokenizer)
        prompts.append(prompt)
        i += 1

    results = generate(nuggets_config, dataset, llm, prompts)
    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate code using VLLM')
    parser.add_argument('--s3_dataset_path', type=str, help='Path to read dataset from S3. Should end in /dataset/')
    parser.add_argument('--generation_dir', type=str, help='Directory to save generations to. Will be prepended by ~/bigcode-evaluation-harness/bigcode_eval/tasks/')
    parser.add_argument('--model', type=str, help='Name of the hf model to be used for generation')
    parser.add_argument('--use_chat_template', action="store_true", help='Use the chat template for generation')
    return parser.parse_args()


def main():
    
    args = parse_arguments()

    nuggets_config, dataset = get_dataset()
    # "codellama/CodeLlama-7b-hf"
    llm = LLM(model=args.model, tensor_parallel_size=4)

    # Run generate_for_one_shot for a bunch of different examples
    # "s3://scale-ml/users/mehermankikar/dagster_ingest_runs/e2e_run1_1200_2/dataset/"
    e2e_run_1_ds = args.s3_dataset_path
    e2e_ds = load_ds_s3(e2e_run_1_ds)
    e2e_ds = e2e_ds["train"]

    # For each task in the dataset
    i = 0
    full_results = {}
    for task in e2e_ds:
        
        start_time = time.time()
        print(f"Starting at task {i}")
        # save it as an example
        # vllm_code_llama_generation_all_turn
        home_directory = os.path.expanduser('~')
        print(home_directory)
        json_path = os.path.join(f"{home_directory}/bigcode-evaluation-harness/bigcode_eval/tasks/vllm_few_shot_examples/", f"public_ots_{i}.json")
        with open(json_path, "w") as f:
            json.dump([task], f)

        # Do the generation
        nuggets_config = NuggetsConfig(
            prompt_quality=2,
            add_context=False,
            example_idxs=None,
            examples_path=f"bigcode_eval/tasks/vllm_few_shot_examples/public_ots_{i}.json",
            use_chat_template=args.use_chat_template
        )
        one_shot_task_generation = generate_for_one_shot(args, nuggets_config, llm, dataset)
        full_results[i] = one_shot_task_generation

        # Delete the file
        os.remove(json_path)

        # Write results to a JSON file
        output_file = f"{home_directory}/bigcode-evaluation-harness/bigcode_eval/tasks/{args.generation_dir}/vllm_generation_results_{i}.json"
        with open(output_file, "w") as file:
            json.dump(one_shot_task_generation, file)

        i += 1
        end_time = time.time()
        print(f"Iteration took {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
