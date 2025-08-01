# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Callable

import ray
import torch
from ray.exceptions import GetTimeoutError

from recipe.simpletir.utils.reward_score import _default_compute_score
from verl import DataProto


@ray.remote
def reward_func_timeout_ray(func: Callable, *args: Any, **kwargs: Any):
    """A Ray remote function that executes the given function with arguments.
    The timeout is handled by Ray's ray.get() method when retrieving results,
    not within this function itself.
    Args:
        func: The function to execute
        timeout_seconds: This parameter is kept for backward compatibility but not used here.
                        The actual timeout is enforced by ray.get() in the calling code.
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
    Returns:
        The result of func(*args, **kwargs)
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # If an exception occurs during execution, return a default fail score
        print(f"Error in reward computation: {e}")
        return {"score": 0.0, "extra_info": {"is_filter": 1}}


class CodeRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.timeout_seconds = 60

    def __call__(self, data: DataProto):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        response_ids = data.batch["responses"]
        sequences_strs = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch["data_source"]
        extra_infos = [
            data_item.non_tensor_batch.get("extra_info", None) for data_item in data
        ]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        # Parse code actions from responses
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        solution_strs = [""] * len(data)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )

            match = re.search(pattern, response_str, re.DOTALL)
            if match:
                code = match.group(1).strip()
                solution_strs[i] = code

        # Compute rewards
        scores: list[float] = [0.0] * len(solution_strs)
        extra_info_dict: dict[
            str, list[float]
        ] = {}  # Key -> list of values for the batch
        print(
            f"Scoring process started over {len(solution_strs)} samples, waiting for results..."
        )

        futures = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]

            future = reward_func_timeout_ray.remote(
                self.compute_score,
                data_source,
                solution_str,
                ground_truth,
                extra_info,
            )
            futures.append(future)

        default_fail_score = {
            "score": 0.0,
            "extra_info": {"is_filter": 1},
        }  # Default on error which should be filtered

        for i, future in enumerate(futures):
            try:
                task_result = ray.get(future, timeout=self.timeout_seconds)

                if isinstance(task_result, dict):
                    assert "extra_info" in task_result, (
                        f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    )
                    score_result = task_result
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"].update({"is_filter": 0})
                elif isinstance(task_result, (int, float)):
                    score_result = {
                        "score": float(task_result),
                        "extra_info": {"is_filter": 0},
                    }
                else:
                    print(
                        f"Unexpected task_result type for item {i}: {type(task_result)}. Using default score. Result: {task_result}"
                    )
                    ray.cancel(future, force=True)
                    score_result = default_fail_score
            except GetTimeoutError:
                print(
                    f"Timeout processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'). Using default score."
                )
                score_result = default_fail_score
            except Exception as e:
                print(
                    f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'): {e}"
                )
                import traceback

                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            scores[i] = float(score_result.get("score", 0.0))

            if "extra_info" in score_result and isinstance(
                score_result["extra_info"], dict
            ):
                for key, value in score_result["extra_info"].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        data_sources = data.non_tensor_batch["data_source"]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        return {"reward_tensor": reward_tensor, "extra_info": extra_info_dict}
