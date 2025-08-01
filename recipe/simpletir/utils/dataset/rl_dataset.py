# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import copy
import os
from typing import List, Optional, Union

import pandas as pd
from omegaconf import ListConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset, process_image
from verl.utils.model import compute_position_id_with_mask


class RLCustomPromptDataset(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        prompt_key="prompt",
        prompt=None,
        image_key="images",
        max_prompt_length=1024,
        filter_prompts=True,
        cache_dir="~/.cache/verl/rlhf",
        chat_template_func=None,
        apply_chat_template=False,
        return_raw_chat=False,
        truncation="error",
        # if using sample_size, trucnate every validation dataset into this to reduce the time used for evaluation each time
        sample_size=None,
        filter_overlong_prompts=True,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.prompt = prompt
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.apply_chat_template = apply_chat_template
        self.truncation = truncation
        self.sample_size = sample_size
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            parquet_file_name = "/".join(parquet_file.split("/")[-2:])
            dataframe["data_source"] = dataframe["data_source"].apply(
                lambda x: parquet_file_name
            )
            if self.sample_size is not None and len(dataframe) > self.sample_size:
                # use random state to ensure it can be reproducible
                dataframe = dataframe.sample(n=self.sample_size, random_state=42)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        if self.prompt is not None:
            self.dataframe[prompt_key] = self.dataframe[prompt_key].apply(
                lambda prompt_array: [
                    {
                        **d,
                        "content": self.prompt + d["content"],
                    }
                    for d in prompt_array
                ]
            )

        if self.apply_chat_template:
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True
                        )
                    )
                    <= self.max_prompt_length,
                    axis=1,
                )
            ]
        else:
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: len(doc[prompt_key][0]["content"])
                    <= self.max_prompt_length,
                    axis=1,
                )
            ]

            print(f"filter dataset len: {len(self.dataframe)}")

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        if self.apply_chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt_with_chat_template = chat[0]["content"]

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace(
                "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
            )
            row_dict["multi_modal_data"] = {
                "image": [
                    process_image(image) for image in row_dict.pop(self.image_key)
                ]
            }
            image_inputs = self.processor.image_processor(
                row_dict["multi_modal_data"]["image"], return_tensors="pt"
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            row_dict["multi_modal_inputs"] = {
                key: val for key, val in image_inputs.items()
            }

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while "<image>" in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        "<image>",
                        "<|vision_start|>"
                        + "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length)
                        + "<|vision_end|>",
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    "<|placeholder|>", self.processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            raw_prompt, add_special_tokens=False
        )

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
