# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain Llama"""
import torch
from functools import partial

import megatron_br
import megatron
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import (
    get_args,
    print_rank_0,
    get_timers,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
)
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron_br.core.offload.common import get_runtime_info
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.training.training import pretrain
from torch_br.supa.profiler_kineto import record_function
from megatron_br.training.real_time_profiler import record_kernel

from megatron.core.models.gpt import GPTModel
from megatron_br.core.models.llama2.llama2_layer_specs import (
    get_llama2_layer_local_spec,
)

from megatron_br.core.datasets.gpt_dataset import get_batch, core_gpt_dataset_config_from_args

def model_provider(pre_process=True, post_process=True):
    args = get_args()

    print_rank_0('building LLama model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron_br.legacy.model.llama_model.LlamaModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:
        transformer_layer_spec = get_llama2_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, config)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    return model

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    # Get the batch.
    timers('batch-generator').start()
    with record_function("_exec_load_micro_batch"):
        with record_kernel("_exec_load_micro_batch"):
            tokens, labels, loss_mask, attention_mask, _ = get_batch(data_iterator, False)
    timers('batch-generator').stop()

    # BIRENTECH ----
    get_runtime_info().increase_runtime_info_micro_batch_id()
    get_runtime_info().reset_layer_number()
    # ---- BIRENTECH

    with record_function("_exec_forward_pass"):
        with record_kernel("_exec_forward_pass"):
            output_tensor = model(tokens, None, attention_mask, labels=labels)
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask)

def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets ' 'for llama ...')

    config = core_gpt_dataset_config_from_args(args)


    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating llama datasets ...")

    return train_ds, valid_ds, test_ds


def fork_exec_startup():
    # use the combination of the fork and exec system calls to prevent the release of memory
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)


if __name__ == "__main__":
    fork_exec_startup()
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step)
