# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import datetime
import os
import torch
import numpy

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.datasets.sft_dataset import SFTDataset

import megatron.legacy.model  # isort: skip
from utils_grouter.core.config import OptimizationConfig, MegatronPaths
from utils_grouter.utils.data_structures import (
    AlignedDispatchBlendedDataset, 
    AlignedDispatchGPTDataset, 
    MergedDataset
)
from megatron.core.datasets import indexed_dataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

# Add the utils_grouter directory to the path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
grouter_path = os.path.join(current_dir, '..', 'utils_grouter')
if os.path.exists(grouter_path) and grouter_path not in sys.path:
    sys.path.insert(0, grouter_path)
    
def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.
    
    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration
        
    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
            )

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
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None, grouter_losses: dict = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)
        grouter_losses (dict, optional): Grouter distillation losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    reporting_dict = {'lm loss': reporting_loss}
    if grouter_losses:
        for key, value in grouter_losses.items():
            reporting_dict[key] = value

    return (loss, num_tokens, reporting_dict)


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        if args.use_grouter_dataset:
            tokens, labels, loss_mask, attention_mask, position_ids, dispatch_ids = get_batch(data_iterator)
        else:
            # Normal training mode - get all data
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    grouter_distillation_losses = {}

    with stimer:
        if args.grouter_enable_distillation:
            if args.use_legacy_models:
                output_tensor = model(tokens, position_ids, attention_mask, labels=None)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
        elif args.use_grouter_dataset:
            # Only support GPTModel now
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, dispatch_ids=dispatch_ids
            )
        else:
            # Normal training mode
            if args.use_legacy_models:
                output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # carry out grouter dilution
    if args.grouter_enable_distillation:
        if args.pipeline_model_parallel_size > 1:
            pp_group = parallel_state.get_pipeline_model_parallel_group()

            if is_pp_first_stage(pp_group):
                grouter_distillation_losses = model.module.module.grouter_distillation_trainer.distill_step(tokens, attention_mask)
        else:
            grouter_distillation_losses = model.module.module.grouter_distillation_trainer.distill_step(tokens, attention_mask)

    if args.grouter_enable_global_migration and args.curr_iteration in model.migration_steps:
        # Execute expert migration
        success = model.migration_manager.migrate_experts(model, args.curr_iteration)
            
        if success:
            print_rank_0("Grouter expert migration completed successfully")
            if hasattr(model, "global_migration_map"):
                model.global_migration_map = model.migration_manager.migration_manager.get_global_migration_plan(model.global_migration_map)
            else:
                model.global_migration_map = model.migration_manager.migration_manager.get_global_migration_plan()
            model.migration_tensor = torch.zeros(args.num_experts, device=torch.distributed.get_rank())
            for eid, replaced_eid in model.global_migration_map:
                model.migration_tensor[eid] = replaced_eid
        else:
            raise RuntimeError("Grouter expert migration failed") 
    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model, grouter_losses=grouter_distillation_losses)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def grouter_token_predispatch_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets with Grouter dispatch information.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
        
    Returns:
        Tuple of (train_ds, valid_ds, test_ds, train_dispatch_ds, valid_dispatch_ds, test_dispatch_ds)
    """
    args = get_args()
    
    print_rank_0("> building Grouter train, validation, and test datasets for GPT ...")
    
    # Import Grouter modules
    assert args.dataloader_type == "external", 'Leave the dataloader creation in datasets_provider by set dataloader_type as external'
    
    # Get Grouter data path
    prefix = args.grouter_data_prefix
    if len(prefix) == 1:
        input_data_prefix = [1.0, os.path.join(args.data_paths.node_data_dir, f"{prefix[0]}_data")]
        input_dispatch_prefix = [1.0, os.path.join(args.data_paths.node_dispatch_dir, f"{prefix[0]}_dispatch")]
    else:
        add_data_prefix = lambda x: os.path.join(args.node_data_dir, f"{x}_data")
        add_dispatch_prefix = lambda x: os.path.join(args.node_dispatch_dir, f"{x}_dispatch")
        input_data_prefix = [add_data_prefix(x) if i % 2 == 1 else x for i, x in enumerate(prefix)]
        input_dispatch_prefix = [add_dispatch_prefix(x) if i % 2 == 1 else x for i, x in enumerate(prefix)]
    # Overlap arguments' datapath
    args.data_path = input_data_prefix
    
    # Load Grouter configuration
    config = OptimizationConfig.from_file(args.grouter_data_config_path)
    
    # Create GPT dataset config
    gpt_config = core_gpt_dataset_config_from_args(args)

    if args.no_document_idx_shuffle:
        gpt_config.no_document_idx_shuffle = True
    if args.no_shuffle_effect:
        gpt_config.no_shuffle_effect = True
    
    # Build standard datasets using BlendedMegatronDatasetBuilder
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, gpt_config
    ).build()
    
    print_rank_0("> finished creating GPT datasets, now creating aligned dispatch datasets ...")
    
    for ds in [train_ds, valid_ds, test_ds]:
        # Create aligned dispatch datasets for each split
        _align_document_index(ds, args.grouter_dataset_align_granularity)

    train_dispatch_ds = None
    valid_dispatch_ds = None  
    test_dispatch_ds = None
    
    # Get current rank to determine which node's data to load
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    
    # Build dispatch datasets for each split
    splits = [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]
    dispatch_datasets = {}
    
    for split_name, dataset in splits:
        if dataset is not None:
            dispatch_ds = _create_aligned_dispatch_dataset(
                dataset, input_dispatch_prefix, config
            )
            dispatch_datasets[split_name] = dispatch_ds
        else:
            dispatch_datasets[split_name] = None
    
    train_dispatch_ds = dispatch_datasets["train"]
    valid_dispatch_ds = dispatch_datasets["valid"]
    test_dispatch_ds = dispatch_datasets["test"]
    
    print_rank_0("> finished creating Grouter datasets with dispatch information ...")
    train_ds = MergedDataset(train_ds, train_dispatch_ds)
    valid_ds = MergedDataset(valid_ds, valid_dispatch_ds)
    test_ds = MergedDataset(test_ds, test_dispatch_ds)
    
    return train_ds, valid_ds, test_ds 

def _align_document_index(blended_dataset, granularity='gpu'):
    """
    Align document index so that each GPU gets data from a specific GPTDataset.
    
    This function modifies the dataset_index of the BlendedDataset to ensure
    that each GPU (identified by data_parallel_rank) only samples from one
    specific GPTDataset within the blend.
    
    The key insight is that MegatronPretrainingPredispatchSampler creates batches
    of size micro_batch_times_data_parallel_size, and each GPU gets a slice
    [data_parallel_rank * micro_batch_size : (data_parallel_rank + 1) * micro_batch_size]
    from each batch.
    
    Args:
        blended_dataset: The BlendedDataset to modify
        granularity: Align different dataset for different gpu or node
    """
    assert granularity == 'gpu' or granularity == 'node', 'Only support gpu or node granularity'
    world_size = torch.distributed.get_world_size()
    
    # Get the number of datasets in the blend
    num_datasets = len(blended_dataset.datasets)
    
    # Ensure we have at least as many datasets as GPUs
    if granularity == 'gpu' and num_datasets < world_size:
        raise RuntimeError(f"Number of datasets ({num_datasets}) is less than world size ({world_size})")
    elif granularity == 'node' and num_datasets < world_size // 8:
        raise RuntimeError(f"Number of datasets ({num_datasets}) is less than dataset needed ({world_size // 8})")
    
    for gpu_rank in range(world_size):
        # Assign each GPU to a specific dataset
        if granularity == 'gpu':
            dataset_id = gpu_rank % num_datasets
        else:
            dataset_id = gpu_rank // 8
    
        # Get the total size of the blended dataset
        total_size = len(blended_dataset.dataset_index)
    
        # Get micro_batch_size from args
        args = get_args()
        micro_batch_size = args.micro_batch_size
        micro_batch_times_data_parallel_size = micro_batch_size * world_size
    
        # The sampler creates batches of size micro_batch_times_data_parallel_size
        # Each GPU gets samples at positions: gpu_rank * micro_batch_size + k * micro_batch_times_data_parallel_size
        # where k is the batch number
    
        # Find all positions where this GPU will sample from
        gpu_sample_positions = []
        for batch_start in range(0, total_size, micro_batch_times_data_parallel_size):
            gpu_start_in_batch = batch_start + gpu_rank * micro_batch_size
            gpu_end_in_batch = min(gpu_start_in_batch + micro_batch_size, total_size)
            for pos in range(gpu_start_in_batch, gpu_end_in_batch):
                if pos < total_size:
                    gpu_sample_positions.append(pos)
    
        if len(gpu_sample_positions) == 0:
            raise RuntimeError(f"No sample positions found for GPU {gpu_rank}")
    
        # Find all positions that currently point to the assigned dataset
        dataset_mask = (blended_dataset.dataset_index == dataset_id)
        dataset_indices = numpy.where(dataset_mask)[0]
    
        if len(dataset_indices) == 0:
            print_rank_0(f"Warning: No samples found for dataset {dataset_id} on GPU {gpu_rank}")
            return
    
        blended_dataset.dataset_index[gpu_sample_positions] = dataset_id
        blended_dataset.dataset_sample_index[gpu_sample_positions] = numpy.arange(len(gpu_sample_positions))
    
    
    print_rank_0(f"GPU {gpu_rank} aligned to use dataset {dataset_id} for {len(gpu_sample_positions)} sample positions")


def _create_aligned_dispatch_dataset(token_dataset, dispatch_prefix, config):
    """Create aligned dispatch dataset for a given split.
    
    Args:
        token_dataset: The token dataset (BlendedDataset)
        dispatch_prefix: The prefix for dispatch data
        config: Grouter configuration
        
    Returns:
        AlignedDispatchBlendedDataset or None
    """
    dispatch_path = MegatronPaths(dispatch_prefix)
    prefixs, _ = get_blend_and_blend_per_split(dispatch_path)[0]

    dispatch_datasets = []
    for prefix in prefixs:
        dataset = indexed_dataset.IndexedDataset(prefix)
        dispatch_datasets.append(dataset)
        
    # Create aligned dispatch dataset by processing each sample through the same pipeline
    aligned_dispatch_dataset = _create_aligned_dispatch_blended_dataset(
        token_dataset, dispatch_datasets, config
    )
        
    return aligned_dispatch_dataset


def _create_aligned_dispatch_blended_dataset(token_dataset, dispatch_dataset, config):
    """Create a dispatch dataset that aligns with the processed token dataset.
    
    Args:
        token_dataset: The token dataset (BlendedDataset)
        dispatch_dataset: The dispatch dataset (IndexedDataset)
        config: Grouter configuration
        
    Returns:
        AlignedDispatchBlendedDataset
    """
    
    # Create aligned datasets for each component in the blended dataset
    aligned_datasets = []
    
    for i, gpt_dataset in enumerate(token_dataset.datasets):
        # Create a dispatch dataset that mirrors the GPT dataset structure
        aligned_dispatch_dataset = AlignedDispatchGPTDataset(
            gpt_dataset, dispatch_dataset[i], config.topk, 
            config.dummy_expert_id, config.eod_token_id
        )
        aligned_datasets.append(aligned_dispatch_dataset)
    
    # Create new BlendedDataset with aligned dispatch data
    aligned_blended_dataset = AlignedDispatchBlendedDataset(
        aligned_datasets,
        token_dataset.weights,
        token_dataset.size,
        token_dataset.config,
        token_dataset.dataset_index,
        token_dataset.dataset_sample_index
    )
    
    return aligned_blended_dataset
    

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    grouter_token_predispatch_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    use_grouter_dataset = os.environ.get('USE_GROUTER_DATASET', "False").lower() == 'true'

    if use_grouter_dataset:
        print_rank_0("Use Grouter dataset")
        pretrain(
            grouter_token_predispatch_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
            store=store,
        )
    else:
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
            store=store,
        )
