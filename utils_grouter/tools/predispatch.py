from pathlib import Path
import sys
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]  # .../general_router
_MEGATRON_ROOT = _PROJECT_ROOT / "Megatron-LM"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGATRON_ROOT))
import torch
import gzip
import time
import json
import sys
import argparse
import os
from megatron.core.datasets import indexed_dataset
from utils_grouter.grouter.general_router import Grouter
from transformers import AutoTokenizer
import torch.distributed as dist
import numpy

class GrouterEncoder():
    def __init__(self, grouter, tokenizer, json_keys, max_length, device, append_eod):
        self.tokenizer = tokenizer
        self.grouter = grouter
        self.json_keys = json_keys
        self.device = device
        self.max_length = max_length
        self.append_eod = append_eod

    def encode_grouter(self, json_line):
        data = json.loads(json_line)
        dispatch_ids = {}
        dispatch_doc_lens = {}
        dispatch_scores = {}
        dispatch_scores_doc_lens = {}
        tokenized_ids = {}
        tokenized_doc_lens = {}
        
        for key in self.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            
            # Batch tokenization + single forward pass (batch processing by sentences within document)
            enc = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=False,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(self.device, non_blocking=True)
            #position_ids = enc["position_ids"]
            
            # Store tokenized data
            doc_tokenized_ids = []
            doc_tokenized_lens = []
            for i in range(input_ids.size(0)):
                sentence_ids = input_ids[i].detach().cpu().tolist()
                if len(sentence_ids) > 0:
                    doc_tokenized_ids.extend(sentence_ids)
                    doc_tokenized_lens.append(len(sentence_ids))
            
            # Add EOD token if specified (following preprocess_data.py pattern)
            if self.append_eod:
                if len(doc_tokenized_ids) > 0:
                    doc_tokenized_ids.append(self.tokenizer.eos_token_id)
                    doc_tokenized_lens[-1] += 1
            
            tokenized_ids[key] = doc_tokenized_ids
            tokenized_doc_lens[key] = doc_tokenized_lens
            
            # Get grouter dispatch results (returns scores[routing_map] and topk_idx)
            with torch.inference_mode():
                # use mixed precision
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    grouter_scores_batch, grouter_topk_idx_batch = self.grouter(input_ids, attention_mask, None)
            
            # scores[routing_map] is a 1D tensor (flattened), convert to float32 and list
            # topk_idx is a 2D tensor [num_tokens, top_k], flatten it to 1D list
            doc_scores = grouter_scores_batch.detach().cpu().float().tolist()
            doc_dispatch_ids = grouter_topk_idx_batch.detach().cpu().flatten().tolist()
            
            dispatch_scores[key] = doc_scores
            dispatch_scores_doc_lens[key] = [len(doc_scores)]
            dispatch_ids[key] = doc_dispatch_ids
            # Record the total length of the entire document (for add_document)
            dispatch_doc_lens[key] = [len(doc_dispatch_ids)]
        
        return dispatch_ids, dispatch_doc_lens, dispatch_scores, dispatch_scores_doc_lens, tokenized_ids, tokenized_doc_lens, len(json_line)

class GrouterPreDispatch():
    def __init__(self, 
                 grouter, 
                 tokenizer, 
                 max_length,
                 args,
                 device,
                 rank,
                 world_size,
                 append_eod):
        self.grouter = grouter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.append_eod = append_eod

    def dispatch_json_file(self, file_path, output_prefix, json_keys):
        input_file_name = file_path
        if self.rank == 0:
            print("Opening", input_file_name)
        fin = gzip.open(input_file_name, 'rt', encoding='utf-8')

        startup_start = time.time()
        encoder = GrouterEncoder(self.grouter, self.tokenizer, json_keys, 
                                 self.max_length, self.device, self.append_eod)
        # Set args attribute for encoder to access append_eod setting
        encoder.args = self.args
        
        # Builders for dispatch data
        dispatch_output_bin_files = {}
        dispatch_output_idx_files = {}
        dispatch_builders = {}
        
        # Builders for dispatch scores
        dispatch_scores_output_bin_files = {}
        dispatch_scores_output_idx_files = {}
        dispatch_scores_builders = {}
        
        # Builders for tokenized data
        tokenized_output_bin_files = {}
        tokenized_output_idx_files = {}
        tokenized_builders = {}

        # Each rank writes its own shard
        shard_prefix = f"{output_prefix}_rank{self.rank}"
        for key in json_keys:
            # Dispatch data files (topk_idx)
            dispatch_output_bin_files[key] = f"{shard_prefix}_{key}_dispatch_ids.bin"
            dispatch_output_idx_files[key] = f"{shard_prefix}_{key}_dispatch_ids.idx"
            dispatch_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                dispatch_output_bin_files[key],
                # if you expert num is greater than 256, your shold substitute
                # below code with 
                # dtype=indexed_dataset.DType.optimal_dtype(self.expert_num),
                dtype=numpy.uint8,
            )
            
            # Dispatch scores files (scores, float32)
            dispatch_scores_output_bin_files[key] = f"{shard_prefix}_{key}_dispatch_scores.bin"
            dispatch_scores_output_idx_files[key] = f"{shard_prefix}_{key}_dispatch_scores.idx"
            dispatch_scores_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                dispatch_scores_output_bin_files[key],
                dtype=numpy.float32,
            )
            
            # Tokenized data files
            tokenized_output_bin_files[key] = f"{shard_prefix}_{key}_tokenized.bin"
            tokenized_output_idx_files[key] = f"{shard_prefix}_{key}_tokenized.idx"
            tokenized_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                tokenized_output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(self.tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        if self.rank == 0:
            print("Time to startup:", startup_end - startup_start)

        # All ranks read the same file, but only process lines they are responsible for (by line number modulo)
        for i, json_line in enumerate(fin):
            if (i % self.world_size) != self.rank:
                continue
            dispatch_doc, dispatch_doc_lens, dispatch_scores_doc, dispatch_scores_doc_lens, tokenized_doc, tokenized_doc_lens, bytes_processed = encoder.encode_grouter(json_line)
            total_bytes_processed += bytes_processed
            
            # Store dispatch data (topk_idx)
            for key in dispatch_doc.keys():
                dispatch_builders[key].add_document(dispatch_doc[key], dispatch_doc_lens[key])
            
            # Store dispatch scores
            for key in dispatch_scores_doc.keys():
                dispatch_scores_builders[key].add_document(dispatch_scores_doc[key], dispatch_scores_doc_lens[key])
            
            # Store tokenized data
            for key in tokenized_doc.keys():
                tokenized_builders[key].add_document(tokenized_doc[key], tokenized_doc_lens[key])
                
            self.print_processing_stats(i + 1, proc_start, total_bytes_processed)

        fin.close()
        
        # finalize dispatch data (topk_idx)
        for key in json_keys:
            dispatch_builders[key].finalize(dispatch_output_idx_files[key])
        
        # finalize dispatch scores
        for key in json_keys:
            dispatch_scores_builders[key].finalize(dispatch_scores_output_idx_files[key])
            
        # finalize tokenized data
        for key in json_keys:
            tokenized_builders[key].finalize(tokenized_output_idx_files[key])

        # Synchronize and merge shards to final output on rank0
        if dist.is_initialized():
            dist.barrier()
        if self.rank == 0:
            # Merge dispatch data shards (topk_idx)
            self._merge_shards(json_keys, output_prefix, "dispatch_ids", dispatch_output_bin_files, dispatch_output_idx_files)
            
            # Merge dispatch scores shards
            self._merge_shards(json_keys, output_prefix, "dispatch_scores", dispatch_scores_output_bin_files, dispatch_scores_output_idx_files)
            
            # Merge tokenized data shards
            self._merge_shards(json_keys, output_prefix, "tokenized", tokenized_output_bin_files, tokenized_output_idx_files)
            
            # Optionally cleanup shards
            if getattr(self.args, "cleanup_shards", False):
                self._cleanup_shards(json_keys, output_prefix, world_size=self.world_size)

    def _merge_shards(self, json_keys, output_prefix, data_type, output_bin_files, output_idx_files):
        """Helper method to merge shards for both dispatch and tokenized data"""
        final_output_bin_files = {}
        final_output_idx_files = {}
        final_builders = {}
        
        for key in json_keys:
            final_prefix = f"{output_prefix}_{key}_{data_type}"
            final_output_bin_files[key] = f"{final_prefix}.bin"
            final_output_idx_files[key] = f"{final_prefix}.idx"
            
            # Use appropriate dtype based on data type
            if data_type == "dispatch_ids":
                dtype = numpy.uint8
            elif data_type == "dispatch_scores":
                dtype = numpy.float32
            else:  # tokenized
                dtype = indexed_dataset.DType.optimal_dtype(self.tokenizer.vocab_size)
                
            final_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                final_output_bin_files[key],
                dtype=dtype,
            )

        # Read shards from each rank and merge in original order (doc0, doc1, ...)
        for key in json_keys:
            shard_datasets = []
            shard_doc_counts = []
            for r in range(self.world_size):
                # Correct to path prefix output_prefix/rank{r}
                shard_key_prefix = f"{output_prefix}_rank{r}_{key}_{data_type}"
                ds = indexed_dataset.IndexedDataset(shard_key_prefix)
                shard_datasets.append(ds)
                shard_doc_counts.append(int(ds.document_indices.shape[0] - 1))

            max_local_docs = max(shard_doc_counts) if shard_doc_counts else 0
            for local_doc in range(max_local_docs):
                for r in range(self.world_size):
                    if local_doc >= shard_doc_counts[r]:
                        continue
                    ds = shard_datasets[r]
                    doc_indices = ds.document_indices
                    start_seq = int(doc_indices[local_doc])
                    end_seq = int(doc_indices[local_doc + 1])
                    sequences = ds[start_seq:end_seq]
                    # Each document now has only one sequence, use directly
                    flat_ids = []
                    lengths = []
                    for seq in sequences:
                        arr = seq
                        n = int(arr.size)
                        if n > 0:
                            flat_ids.extend(arr.tolist())
                        lengths.append(n)
                    final_builders[key].add_document(flat_ids, lengths)

            final_builders[key].finalize(final_output_idx_files[key])

    def _cleanup_shards(self, json_keys, output_prefix, world_size):
        """Helper method to cleanup shard files"""
        for r in range(world_size):
            shard_prefix_r = f"{output_prefix}_rank{r}"
            for key in json_keys:
                # Cleanup dispatch files (topk_idx)
                dispatch_bin_path = f"{shard_prefix_r}_{key}_dispatch_ids.bin"
                dispatch_idx_path = f"{shard_prefix_r}_{key}_dispatch_ids.idx"
                # Cleanup dispatch scores files
                dispatch_scores_bin_path = f"{shard_prefix_r}_{key}_dispatch_scores.bin"
                dispatch_scores_idx_path = f"{shard_prefix_r}_{key}_dispatch_scores.idx"
                # Cleanup tokenized files
                tokenized_bin_path = f"{shard_prefix_r}_{key}_tokenized.bin"
                tokenized_idx_path = f"{shard_prefix_r}_{key}_tokenized.idx"
                
                for path in [dispatch_bin_path, dispatch_idx_path, dispatch_scores_bin_path, dispatch_scores_idx_path, tokenized_bin_path, tokenized_idx_path]:
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        """copy from megatron.tools.preproess_data.Partition"""
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    #def select_micro_batch_sample(self):

def parse_args(args):
    parser = argparse.ArgumentParser(description='argument grouter predispatch')
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--cleanup_shards", action="store_true")
    parser.add_argument("--key", type=str, default='text')

    # for grouter, if you have set grouter config, the below argument
    # can be set to none
    parser.add_argument("--grouter_ckpt", required=True, default=str)
    parser.add_argument("--grouter_config", default=None, type=str)
    parser.add_argument("--grouter_bias_path", default=None, type=str)

    # predispatch data
    parser.add_argument("--output_prefix", type=str)
    parser.add_argument("--max_length", type=int, default=100000)
    
    # Tokenization options (following preprocess_data.py pattern)
    parser.add_argument("--append_eod", action="store_true",
                       help="Append an <eod> token to the end of a document in tokenized data")

    args = parser.parse_args(args)
    return args


def main(args):
    # we now only support HuggingFaceType tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    # Initialize distributed training (environment variables set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.grouter_config:
        with open(args.grouter_config, "r") as f:
            grt_config = json.load(f)
        grt = Grouter(**grt_config, predispatch_mode=True)
    ckpt = torch.load(args.grouter_ckpt, map_location="cpu")
    grt.load_state_dict(ckpt)
    if args.grouter_bias_path is not None:
        grt.load_bias(args.grouter_bias_path)
    grt.eval()
    grt.to(torch.device(f"cuda:{local_rank}"))

    device = torch.device(f"cuda:{local_rank}")
    processor = GrouterPreDispatch(grt, tokenizer, args.max_length, args, device, rank, world_size, args.append_eod)
    processor.dispatch_json_file(args.data_path, args.output_prefix, [args.key])

    # Destroy process group to avoid blocking on exit
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    args = parse_args(None)
    main(args)





