#!/usr/bin/env python3
"""
Batch Expert Preference Vector Analysis

This script processes multiple predispatch files in batch and generates
expert preference vectors for each file, then combines the results.
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import glob

# Add project paths
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]  # .../general_router
_MEGATRON_ROOT = _PROJECT_ROOT / "Megatron-LM"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGATRON_ROOT))

from expert_preference_analyzer import ExpertPreferenceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_expert_preference_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class BatchExpertPreferenceAnalyzer:
    """Batch expert preference vector analyzer"""
    
    def __init__(self, num_experts: int, key: str = 'text'):
        """
        Initialize batch analyzer
        
        Args:
            num_experts: Total number of experts in the MoE model
            key: Data key name (default: 'text')
        """
        self.num_experts = num_experts
        self.key = key
        self.analyzer = ExpertPreferenceAnalyzer(num_experts, key)
        logger.info(f"Initialized batch analyzer with {num_experts} experts, key: {key}")
    
    def find_predispatch_files(self, input_pattern: str) -> List[str]:
        """
        Find all predispatch files matching the pattern
        
        Args:
            input_pattern: Glob pattern to match predispatch files
            
        Returns:
            List of predispatch file prefixes
        """
        # Find all .bin files matching the pattern
        bin_files = glob.glob(f"{input_pattern}*_{self.key}_dispatch_ids.bin")
        
        # Extract prefixes (remove the suffix)
        prefixes = []
        for bin_file in bin_files:
            prefix = bin_file.replace(f"_{self.key}_dispatch_ids.bin", "")
            prefixes.append(prefix)
        
        prefixes.sort()  # Sort for consistent ordering
        logger.info(f"Found {len(prefixes)} predispatch files matching pattern: {input_pattern}")
        
        return prefixes
    
    def process_single_file(self, predispatch_prefix: str, output_dir: Path) -> Tuple[str, int]:
        """
        Process a single predispatch file
        
        Args:
            predispatch_prefix: Prefix path to predispatch files
            output_dir: Output directory
            
        Returns:
            (output_file_path, sample_count)
        """
        prefix_name = Path(predispatch_prefix).name
        output_file = output_dir / f"{prefix_name}_expert_preference_vectors.txt"
        
        logger.info(f"Processing: {predispatch_prefix}")
        
        # Analyze samples
        results = self.analyzer.analyze_samples(predispatch_prefix)
        
        # Save results
        self.analyzer.save_results_to_txt(
            results, 
            str(output_file), 
            include_stats=True
        )
        
        logger.info(f"Completed: {output_file} ({len(results)} samples)")
        return str(output_file), len(results)
    
    def process_batch(self, input_pattern: str, output_dir: str, 
                     combine_results: bool = True) -> Dict[str, any]:
        """
        Process multiple predispatch files in batch
        
        Args:
            input_pattern: Glob pattern to match predispatch files
            output_dir: Output directory
            combine_results: Whether to combine all results into a single file
            
        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting batch processing...")
        start_time = time.time()
        
        # Find all predispatch files
        predispatch_prefixes = self.find_predispatch_files(input_pattern)
        
        if not predispatch_prefixes:
            raise ValueError(f"No predispatch files found matching pattern: {input_pattern}")
        
        # Process each file
        results = {
            'processed_files': [],
            'total_samples': 0,
            'output_files': [],
            'processing_times': []
        }
        
        for i, prefix in enumerate(predispatch_prefixes):
            file_start_time = time.time()
            
            try:
                output_file, sample_count = self.process_single_file(prefix, output_dir)
                
                results['processed_files'].append(prefix)
                results['output_files'].append(output_file)
                results['total_samples'] += sample_count
                results['processing_times'].append(time.time() - file_start_time)
                
                logger.info(f"Progress: {i+1}/{len(predispatch_prefixes)} files completed")
                
            except Exception as e:
                logger.error(f"Failed to process {prefix}: {e}")
                continue
        
        # Combine results if requested
        if combine_results and len(results['output_files']) > 1:
            combined_file = self.combine_results(results['output_files'], output_dir)
            results['combined_file'] = combined_file
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['average_time_per_file'] = np.mean(results['processing_times'])
        
        logger.info("Batch processing completed!")
        logger.info(f"Total files processed: {len(results['processed_files'])}")
        logger.info(f"Total samples: {results['total_samples']}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per file: {results['average_time_per_file']:.2f} seconds")
        
        return results
    
    def combine_results(self, output_files: List[str], output_dir: Path) -> str:
        """
        Combine multiple result files into a single file
        
        Args:
            output_files: List of output file paths
            output_dir: Output directory
            
        Returns:
            Path to combined file
        """
        combined_file = output_dir / "combined_expert_preference_vectors.txt"
        
        logger.info(f"Combining {len(output_files)} result files...")
        
        total_samples = 0
        sample_id_offset = 0
        
        with open(combined_file, 'w', encoding='utf-8') as outf:
            # Write header
            outf.write("# Combined Expert Preference Vector Analysis Results\n")
            outf.write(f"# Number of experts: {self.num_experts}\n")
            outf.write(f"# Format: sample_id sequence_length expert_0_freq expert_1_freq ... expert_{self.num_experts-1}_freq\n")
            outf.write("#\n")
            
            for i, output_file in enumerate(output_files):
                logger.info(f"Processing file {i+1}/{len(output_files)}: {output_file}")
                
                with open(output_file, 'r', encoding='utf-8') as inf:
                    for line in inf:
                        line = line.strip()
                        
                        # Skip comments and empty lines
                        if line.startswith('#') or not line:
                            continue
                        
                        # Parse line and update sample ID
                        parts = line.split()
                        if len(parts) >= 2:
                            # Update sample ID with offset
                            parts[0] = str(sample_id_offset)
                            outf.write(' '.join(parts) + '\n')
                            sample_id_offset += 1
                            total_samples += 1
        
        # Update header with total samples
        with open(combined_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace(
            "# Combined Expert Preference Vector Analysis Results\n",
            f"# Combined Expert Preference Vector Analysis Results\n# Total samples: {total_samples}\n"
        )
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Combined results saved to: {combined_file}")
        logger.info(f"Total samples in combined file: {total_samples}")
        
        return str(combined_file)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Batch analyze expert preference vectors from multiple predispatch results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_pattern', 
        required=True, 
        type=str,
        help='Glob pattern to match predispatch files (e.g., "/path/to/predispatch/tf-c4-*")'
    )
    
    parser.add_argument(
        '--output_dir', 
        required=True, 
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--num_experts', 
        required=True, 
        type=int,
        help='Total number of experts in the MoE model'
    )
    
    parser.add_argument(
        '--key', 
        default='text', 
        type=str,
        help='Data key name (default: text)'
    )
    
    parser.add_argument(
        '--no_combine', 
        action='store_true',
        help='Do not combine results into a single file'
    )
    
    parser.add_argument(
        '--log_level', 
        default='INFO', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting Batch Expert Preference Vector Analysis")
    logger.info(f"Input pattern: {args.input_pattern}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Number of experts: {args.num_experts}")
    logger.info(f"Data key: {args.key}")
    logger.info(f"Combine results: {not args.no_combine}")
    
    try:
        # Initialize batch analyzer
        batch_analyzer = BatchExpertPreferenceAnalyzer(
            num_experts=args.num_experts,
            key=args.key
        )
        
        # Process batch
        results = batch_analyzer.process_batch(
            args.input_pattern,
            args.output_dir,
            combine_results=not args.no_combine
        )
        
        logger.info("Batch analysis completed successfully!")
        logger.info(f"Processed {len(results['processed_files'])} files")
        logger.info(f"Total samples: {results['total_samples']}")
        
        if 'combined_file' in results:
            logger.info(f"Combined results: {results['combined_file']}")
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
