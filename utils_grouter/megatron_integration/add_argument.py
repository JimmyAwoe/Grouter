"""
This module provides integration interfaces with Megatron-LM framework, supporting:
Command line argument extension
"""

import argparse

def _add_grouter_args(parser: argparse.ArgumentParser) -> None:
    """
    Add Grouter related command line arguments
    
    Args:
        parser: Megatron's ArgumentParser instance
    """
    group = parser.add_argument_group('Grouter Related')

    
    # Grouter Expert Migration arguments
    group.add_argument('--grouter-expert-placement-path', 
                      type=str, 
                      default=None,
                      help='Expert placement configuration JSON file path')
    
    group.add_argument('--grouter-enable-global-migration',
                      action='store_true',
                      help='Enable global expert migration at training start')
    
    group.add_argument('--grouter-migration-strategy',
                      type=str,
                      choices=['allreduce', 'p2p'],
                      default='allreduce',
                      help='Expert migration communication strategy')
    
    group.add_argument('--grouter-compression-ratio',
                      type=float,
                      default=0.8,
                      help='Parameter compression ratio for expert migration (0.0-1.0)')
    
    group.add_argument('--grouter-migration-timeout',
                      type=float,
                      default=30.0,
                      help='Migration timeout in seconds')
    
    group.add_argument('--grouter-migration-max-retries',
                      type=int,
                      default=3,
                      help='Maximum number of migration retries')
    
    group.add_argument('--grouter-migration-verbose',
                      action='store_true',
                      help='Enable detailed migration logs')

    group.add_argument('--expert-migration-steps-config-path', 
                      type=str, 
                      default=None,
                      help='Expert migration steps configuration JSON file path')

    # Combine Grouter in training arguments
    group.add_argument('--moe-use-grouter', 
                       action='store_true',
                       help='Use grouter to achieve faster routing.')

    group.add_argument('--fp32-grouter', 
                       action='store_true',
                       help='Use fp32 type grouter')

    group.add_argument('--use-single-grouter', 
                       action='store_true',
                       help='Use fp32 type grouter')

    group.add_argument('--grouter-checkpoint-path', 
                       type=str,
                       help='The path to load grouter checkpoint')

    group.add_argument('--use-grouter-weight', 
                       action='store_true',
                       help='Use grouter logits to compute scores')

    group.add_argument('--grouter-convert', action="store_true", default=None)

    group.add_argument('--grouter-output-logits', action="store_true")

    group.add_argument('--dynamic-act', action="store_true")

    group.add_argument('--dynamic-act-threshold', type=float, default=0.0)

    group.add_argument('--grouter-bias-checkpoint-path', type=str, default=None)

    # Grouter Dataset arguments
    group.add_argument('--use-grouter-dataset',
                       action='store_true',
                       help='Use pre-processed Grouter datasets instead of standard Megatron datasets')
    
    group.add_argument('--grouter-data-prefix',
                       nargs='*',
                       default=None,
                       help='Path to the directory containing Grouter pre-processed datasets')
    
    parser.add_argument('--node-data-dir', 
                        type=str, default=None,
                        help="Combine with grouter-data-prefix to locate where to load data")

    parser.add_argument('--node-dispatch-dir', 
                        type=str, default=None,
                        help="Combine with grouter-data-prefix to locate where to load dispatch")

    parser.add_argument('--grouter-data-config-path', 
                        type=str, default=None,
                        help="Where to load grouter dataset config")
    
    parser.add_argument('--no-document-idx-shuffle', action='store_true',
                        help="use sequential index in GPTDataset document index")

    parser.add_argument('--no-shuffle-effect', action='store_true',
                        help="use sequential index in GPTDataset shuffle index")
    
    parser.add_argument('--grouter-dataset-align-granularity', default='gpu', choices=['gpu', 'node'],
                        type=str, help='Determine align dataset by gpu or node')

    # Grouter Dilution arguments
    group.add_argument('--grouter-enable-distillation', action='store_true',
                      help='Enable Grouter distillation training')

    group.add_argument('--grouter-allow-partial-load', action='store_true',
                      help='Accept only load partial teacher model to enabel faster distillation')
    
    group.add_argument('--grouter-distillation-temperature', type=float, default=2.0,
                      help='Temperature for distillation')
    
    group.add_argument('--grouter-moe-layer-start', type=int, default=0,
                      help='Start layer index for MoE layers')
    
    group.add_argument('--grouter-moe-layer-end', type=int, default=None,
                      help='End layer index for MoE layers')
    
    group.add_argument('--grouter-checkpoint-dir', type=str, default=None,
                      help='Directory to save Grouter checkpoints')
    
    group.add_argument('--grouter-checkpoint-interval', type=int, default=None,
                      help='Save Grouter checkpoint every N steps')

    group.add_argument('--grouter-init-seed', type=int, default=1234,
                      help='Seed for grouter initialization')
    
    group.add_argument('--grouter-resume-from', type=str, default=None,
                      help='Path to Grouter checkpoint to resume from')
    
    group.add_argument('--grouter-config-path', type=str, default=None,
                      help='Path to JSON file containing Grouter configuration parameters')

    group.add_argument('--grouter-distillation-finetune-scores', action='store_true')

    return parser