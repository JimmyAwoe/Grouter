"""Get the dummy dispatch for eod token"""

import torch
import json
import sys
import os
import argparse
from pathlib import Path

current_script_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(tools_dir)
sys.path.append(project_root)

from grouter.general_router import grouter


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Get dummy dispatch for eod token using grouter")
    
    parser.add_argument("--config-path", type=str, required=True, 
                       help="Path to grouter config JSON file")
    
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to grouter checkpoint file")
    
    parser.add_argument("--eod-token-id", type=int, default=100001,
                       help="EOD token ID to get dispatch for")
    
    return parser.parse_args()


def load_grouter(config_path, checkpoint_path):
    """Load grouter model from config and checkpoint"""
    # Load config
    with open(config_path, "r") as f:
        grt_config = json.load(f)
    
    # Initialize grouter
    grt = grouter(**grt_config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    grt.load_state_dict(ckpt)
    grt.eval()
    
    return grt


def get_dummy_dispatch(grt, eod_token_id):
    """Get dummy dispatch for eod token"""
    input_ids = torch.tensor([[eod_token_id]])
    attention_mask = torch.tensor([[1]])
    
    with torch.no_grad():
        dummy_dispatch_ids = grt(input_ids, attention_mask, None)
    
    return dummy_dispatch_ids[0][0]


def main():
    """Main function"""
    args = parse_args()
    
    # Validate file paths
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    
    # Load grouter
    print(f"Loading grouter from config: {args.config_path}")
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    grt = load_grouter(args.config_path, args.checkpoint_path)
    
    # Get dummy dispatch
    print(f"Getting dispatch for EOD token ID: {args.eod_token_id}")
    dispatch_ids = get_dummy_dispatch(grt, args.eod_token_id)
    
    # Output results
    print(f"The dispatch ids for eod token is {dispatch_ids}.")
    
if __name__ == "__main__":
    main()
