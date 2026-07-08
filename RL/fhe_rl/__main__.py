
import sys
import os
import argparse
from .run import run_agent
from .train import train_agent
from .test import test_agent
from .utils import load_embeddings
from .TRAE_bpe import BPETokenizer  # Import for pickle compatibility
from .config import (
    get_model_path, get_tokenizer_type, 
    print_config
)
from .morl import run_interactive, add_subparser

def parse_arguments(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FHE RL Agent")
    
    # Add tokenizer type argument
    parser.add_argument(
        '--tokenizer_type', 
        choices=['dynamic', 'bpe'], 
        default=get_tokenizer_type(),
        help='Tokenizer type to use (default: from config)'
    )
    
    # Add config flag
    parser.add_argument(
        '--show_config', 
        action='store_true',
        help='Show current configuration and exit'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument(
        '--dataset',
        type=str,
        default='./fhe_rl/datasets/final_llm_dataset.txt',
        help='Path to the training expressions file'
    )
    train_parser.add_argument(
        '--n_envs',
        type=int,
        default=8,
        help='Number of parallel environments (default: 8)'
    )
    train_parser.add_argument(
        '--total_timesteps',
        type=int,
        default=2_000_000,
        help='Total environment steps to train for (default: 2_000_000)'
    )
    # ── MORL / reward hyperparameters ─────────────────────────────────────────
    train_parser.add_argument(
        '--n_cycle',
        type=int,
        default=1,
        help=(
            'Biased-preference cycle length N_cycle. '
            'Every N_cycle+1 episodes: N_cycle episodes use the fixed '
            'speed-focus preference [1,0], then 1 episode uses a random '
            'preference from Ω. Set to 0 to always use random preferences. (default: 1)'
        )
    )
    train_parser.add_argument(
        '--n_budget',
        type=int,
        default=5,
        help=(
            'Fixed normalisation budget for the rotation-key cost component '
            'r_keys = (C_keys_old - C_keys_new) / N_budget (default: 5)'
        )
    )
    train_parser.add_argument(
        '--lambda_env',
        type=float,
        default=0.0,
        help=(
            'Weight for the Pareto-envelope bonus '
            'added to the linear reward (default: 0.0)'
        )
    )
    train_parser.add_argument(
        '--lambda_kl',
        type=float,
        default=0.0,
        help=(
            'Weight for the KL-divergence exploration bonus '
            'added to the linear reward (default: 0.0)'
        )
    )
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the agent')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the agent')
    run_parser.add_argument('input_expr_file', help='Input expression file')
    run_parser.add_argument('output_vector_file', help='Output vector file')

    run_parser.add_argument('--w_ops', type=float, default=0.5, help='Weight for operations')
    run_parser.add_argument('--w_keys', type=float, default=0.5, help='Weight for keys')

    add_subparser(subparsers)

    return parser.parse_args(args)


def usage() -> None:
    print(
        "Usage:\n"
        "  python -m fhe_rl train [options]      Train the MORL agent\n"
        "  python -m fhe_rl interactive [--mode {direct,menu}]\n"
        "                                         Optimise FHE circuits interactively\n"
        "                                         direct: single preference -> one circuit\n"
        "                                         menu:   preference range  -> Pareto frontier + selection\n"
        "\n"
        "Train options:\n"
        "  --dataset PATH              Training expressions file\n"
        "  --n_envs  INT               Number of parallel environments (default: 8)\n"
        "  --total_timesteps INT       Total training steps (default: 2_000_000)\n"
        "  --n_cycle INT               Biased-preference cycle length (default: 1)\n"
        "  --n_budget INT              Key-cost normalisation budget (default: 5)\n"
        "  --lambda_env FLOAT          Pareto-envelope bonus weight (default: 0.0)\n"
        "  --lambda_kl  FLOAT          KL-divergence bonus weight (default: 0.0)\n"
        "  --tokenizer_type {dynamic,bpe}\n"
    )
    sys.exit(1)


def load_embeddings_from_config(tokenizer_type=None):
    """Load embeddings using the configuration system"""
    try:
        # Determine the correct embeddings model based on tokenizer type
        if tokenizer_type == "bpe" or (tokenizer_type is None and get_tokenizer_type() == "bpe"):
            embeddings_path = get_model_path("bpe_embeddings_model")
        else:
            embeddings_path = get_model_path("dynamic_embeddings_model")
        
        return load_embeddings(tokenizer_type=tokenizer_type, checkpoint_path=embeddings_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main(args=None):
    """Main function with configuration support"""
    parsed_args = parse_arguments(args)
    
    # Show configuration if requested
    if parsed_args.show_config:
        print_config()
        return
    
    mode = parsed_args.mode
    if not mode:
        usage()

    # ────────────────────────────── TRAIN ─────────────────────────────
    if mode == "train":
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        train_agent(
            expressions_file=parsed_args.dataset,
            embeddings_model=embeddings,
            total_timesteps=parsed_args.total_timesteps,
            num_envs=parsed_args.n_envs,
            n_cycle=parsed_args.n_cycle,
            n_budget=parsed_args.n_budget,
            lambda_env=parsed_args.lambda_env,
            lambda_kl=parsed_args.lambda_kl,
        )

    # ─────────────────────────────── TEST ─────────────────────────────
    elif mode == "test":
        agent_zip = get_model_path("agent_model")
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        test_agent("./fhe_rl/datasets/benchmarks.txt", embeddings, agent_zip)

    # ─────────────────────────────── RUN ──────────────────────────────
    elif mode == "run":
        agent_zip = get_model_path("agent_model")
        input_file = parsed_args.input_expr_file
        output_file = parsed_args.output_vector_file
        embeddings, tokenizer = load_embeddings_from_config(parsed_args.tokenizer_type)
        run_agent(input_file, embeddings, agent_zip, output_file, w_ops=parsed_args.w_ops, w_keys=parsed_args.w_keys)

    elif mode == "interactive":
        run_interactive(mode=getattr(parsed_args, "interactive_mode", None))    
    else:
        print("Invalid command. Use 'train', 'test' or 'run'.")
        usage()




if __name__ == "__main__":
    main()