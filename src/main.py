import argparse
import logging
import os
import sys
from pathlib import Path
import glob
from typing import List
from tqdm import tqdm

from train import train_model, test_model


def get_available_versions() -> List[str]:
    dataset_files = glob.glob('data/dataset_*.csv')
    versions: List[str] = []
    for file in dataset_files:
        filename = os.path.basename(file)
        if filename.startswith('dataset_') and filename.endswith('.csv'):
            version = filename[8:-4]
            versions.append(version)
    return sorted(versions)


def train_command(args: argparse.Namespace) -> None:
    Path('models').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    version: str = args.version
    
    if version == 'all':
        available_versions: List[str] = get_available_versions()
        
        if not available_versions:
            logging.error("No dataset versions found in data/ directory")
            sys.exit(1)
        
        for v in tqdm(available_versions, desc="Training versions"):
            try:
                train_model(v)
                test_model(v)
            except Exception as e:
                logging.error(f"Failed to train version {v}: {e}")
                if args.stop_on_error:
                    sys.exit(1)
                continue
    else:
        if not os.path.exists(f'data/dataset_{version}.csv'):
            available: List[str] = get_available_versions()
            logging.error(f"Invalid version '{version}' - dataset file not found. Available: {', '.join(available) if available else 'none'}")
            sys.exit(1)
        
        try:
            train_model(version)
            test_model(version)
        except Exception as e:
            logging.error(f"Training failed for version {version}: {e}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model(s)')
    train_parser.add_argument(
        '--version', 
        default='all',
        help='Model version to train (v1, v2, v3, etc.) or "all" to train all available versions'
    )
    train_parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop training all versions if one fails (only applies when --version=all)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()