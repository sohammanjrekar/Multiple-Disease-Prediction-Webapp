#!/usr/bin/env python3
import argparse
import yaml
from training import run_full_pipeline
from config_utils import load_config
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config.yml')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    cfg = load_config(args.config)
    run_full_pipeline(cfg)

if __name__ == "__main__":
    main()
