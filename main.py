import os, sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import yaml
from pipeline import PMCPipeline


def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pipeline = PMCPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()