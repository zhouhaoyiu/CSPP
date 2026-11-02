"""Inspect a checkpoint through the current package."""

from __future__ import annotations

import argparse
import json

from cspp.checkpoint import load_checkpoint_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model, checkpoint = load_checkpoint_model(args.checkpoint)
    print(
        json.dumps(
            {
                "model_class": checkpoint["model_class"],
                "parameters": sum(value.numel() for value in model.parameters()),
                "state_size_scalars": model.state_size_scalars,
                "model_config": model.config.as_dict(),
                "candidate_policy": checkpoint.get("candidate_policy"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
