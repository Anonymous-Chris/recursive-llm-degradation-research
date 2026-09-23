from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def create_checkpoint(
    project_root: Path,
    condition: str,
    seed: int,
    generation: int,
) -> Path:

    run_dir = (
        project_root
        / "results"
        / condition
        / f"seed_{seed}"
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    zip_name = (
        f"{condition}_seed{seed}_G{generation}_checkpoint.zip"
    )

    zip_path = checkpoint_dir / zip_name

    # Files needed to preserve a completed generation
    generation_dir = run_dir / f"g{generation}"

    files = [
        generation_dir / "training_input.csv",
        generation_dir / "adapter" / "adapter_config.json",
        generation_dir / "adapter" / "adapter_model.safetensors",
        run_dir / f"g{generation}_predictions.csv",
        run_dir / f"g{generation}_synthetic.csv",
        run_dir / "run_metadata.json",
    ]

    # Include split metadata when available
    split_metadata = project_root / "data" / "split_metadata.json"
    if split_metadata.exists():
        files.append(split_metadata)

    # Validate
    missing = [str(p) for p in files if not p.exists()]

    if missing:
        print("\nMissing files:")
        for path in missing:
            print("  ", path)

        raise FileNotFoundError(
            "Checkpoint cannot be created because required files are missing."
        )

    # Create checkpoint
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:

        for file in files:

            if file.is_relative_to(project_root):
                arcname = file.relative_to(project_root)
            else:
                arcname = file.name

            z.write(file, arcname=str(arcname))

    print("\n✅ Checkpoint created")
    print("Condition:", condition)
    print("Seed:", seed)
    print("Generation:", generation)
    print("Path:", zip_path)
    print(
        "Size:",
        f"{zip_path.stat().st_size / 1024 / 1024:.2f} MB",
    )

    return zip_path


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Create a generation-level experiment checkpoint."
    )

    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "recursive",
            "human_control",
            "anchor_10",
        ],
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 123],
    )

    parser.add_argument(
        "--generation",
        required=True,
        type=int,
        choices=[0, 1, 2, 3],
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    args = parser.parse_args()

    create_checkpoint(
        project_root=Path(args.project_root).resolve(),
        condition=args.condition,
        seed=args.seed,
        generation=args.generation,
    )


if __name__ == "__main__":
    main()