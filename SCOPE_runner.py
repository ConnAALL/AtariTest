import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def build_config(
    comp_type: str,
    nonlinear_type: str,
    mapping_type: str,
    steps: int,
    generations: int,
    game: str,
    frameskip: int,
    repeat_action_prob: float,
    popsize: int | None,
) -> dict:
    comp_cfg: dict = {"type": comp_type}
    # Provide minimal, non-swept defaults. Always include k for shaping compatibility.
    comp_cfg.update({"k": 75, "channel_last": True})

    if comp_type in ("dct", "dft"):
        comp_cfg.update({
            "norm": "ortho",
            "channels": 1,
        })
    elif comp_type == "wavelet":
        comp_cfg.update({
            "wavelet": "bior4.4",
            "wavelet_levels": 3,
            "mode": "reflect",
        })
    elif comp_type == "conv":
        # Use a simple Gaussian kernel by default; SCOPE_labs extracts k×k from output.
        comp_cfg.update({
            "kernel_type": "gaussian",
            "kernel_size": 5,
            "sigma": 1.0,
        })
    elif comp_type == "aed":
        # Let SCOPE_labs build a small AED on the fly when no checkpoint_path is provided.
        comp_cfg.update({
            "checkpoint_path": "tmp/flax_ckpt/flax-checkpointing",
            "encoder_features": [32, 64, 32],
            "encoder_kernels": [[3, 3], [3, 3], [3, 3]],
            "encoder_strides": [[2, 2], [2, 2], [2, 2]],
            "decoder_features": [32, 32, 1],
            "decoder_kernels": [[4, 4], [4, 4], [4, 4]],
            "decoder_strides": [[2, 2], [2, 2], [2, 2]],
        })
    elif comp_type == "gaussian":
        # Caution: May require CUDA/GPU. Provide conservative defaults.
        comp_cfg.update({
            "num_gaussians": 1000,
            "init_scale": 49.0,
            "device": "cuda",
        })

    # Nonlinearity (no parameter sweep)
    if nonlinear_type == "sparsifier":
        nonlinear_cfg = {"type": "sparsifier", "q": 90.0}
    elif nonlinear_type == "quantizer":
        nonlinear_cfg = {"type": "quantizer", "num_levels": 256}
    elif nonlinear_type == "dropout":
        # rate 0.0 to avoid PRNG key requirements, still goes through the path.
        nonlinear_cfg = {"type": "dropout", "rate": 0.0}
    else:
        raise ValueError(f"Unsupported nonlinear type: {nonlinear_type}")

    mapping_cfg = {"type": mapping_type}

    cfg = {
        "name": f"{comp_type}-{nonlinear_type}-{mapping_type}",
        "trainer": "SCOPE_labs",
        "safe_for_atari_train": False,
        "comp_cfg": comp_cfg,
        "nonlinear_cfg": nonlinear_cfg,
        "mapping_cfg": mapping_cfg,
        "game": game,
        "frameskip": frameskip,
        "repeat_action_prob": repeat_action_prob,
        "steps": steps,
        "min_gen": generations,
    }
    if popsize is not None:
        cfg["popsize"] = int(popsize)
    return cfg


def run_scope_with_config(scope_path: Path, cfg: dict) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(cfg, tf)
        tf.flush()
        cfg_path = Path(tf.name)
    print(f"\n=== Running {cfg['name']} ===")
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(scope_path), "--config", str(cfg_path)],
            check=False,
        )
        if proc.returncode != 0:
            print(f"[ERROR] Return code {proc.returncode} for {cfg['name']}")
        return proc.returncode
    finally:
        try:
            cfg_path.unlink(missing_ok=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run SCOPE_labs.py across compressor/nonlinearity/mapping combinations.")
    parser.add_argument("--steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--generations", type=int, default=100, help="Generations per combo")
    parser.add_argument("--popsize", type=int, default=None, help="Population size; if omitted, CMA-ES default is used")
    parser.add_argument("--game", type=str, default="ALE/SpaceInvaders-v5", help="Gymnasium game id")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip")
    parser.add_argument("--repeat_action_prob", type=float, default=0.0, help="Repeat action probability")
    parser.add_argument("--include_gaussian", action="store_true", help="Include Gaussian compression combos (may require CUDA)")
    parser.add_argument("--dry-run", action="store_true", help="List planned runs without executing them")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    scope_path = repo_root / "SCOPE_labs.py"

    # Enumerate combinations
    compressor_types = ["dct", "dft", "wavelet", "conv", "aed"]
    if args.include_gaussian:
        compressor_types.append("gaussian")
    nonlinear_types = ["sparsifier", "quantizer", "dropout"]
    mapping_types = ["ShapingMapper"]

    results: list[tuple[str, int]] = []
    for comp in compressor_types:
        for nl in nonlinear_types:
            for mapper in mapping_types:
                cfg = build_config(
                    comp_type=comp,
                    nonlinear_type=nl,
                    mapping_type=mapper,
                    steps=args.steps,
                    generations=args.generations,
                    game=args.game,
                    frameskip=args.frameskip,
                    repeat_action_prob=args.repeat_action_prob,
                    popsize=args.popsize,
                )
                if args.dry_run:
                    print(f"[DRY-RUN] Would run: {cfg['name']}")
                    results.append((cfg["name"], 0))
                    continue
                rc = run_scope_with_config(scope_path, cfg)
                results.append((cfg["name"], rc))

    print("\n=== Summary ===")
    for name, rc in results:
        status = "OK" if rc == 0 else f"FAILED({rc})"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()


