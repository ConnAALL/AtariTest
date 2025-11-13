import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def build_config(comp_type: str, nonlinear_type: str, mapping_type: str, steps: int, generations: int, game: str, frameskip: int, repeat_action_prob: float, popsize: int | None) -> dict:
    comp_cfg: dict = {"type": comp_type}
    # Provide only required/default fields (no parameter sweeps). DCT/DFT need k.
    if comp_type in ("dct", "dft"):
        comp_cfg.update({
            "k": 75,
            "channel_last": True,
            "norm": "ortho",
            "channels": 1
        })
    # Nonlinear defaults (no sweeps)
    if nonlinear_type == "sparsifier":
        nonlinear_cfg = {"type": "sparsifier", "q": 90.0}
    elif nonlinear_type == "quantizer":
        nonlinear_cfg = {"type": "quantizer", "num_levels": 256}
    elif nonlinear_type == "dropout":
        # Use rate 0.0 to avoid needing a PRNG key; still exercises the path.
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
        "min_gen": generations
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
    parser = argparse.ArgumentParser(description="Run SCOPE_labs.py across all compressor/nonlinearity/shaping combinations.")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode to pass to SCOPE config")
    parser.add_argument("--generations", type=int, default=3, help="Generations to run per combo")
    parser.add_argument("--popsize", type=int, default=3, help="Population size (number of individuals). If omitted, CMA-ES default is used.")
    parser.add_argument("--game", type=str, default="ALE/SpaceInvaders-v5", help="Gymnasium game id")
    parser.add_argument("--frameskip", type=int, default=4, help="Frameskip")
    parser.add_argument("--repeat_action_prob", type=float, default=0.0, help="Repeat action probability")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    scope_path = repo_root / "SCOPE_labs.py"

    # Enumerate method combinations (no parameter sweeps)
    compressor_types = ["dct", "dft", "wavelet", "conv", "gaussian"]
    nonlinear_types = ["sparsifier", "quantizer", "dropout"]
    shaping_types = ["ShapingMapper"]  # currently only one available

    supported_for_shaping = {"dct", "dft"}  # SCOPE_labs shaping expects k x k real-valued input

    results: list[tuple[str, int]] = []
    for comp in compressor_types:
        if comp not in supported_for_shaping:
            print(f"\n[SKIP] Compressor '{comp}' not supported in current shaping pipeline (expects k×k).")
            continue
        for nl in nonlinear_types:
            for mapper in shaping_types:
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
                rc = run_scope_with_config(scope_path, cfg)
                results.append((cfg["name"], rc))

    print("\n=== Summary ===")
    for name, rc in results:
        status = "OK" if rc == 0 else f"FAILED({rc})"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()


