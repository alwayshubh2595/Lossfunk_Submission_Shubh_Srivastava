#!/usr/bin/env python3
"""Generate encoder-swap phase-B commands (v4 replan: ou only, sources are
the checkpointed ou/sigreg cells from the core sweep)."""
import json
from pathlib import Path

lam = json.loads(Path("lambda_map.json").read_text()).get("ou:sigreg", 0.1)
cmds = []
for seed in range(3):
    for enc_proto in ["1step", "bptt"]:
        ckpt = Path(f"runs/ou__sigreg__{enc_proto}__seed{seed}/model.pt")
        if not ckpt.exists():
            print(f"# MISSING CHECKPOINT: {ckpt}")
            continue
        for dyn_proto in ["1step", "bptt"]:
            cmds.append(
                f"/usr/bin/python3 -m src.train --env ou --regularizer sigreg "
                f"--protocol {dyn_proto} --seed {seed} --lambda-reg {lam} "
                f"--steps 10000 --n-train 3000 --n-val 512 --n-test 512 "
                f"--load-encoder-from {ckpt} --output-dir swap_runs"
            )
print("\n".join(cmds))
