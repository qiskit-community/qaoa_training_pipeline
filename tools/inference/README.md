# Publisher tooling for the inference model zoo

These scripts maintain the HuggingFace side of the model zoo. They are for
whoever publishes the bundles — nothing in `qaoa_training_pipeline` imports
them, and they are not part of the installed package.

Run them from the repository root, in an environment where the package is
installed (`pip install -e ".[inference]"`) and a HuggingFace token is available
(`hf auth login`, or `HF_TOKEN`):

| Script | What it does |
| --- | --- |
| `upload_to_hf.py --source <dir>` | Creates/updates one repo per bundle from a local export tree laid out as `<dir>/<model>/p<p>/`, strips the private `checkpoint` field from `model_config.json`, and pins the resulting commit sha. |
| `hf_setup.py` | Re-reads every repo's current head and pins it as `revision`. Reports which bundles are still unpublished. |
| `model_keys.py` | Shared helper: turns `--only gcn` / `gcn/p3` / `all` into bundle keys. Not run directly. |

Both commands take `--only <key|architecture|all>` and `--dry-run`. They read
and write whichever setup file the runtime resolves: `$QAOA_HF_SETUP` when it is
set, otherwise the packaged
`qaoa_training_pipeline/inference/huggingface/hf_setup.json`. When writing the
packaged file, commit it after a successful upload, since it is what pins the
runtime to a reproducible revision.

A bundle with `revision: null` is not published yet, and the runtime refuses to
download it, naming the bundle. To work against bundles outside the shipped
zoo, write your own setup file and point `$QAOA_HF_SETUP` at it rather than
editing the tracked one.
