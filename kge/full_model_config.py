"""Load the frozen Hype-DTI configuration used by the reported experiments."""

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hype_dti_full.json"
_DERIVED_KEYS = {"dataset", "device", "data_path", "model_path", "pkl_path"}


def add_full_model_args(parser):
    """Expose runtime locations only; model and training choices stay frozen."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        defaults = json.load(config_file)
    for key in _DERIVED_KEYS:
        defaults.pop(key, None)

    parser.set_defaults(**defaults)
    parser.add_argument("--dataset", default="ago_10shots_0")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id; use -1 for CPU.")
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    parser.add_argument(
        "--load_kge_model",
        default=defaults.get("load_kge_model"),
        help="Pretrained KG checkpoint filename under var_models/.",
    )
    parser.add_argument(
        "--filter-protected-pseudo-candidates",
        action="store_true",
        help=(
            "Exclude validation/test pairs from pseudo-label candidates; cold splits "
            "also exclude evaluation-only entities. Recommended for new audit runs."
        ),
    )
    return parser
