import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from net.model import PromptIR
from utils.dataset_utils import PromptTrainDataset
from utils.image_io import save_image_tensor
from utils.pytorch_ssim import ssim as pytorch_ssim

try:
    import wandb
except Exception:
    wandb = None


DE_ID_TO_NAME = {
    0: "denoise_15",
    1: "denoise_25",
    2: "denoise_50",
    3: "derain",
    4: "dehaze",
}

SIM_VARIANTS = [
    "input",
    "m_input",
    "m_target",
    "mm_input",
]

RANK_VARIANTS = [
    "input",
    "m_input",
    "mm_input",
]

MODE_TO_DE_TYPES = {
    0: ["denoise_15", "denoise_25", "denoise_50"],
    1: ["derain"],
    2: ["dehaze"],
    3: ["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze"],
}


def _as_abs_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return str((Path(base_dir) / path_value).resolve())


def _ensure_trailing_sep(path_value):
    if path_value.endswith(os.sep):
        return path_value
    return path_value + os.sep


def normalize_dataset_paths(testopt):
    promptir_root = Path(__file__).resolve().parent
    testopt.data_file_dir = _as_abs_path(promptir_root, testopt.data_file_dir)
    testopt.denoise_dir = _ensure_trailing_sep(_as_abs_path(promptir_root, testopt.denoise_dir))
    testopt.derain_dir = _ensure_trailing_sep(_as_abs_path(promptir_root, testopt.derain_dir))
    testopt.dehaze_dir = _ensure_trailing_sep(_as_abs_path(promptir_root, testopt.dehaze_dir))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=3, choices=[0, 1, 2, 3],
                        help="0: denoise, 1: derain, 2: dehaze, 3: all-in-one")
    parser.add_argument("--data_split", type=str, default="test", choices=["train", "val", "test"],
                        help="manifest split used by PromptTrainDataset")
    parser.add_argument("--patch_size", type=int, default=128, help="patch size for center crop")
    parser.add_argument("--degradation_size", type=int, default=-1,
                        help="<=0 means use full split, >0 means sample to fixed size")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help=">0 to stop early for quick smoke tests")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="evaluation device, default is cpu")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="number of gpus to use when --device cuda")

    parser.add_argument("--data_file_dir", type=str, default="data_dir/")
    parser.add_argument("--denoise_dir", type=str, default="/home/huhao/adv_ir/dataset/noise/mix/")
    parser.add_argument("--derain_dir", type=str, default="/home/huhao/adv_ir/dataset/")
    parser.add_argument("--dehaze_dir", type=str, default="/home/huhao/adv_ir/dataset/")

    parser.add_argument("--output_path", type=str, default="/home/huhao/adv_ir/tmp_demo/promptir_test_cpu/")
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="/home/huhao/adv_ir/PromptIR/train_ckpt_8192/epoch=127-step=229376.ckpt",
        help="checkpoint path (absolute path supported)",
    )
    parser.add_argument("--save_restored", action="store_true",
                        help="save restored images grouped by task")
    parser.add_argument("--wandb_project", type=str, default="promptir_test",
                        help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="",
                        help="optional wandb run name")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="disable wandb logging")
    return parser


def resolve_ckpt_path(ckpt_name):
    if os.path.isabs(ckpt_name):
        return ckpt_name
    return os.path.join("ckpt", ckpt_name)


def load_promptir_model(checkpoint_path, device):
    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (TypeError, ValueError, RuntimeError, pickle.UnpicklingError):
        checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint_obj["state_dict"] if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj else checkpoint_obj
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a valid state_dict")

    promptir_state = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            promptir_state[key[4:]] = value

    if len(promptir_state) == 0:
        raise ValueError("no 'net.' prefixed keys found in checkpoint state_dict")

    model = PromptIR(decoder=True)
    missing_keys, unexpected_keys = model.load_state_dict(promptir_state, strict=False)
    if len(unexpected_keys) > 0:
        raise RuntimeError(f"unexpected PromptIR keys while loading checkpoint: {unexpected_keys[:10]}")
    if len(missing_keys) > 0:
        raise RuntimeError(f"missing PromptIR keys while loading checkpoint: {missing_keys[:10]}")

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def compute_psnr_ssim_single(restored, clean):
    mse = torch.mean((restored - clean) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
    ssim_val = pytorch_ssim(torch.clamp(restored, 0, 1), torch.clamp(clean, 0, 1), size_average=True)
    return float(psnr.mean().item()), float(ssim_val.item())


def compute_distribution_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def plot_metric_distribution(records, task_name, output_dir, value_prefix="m_input"):
    output_dir.mkdir(parents=True, exist_ok=True)
    psnr_values = np.asarray([x[f"psnr_{value_prefix}"] for x in records], dtype=np.float64)
    ssim_values = np.asarray([x[f"ssim_{value_prefix}"] for x in records], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(psnr_values, bins=25, color="#2C7FB8", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[0].set_title(f"{task_name} PSNR ({value_prefix} vs target)")
    axes[0].set_xlabel("PSNR")
    axes[0].set_ylabel("Count")

    axes[1].hist(ssim_values, bins=25, color="#41AB5D", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[1].set_title(f"{task_name} SSIM ({value_prefix} vs target)")
    axes[1].set_xlabel("SSIM")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_dir / f"{task_name}_{value_prefix}_distribution.png", dpi=150)
    plt.close(fig)


def rank_variants_by_similarity(record, metric_name):
    items = []
    for name in RANK_VARIANTS:
        items.append((name, float(record[f"{metric_name}_{name}"])))
    # Descending similarity, deterministic tie-break by name.
    items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
    return [x[0] for x in items_sorted], [x[1] for x in items_sorted]


def plot_pair_delta_distribution(deltas, metric_name, level_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_m_input_minus_input = np.asarray([x["m_input_minus_input"] for x in deltas], dtype=np.float64)
    delta_mm_input_minus_m_input = np.asarray([x["mm_input_minus_m_input"] for x in deltas], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].hist(delta_m_input_minus_input, bins=25, color="#E6550D", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[0].set_title(f"{level_name} {metric_name.upper()} Delta: M(input)-input")
    axes[0].set_xlabel("Delta")
    axes[0].set_ylabel("Count")

    axes[1].hist(delta_mm_input_minus_m_input, bins=25, color="#31A354", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[1].set_title(f"{level_name} {metric_name.upper()} Delta: M(M(input))-M(input)")
    axes[1].set_xlabel("Delta")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_dir / f"{level_name}_{metric_name}_pair_delta_distribution.png", dpi=150)
    plt.close(fig)


def save_records_csv(records, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_name", "de_id", "task",
                "psnr_input", "ssim_input",
                "psnr_m_input", "ssim_m_input",
                "psnr_m_target", "ssim_m_target",
                "psnr_mm_input", "ssim_mm_input",
                "rank_psnr", "rank_ssim",
                "psnr_delta_m_input_minus_input", "psnr_delta_mm_input_minus_m_input",
                "ssim_delta_m_input_minus_input", "ssim_delta_mm_input_minus_m_input",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def select_runtime_device(testopt):
    if testopt.device == "cuda" and torch.cuda.is_available():
        gpu_count = max(1, torch.cuda.device_count())
        used_gpus = min(max(1, testopt.num_gpus), gpu_count)
        return torch.device("cuda:0"), used_gpus
    return torch.device("cpu"), 0


def init_wandb_if_needed(testopt, config_dict) -> Optional[object]:
    if testopt.disable_wandb:
        print("[Info] W&B disabled by --disable_wandb")
        return None
    if wandb is None:
        print("[Warn] wandb not installed in current environment; skip cloud logging")
        return None
    try:
        run = wandb.init(
            project=testopt.wandb_project,
            name=(testopt.wandb_run_name or None),
            config=config_dict,
        )
        print(f"[Info] W&B enabled. project={testopt.wandb_project}")
        return run
    except Exception as exc:
        print(f"[Warn] failed to initialize wandb: {exc}")
        return None


def log_to_wandb(wandb_run, summary, records, output_root):
    if wandb_run is None:
        return

    try:
        flat_metrics = {}
        for task_name, task_info in summary.items():
            if not isinstance(task_info, dict):
                continue
            if "count" not in task_info:
                continue
            if not all(variant in task_info for variant in SIM_VARIANTS):
                continue

            flat_metrics[f"{task_name}/count"] = task_info["count"]
            for variant in SIM_VARIANTS:
                flat_metrics[f"{task_name}/{variant}/psnr_mean"] = task_info[variant]["psnr"]["mean"]
                flat_metrics[f"{task_name}/{variant}/ssim_mean"] = task_info[variant]["ssim"]["mean"]

        if "ranking_counts" in summary:
            for metric_name, order_counts in summary["ranking_counts"].items():
                for order_name, count in order_counts.items():
                    flat_metrics[f"ranking/{metric_name}/{order_name}"] = count

        if len(flat_metrics) > 0:
            wandb.log(flat_metrics)

        table = wandb.Table(columns=[
            "sample_name", "de_id", "task",
            "psnr_input", "ssim_input",
            "psnr_m_input", "ssim_m_input",
            "psnr_m_target", "ssim_m_target",
            "psnr_mm_input", "ssim_mm_input",
            "rank_psnr", "rank_ssim",
            "psnr_delta_m_input_minus_input", "psnr_delta_mm_input_minus_m_input",
            "ssim_delta_m_input_minus_input", "ssim_delta_mm_input_minus_m_input",
        ])
        for row in records:
            table.add_data(
                row["sample_name"], row["de_id"], row["task"],
                row["psnr_input"], row["ssim_input"],
                row["psnr_m_input"], row["ssim_m_input"],
                row["psnr_m_target"], row["ssim_m_target"],
                row["psnr_mm_input"], row["ssim_mm_input"],
                row["rank_psnr"], row["rank_ssim"],
                row["psnr_delta_m_input_minus_input"], row["psnr_delta_mm_input_minus_m_input"],
                row["ssim_delta_m_input_minus_input"], row["ssim_delta_mm_input_minus_m_input"],
            )
        wandb.log({"metrics_per_sample": table})

        plots_dir = output_root / "plots"
        for plot_path in sorted(plots_dir.glob("*.png")):
            wandb.log({f"plot/{plot_path.stem}": wandb.Image(str(plot_path))})

        artifact = wandb.Artifact("promptir_test_results", type="evaluation")
        artifact.add_file(str(output_root / "metrics_per_sample.csv"))
        artifact.add_file(str(output_root / "metrics_summary.json"))
        for plot_path in sorted(plots_dir.glob("*.png")):
            artifact.add_file(str(plot_path))
        wandb_run.log_artifact(artifact)
    except Exception as exc:
        print(f"[Warn] failed while logging to wandb: {exc}")


def evaluate(testopt):
    np.random.seed(0)
    torch.manual_seed(0)

    device, used_gpus = select_runtime_device(testopt)
    if device.type == "cuda":
        print(f"[Info] Evaluation device: cuda (gpus={used_gpus})")
    else:
        print("[Info] Evaluation device: cpu")

    ckpt_path = resolve_ckpt_path(testopt.ckpt_name)
    print(f"[Info] Checkpoint: {ckpt_path}")

    testopt.de_type = MODE_TO_DE_TYPES[testopt.mode]
    if testopt.degradation_size <= 0:
        testopt.degradation_size = None
    normalize_dataset_paths(testopt)

    dataset = PromptTrainDataset(testopt)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=False,
        shuffle=False,
        num_workers=testopt.num_workers,
    )

    model = load_promptir_model(ckpt_path, device=device)

    output_root = Path(testopt.output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_if_needed(
        testopt,
        {
            "mode": testopt.mode,
            "data_split": testopt.data_split,
            "device": device.type,
            "num_gpus": used_gpus,
            "ckpt_path": ckpt_path,
            "output_path": str(output_root),
        },
    )

    if device.type == "cuda" and used_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(used_gpus)))

    all_records = []
    with torch.no_grad():
        for idx, ([clean_name, de_id], degrad_patch, clean_patch) in enumerate(tqdm(dataloader)):
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            m_input = model(degrad_patch)
            m_target = model(clean_patch)
            mm_input = model(m_input)

            psnr_input, ssim_input = compute_psnr_ssim_single(degrad_patch, clean_patch)
            psnr_m_input, ssim_m_input = compute_psnr_ssim_single(m_input, clean_patch)
            psnr_m_target, ssim_m_target = compute_psnr_ssim_single(m_target, clean_patch)
            psnr_mm_input, ssim_mm_input = compute_psnr_ssim_single(mm_input, clean_patch)

            de_id_value = int(de_id.item())
            task_name = DE_ID_TO_NAME.get(de_id_value, f"unknown_{de_id_value}")
            sample_name = str(clean_name[0])

            sample_record = {
                "sample_name": sample_name,
                "de_id": de_id_value,
                "task": task_name,
                "psnr_input": float(psnr_input),
                "ssim_input": float(ssim_input),
                "psnr_m_input": float(psnr_m_input),
                "ssim_m_input": float(ssim_m_input),
                "psnr_m_target": float(psnr_m_target),
                "ssim_m_target": float(ssim_m_target),
                "psnr_mm_input": float(psnr_mm_input),
                "ssim_mm_input": float(ssim_mm_input),
            }

            rank_psnr_names, rank_psnr_values = rank_variants_by_similarity(sample_record, "psnr")
            rank_ssim_names, rank_ssim_values = rank_variants_by_similarity(sample_record, "ssim")
            sample_record["rank_psnr"] = ">".join(rank_psnr_names)
            sample_record["rank_ssim"] = ">".join(rank_ssim_names)
            sample_record["psnr_delta_m_input_minus_input"] = float(sample_record["psnr_m_input"] - sample_record["psnr_input"])
            sample_record["psnr_delta_mm_input_minus_m_input"] = float(sample_record["psnr_mm_input"] - sample_record["psnr_m_input"])
            sample_record["ssim_delta_m_input_minus_input"] = float(sample_record["ssim_m_input"] - sample_record["ssim_input"])
            sample_record["ssim_delta_mm_input_minus_m_input"] = float(sample_record["ssim_mm_input"] - sample_record["ssim_m_input"])

            all_records.append(sample_record)

            if testopt.save_restored:
                save_dir = output_root / "restored" / task_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_name = Path(sample_name).stem + ".png"
                save_image_tensor(m_input.cpu(), str(save_dir / save_name))

            if testopt.max_samples > 0 and (idx + 1) >= testopt.max_samples:
                break

    if len(all_records) == 0:
        raise RuntimeError("no samples were evaluated; please check split manifests and dataset paths")

    save_records_csv(all_records, output_root / "metrics_per_sample.csv")

    grouped_records = {}
    for record in all_records:
        grouped_records.setdefault(record["task"], []).append(record)

    summary = {}
    ranking_counts = {
        "psnr": {},
        "ssim": {},
    }
    ranking_counts_by_task = {}
    pair_delta_records = {
        "overall": {
            "psnr": [],
            "ssim": [],
        }
    }
    for task_name, task_records in sorted(grouped_records.items()):
        summary[task_name] = {"count": len(task_records)}
        ranking_counts_by_task[task_name] = {
            "psnr": {},
            "ssim": {},
        }
        pair_delta_records[task_name] = {
            "psnr": [],
            "ssim": [],
        }

        for variant in SIM_VARIANTS:
            summary[task_name][variant] = {
                "psnr": compute_distribution_stats([x[f"psnr_{variant}"] for x in task_records]),
                "ssim": compute_distribution_stats([x[f"ssim_{variant}"] for x in task_records]),
            }
            plot_metric_distribution(task_records, task_name, output_root / "plots", value_prefix=variant)

        for sample in task_records:
            psnr_order = sample["rank_psnr"]
            ssim_order = sample["rank_ssim"]
            ranking_counts["psnr"][psnr_order] = ranking_counts["psnr"].get(psnr_order, 0) + 1
            ranking_counts["ssim"][ssim_order] = ranking_counts["ssim"].get(ssim_order, 0) + 1
            ranking_counts_by_task[task_name]["psnr"][psnr_order] = ranking_counts_by_task[task_name]["psnr"].get(psnr_order, 0) + 1
            ranking_counts_by_task[task_name]["ssim"][ssim_order] = ranking_counts_by_task[task_name]["ssim"].get(ssim_order, 0) + 1

            task_psnr_delta = {
                "m_input_minus_input": sample["psnr_delta_m_input_minus_input"],
                "mm_input_minus_m_input": sample["psnr_delta_mm_input_minus_m_input"],
            }
            task_ssim_delta = {
                "m_input_minus_input": sample["ssim_delta_m_input_minus_input"],
                "mm_input_minus_m_input": sample["ssim_delta_mm_input_minus_m_input"],
            }
            pair_delta_records[task_name]["psnr"].append(task_psnr_delta)
            pair_delta_records[task_name]["ssim"].append(task_ssim_delta)
            pair_delta_records["overall"]["psnr"].append(task_psnr_delta)
            pair_delta_records["overall"]["ssim"].append(task_ssim_delta)

        summary[task_name]["pair_delta"] = {
            "psnr": {
                "m_input_minus_input": compute_distribution_stats([x["psnr_delta_m_input_minus_input"] for x in task_records]),
                "mm_input_minus_m_input": compute_distribution_stats([x["psnr_delta_mm_input_minus_m_input"] for x in task_records]),
            },
            "ssim": {
                "m_input_minus_input": compute_distribution_stats([x["ssim_delta_m_input_minus_input"] for x in task_records]),
                "mm_input_minus_m_input": compute_distribution_stats([x["ssim_delta_mm_input_minus_m_input"] for x in task_records]),
            },
        }

        plot_pair_delta_distribution(pair_delta_records[task_name]["psnr"], "psnr", task_name, output_root / "plots")
        plot_pair_delta_distribution(pair_delta_records[task_name]["ssim"], "ssim", task_name, output_root / "plots")

    summary["overall"] = {
        "count": len(all_records),
    }
    for variant in SIM_VARIANTS:
        summary["overall"][variant] = {
            "psnr": compute_distribution_stats([x[f"psnr_{variant}"] for x in all_records]),
            "ssim": compute_distribution_stats([x[f"ssim_{variant}"] for x in all_records]),
        }
        plot_metric_distribution(all_records, "overall", output_root / "plots", value_prefix=variant)

    summary["overall"]["pair_delta"] = {
        "psnr": {
            "m_input_minus_input": compute_distribution_stats([x["psnr_delta_m_input_minus_input"] for x in all_records]),
            "mm_input_minus_m_input": compute_distribution_stats([x["psnr_delta_mm_input_minus_m_input"] for x in all_records]),
        },
        "ssim": {
            "m_input_minus_input": compute_distribution_stats([x["ssim_delta_m_input_minus_input"] for x in all_records]),
            "mm_input_minus_m_input": compute_distribution_stats([x["ssim_delta_mm_input_minus_m_input"] for x in all_records]),
        },
    }
    plot_pair_delta_distribution(pair_delta_records["overall"]["psnr"], "psnr", "overall", output_root / "plots")
    plot_pair_delta_distribution(pair_delta_records["overall"]["ssim"], "ssim", "overall", output_root / "plots")
    summary["ranking_counts"] = ranking_counts
    summary["ranking_counts_by_task"] = ranking_counts_by_task

    summary_path = output_root / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_to_wandb(wandb_run, summary, all_records, output_root)
    if wandb_run is not None:
        wandb_run.finish()

    print("=" * 60)
    print(f"[Done] Evaluated {len(all_records)} samples")
    for task_name, task_info in summary.items():
        if task_name in ("ranking_counts", "ranking_counts_by_task"):
            continue
        print(
            "[{task}] count={count}, input_psnr={input_psnr:.4f}, m_input_psnr={m_input_psnr:.4f}, mm_input_psnr={mm_input_psnr:.4f}".format(
                task=task_name,
                count=task_info["count"],
                input_psnr=task_info["input"]["psnr"]["mean"],
                m_input_psnr=task_info["m_input"]["psnr"]["mean"],
                mm_input_psnr=task_info["mm_input"]["psnr"]["mean"],
            )
        )
    print(f"[Ranking-PSNR] {summary['ranking_counts']['psnr']}")
    print(f"[Ranking-SSIM] {summary['ranking_counts']['ssim']}")
    print(f"[Saved] {output_root / 'metrics_per_sample.csv'}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {output_root / 'plots'}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    evaluate(args)