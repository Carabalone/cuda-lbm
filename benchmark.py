#!/usr/bin/env python3
"""Portable, one-command CUDA-LBM TGV benchmark campaign."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / ".benchmark"
WARNING_RE = re.compile(r"\[WARNING\]", re.IGNORECASE)
PERF_RE = re.compile(r"Performance:\s*([0-9]+(?:\.[0-9]+)?)\s*MLUPS", re.I)
AVG_RE = re.compile(r"Average timestep:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.I)
STEPS_RE = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*steps/s", re.I)


@dataclass(frozen=True)
class Case:
    label: str
    collision: str
    layout: str

    @property
    def definitions(self) -> list[str]:
        defs = {"BGK": "LBM_COLLISION_BGK", "MRT": "LBM_COLLISION_MRT",
                "MRT_BASELINE": "LBM_COLLISION_MRT_BASELINE"}
        result = [defs[self.collision]] if self.collision != "CM" else []
        if self.layout != "SOA":
            result.append("LBM_LAYOUT_" + self.layout)
        return result


CASES = [
    Case("unop_MRT", "MRT_BASELINE", "AOS"),
    Case("BGK_SoA", "BGK", "SOA"), Case("BGK_AoS", "BGK", "AOS"),
    Case("BGK_CSoA", "BGK", "CSOA"), Case("MRT_opt_SoA", "MRT", "SOA"),
    Case("MRT_opt_AoS", "MRT", "AOS"), Case("MRT_opt_CSoA", "MRT", "CSOA"),
    Case("CM_SoA", "CM", "SOA"), Case("CM_AoS", "CM", "AOS"),
    Case("CM_CSoA", "CM", "CSOA"),
]
CSV_FIELDS = ["gpu_key", "case", "collision", "layout", "sample", "mlups",
              "median_mlups", "average_timestep_ms", "steps_per_second", "status",
              "warmup_steps", "measured_steps", "date", "protocol", "log"]
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def slug(value: str) -> str:
    return re.sub(r"(^_+|_+$)", "", re.sub(r"[^a-z0-9]+", "_", value.lower())) or "unknown"


def gpu_key(value: str) -> str:
    lowered = value.lower()
    for needle, key in (("1050 ti", "gtx1050ti"), ("2080", "rtx2080"), ("h100", "h100")):
        if needle in lowered:
            return key
    return slug(value)


def parse_compute_capability(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\s*\.\s*(\d+)", value)
    if not match:
        match = re.search(r"(?:sm_)?(\d)(\d)", value.lower())
    if not match:
        raise ValueError(f"Cannot parse compute capability: {value!r}")
    return int(match.group(1)), int(match.group(2))


def parse_visible_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if not value.strip() or value.strip() == "-1":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def contains_warning(text: str) -> bool:
    return WARNING_RE.search(text) is not None


def command_output(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=ROOT, env=env, text=True,
                                   stderr=subprocess.STDOUT).strip()


def query_gpus() -> list[dict[str, str]]:
    fields = ["index", "name", "uuid", "memory.total", "compute_cap", "driver_version",
              "temperature.gpu", "clocks.current.sm", "clocks.current.memory"]
    try:
        output = command_output(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                 "--format=csv,noheader,nounits"])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("No NVIDIA GPU was detected. Check the driver, nvidia-smi, and "
                           "container GPU exposure.") from exc
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    return [{key: value.strip() for key, value in zip(fields, row)} for row in rows]


def select_gpu(gpus: list[dict[str, str]], requested: int | None,
               visible_env: str | None) -> tuple[dict[str, str], str]:
    visible = parse_visible_devices(visible_env)
    candidates = gpus
    tokens: list[str]
    if visible is None:
        tokens = [gpu["index"] for gpu in gpus]
    else:
        tokens = visible
        candidates = []
        for token in tokens:
            matches = [gpu for gpu in gpus if gpu["index"] == token or
                       gpu["uuid"] == token or gpu["uuid"].startswith(token)]
            if matches:
                candidates.append(matches[0])
        if not candidates:
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no GPU visible to nvidia-smi.")
    if requested is None and len(candidates) != 1:
        raise RuntimeError(f"{len(candidates)} GPUs are visible; select one with --device INDEX.")
    ordinal = 0 if requested is None else requested
    if ordinal < 0 or ordinal >= len(candidates):
        raise RuntimeError(f"--device {ordinal} is outside the {len(candidates)} visible GPU(s).")
    return candidates[ordinal], tokens[ordinal]


def nvcc_info() -> dict[str, str]:
    try:
        version = command_output(["nvcc", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvcc was not found. Install a CUDA toolkit that supports the GPU.") from exc
    release = re.search(r"release\s+([0-9.]+)", version)
    return {"path": shutil.which("nvcc") or "nvcc", "version_output": version,
            "toolkit_version": release.group(1) if release else "unknown"}


def validate_architecture(arch: str, toolkit: dict[str, str]) -> None:
    try:
        supported = command_output(["nvcc", "--list-gpu-code"])
    except subprocess.CalledProcessError:
        supported = ""
    code = "sm_" + arch.removeprefix("sm_")
    if code not in supported.split():
        hint = " CUDA 12.8+ is required for Blackwell sm_100." if code == "sm_100" else ""
        raise RuntimeError(f"nvcc {toolkit['toolkit_version']} does not accept {code}. Install a "
                           f"toolkit supporting this GPU; no architecture fallback was used.{hint}")


def git_metadata() -> dict[str, object]:
    try:
        commit = command_output(["git", "rev-parse", "HEAD"])
        dirty = bool(command_output(["git", "status", "--porcelain"]))
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": None}


def run_streamed(command: list[str], log_path: Path, timeout: float | None,
                 env: dict[str, str]) -> tuple[str, str, int | None, float]:
    """Stream output and terminate the whole process group as soon as a warning appears."""
    start = time.monotonic()
    global ACTIVE_PROCESS
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    ACTIVE_PROCESS = proc
    lines: list[str] = []
    messages: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=reader, daemon=True).start()
    status = "ok"
    terminated = False
    termination_time = 0.0
    with log_path.open("w", encoding="utf-8") as log:
        while True:
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                lines.append(line)
                if contains_warning(line) and not terminated:
                    status, terminated = "warning_terminated", True
                    termination_time = time.monotonic()
                    os.killpg(proc.pid, signal.SIGINT)
            if timeout and time.monotonic() - start > timeout and not terminated:
                status, terminated = "timeout", True
                termination_time = time.monotonic()
                os.killpg(proc.pid, signal.SIGINT)
            if terminated and proc.poll() is None and time.monotonic() - termination_time > 2:
                os.killpg(proc.pid, signal.SIGTERM)
            if terminated and proc.poll() is None and time.monotonic() - termination_time > 4:
                os.killpg(proc.pid, signal.SIGKILL)
        returncode = proc.wait()
    ACTIVE_PROCESS = None
    text = "".join(lines)
    if status == "ok" and returncode != 0:
        status = "signal_interruption" if returncode < 0 else "process_failure"
    return status, text, returncode, time.monotonic() - start


def make_command(case: Case, arch: str, grid: int, build_dir: Path) -> list[str]:
    definitions = [f"-DLBM_BENCHMARK_GRID={grid}"] + [f"-D{x}" for x in case.definitions]
    return ["make", "-j" + str(max(1, os.cpu_count() or 1)), f"GPU_ARCH={arch}",
            f"BINDIR={build_dir}", "EXTRA_DEFINES=" + " ".join(definitions), "all"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def parse_legacy_result(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse metadata comments and rows emitted by the old shell runner."""
    metadata: dict[str, str] = {}
    content = path.read_text(encoding="utf-8").splitlines()
    for line in content:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[slug(key)] = value.strip()
    rows = list(csv.DictReader(line for line in content if line and not line.startswith("#")))
    return metadata, rows


def render_plots(run_dir: Path, current_key: str) -> None:
    import matplotlib.pyplot as plt
    datasets: dict[str, list[dict[str, str]]] = {}
    for path in sorted((ROOT / "benchmark_data").glob("*.csv")) + [run_dir / "results.csv"]:
        rows = load_rows(path)
        if rows and "gpu_key" in rows[0]:
            datasets.setdefault(rows[0]["gpu_key"], []).extend(rows)
    pretty = {"gtx1050ti": "GTX 1050 Ti", "rtx2080": "RTX 2080", "h100": "H100"}
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    for key in ("gtx1050ti", "rtx2080", "h100"):
        rows = datasets.get(key, [])
        med = {row["case"]: float(row["median_mlups"]) for row in rows
               if row.get("median_mlups") and row.get("status") == "ok"}
        fig, ax = plt.subplots(figsize=(11, 5.5))
        labels = [case.label for case in CASES]
        values = [med.get(label, 0) for label in labels]
        bars = ax.bar(labels, values, color=["#777777"] + ["#2878b5"] * 9)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        ax.set(title=f"128³ TGV benchmark — {pretty.get(key, key)}", ylabel="MLUPS")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(plots / f"matrix_{key}.{ext}", dpi=180)
        plt.close(fig)
    keys = ["gtx1050ti", "rtx2080", "h100"]
    operators = [("BGK_SoA", "BGK"), ("MRT_opt_SoA", "MRT"), ("CM_SoA", "CM")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = .24
    base = {}
    for row in datasets.get("gtx1050ti", []):
        if row.get("median_mlups"):
            base[row["case"]] = float(row["median_mlups"])
    for offset, (case_name, operator) in enumerate(operators):
        values = []
        for key in keys:
            matches = [r for r in datasets.get(key, []) if r["case"] == case_name and
                       r.get("median_mlups") and r.get("status") == "ok"]
            values.append(float(matches[-1]["median_mlups"]) if matches else 0)
        xs = [i + (offset - 1) * width for i in range(3)]
        bars = ax.bar(xs, values, width, label=operator)
        labels = [f"{v:.1f}\n{v/base[case_name]:.1f}×" if v and key != "gtx1050ti" else
                  (f"{v:.1f}" if v else "n/a") for v, key in zip(values, keys)]
        ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
    ax.set_xticks(range(3), [pretty[k] for k in keys])
    ax.set_ylabel("MLUPS")
    ax.set_title("Optimized SoA throughput (labels: MLUPS and speedup vs GTX 1050 Ti)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(plots / f"summary_soa_by_gpu.{ext}", dpi=180)
    plt.close(fig)


def plotting_python(no_bootstrap: bool) -> Path:
    try:
        import matplotlib  # noqa: F401
        return Path(sys.executable)
    except ImportError:
        if no_bootstrap:
            raise RuntimeError("matplotlib is unavailable and --no-bootstrap was requested. "
                               "Install benchmark-requirements.txt first.")
    venv = DEFAULT_OUTPUT / ".venv"
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
    subprocess.check_call([str(python), "-m", "pip", "install", "-r",
                           str(ROOT / "benchmark-requirements.txt")])
    return python


def write_summary(run_dir: Path, metadata: dict[str, object], rows: list[dict[str, object]]) -> None:
    medians = {}
    for case in CASES:
        match = [r for r in rows if r["case"] == case.label and r["median_mlups"] != ""]
        medians[case.label] = match[-1]["median_mlups"] if match else "—"
    lines = ["# CUDA-LBM benchmark", "", f"GPU: {metadata['gpu']['name']} ({metadata['arch']})",
             f"Grid: {metadata['grid']}³; window: 100 warm-up + 250 measured steps", "",
             "| Case | Median MLUPS |", "|---|---:|"]
    lines += [f"| {case.label} | {medians[case.label]} |" for case in CASES]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, help="CUDA-visible device ordinal")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per sample")
    parser.add_argument("--cooldown-run", type=float, default=2.0)
    parser.add_argument("--cooldown-case", type=float, default=5.0)
    parser.add_argument("--plot-only", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plot_only:
        render_plots(args.plot_only, "h100")
        return 0
    gpus = query_gpus()
    gpu, visibility_token = select_gpu(gpus, args.device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    major, minor = parse_compute_capability(gpu["compute_cap"])
    arch = f"sm_{major}{minor}"
    toolkit = nvcc_info()
    validate_architecture(arch, toolkit)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output.resolve() / f"{timestamp}_{slug(gpu['name'])}_{arch}"
    (run_dir / "raw" / "build").mkdir(parents=True)
    (run_dir / "raw" / "runs").mkdir(parents=True)
    (run_dir / "plots").mkdir(parents=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visibility_token
    compiler_flags = "-O3 -rdc=true --std=c++17 " + arch
    metadata: dict[str, object] = {
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "grid": args.grid,
        "arch": arch, "gpu": gpu, "cuda_visible_devices_original": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_selected": visibility_token, "mig": "MIG-" in gpu["uuid"],
        "toolkit": toolkit, "compiler_flags": compiler_flags, "host": platform.uname()._asdict(),
        "git": git_metadata(), "dry_run": args.dry_run,
        "manifest": []}
    rows: list[dict[str, object]] = []
    interrupted = False
    try:
        for case in CASES:
            build_dir = args.output.resolve() / ".build" / arch / f"grid{args.grid}" / slug(case.label)
            build = make_command(case, arch, args.grid, build_dir)
            binary = build_dir / "lbm_solver"
            entry = {"case": asdict(case), "build_command": build,
                     "run_command": [str(binary), "--benchmark"], "samples": 3}
            metadata["manifest"].append(entry)  # type: ignore[union-attr]
            if args.dry_run:
                print(json.dumps(entry, indent=2))
                continue
            print(f"\n=== Building {case.label} ({case.collision}/{case.layout}) ===")
            build_log = run_dir / "raw" / "build" / f"{case.label}.log"
            status, _, rc, _ = run_streamed(build, build_log, None, env)
            if status != "ok":
                for sample in range(1, 4):
                    rows.append({"gpu_key": gpu_key(gpu["name"]), "case": case.label,
                                 "collision": case.collision, "layout": case.layout, "sample": sample,
                                 "mlups": "", "median_mlups": "", "average_timestep_ms": "",
                                 "steps_per_second": "", "status": "build_failure", "warmup_steps": 100,
                                 "measured_steps": 250, "date": timestamp, "protocol": "100+250",
                                 "log": str(build_log.relative_to(run_dir))})
                continue
            if args.cooldown_case:
                time.sleep(args.cooldown_case)
            case_rows = []
            for sample in range(1, 4):
                log = run_dir / "raw" / "runs" / f"{case.label}_sample{sample}.log"
                print(f"--- {case.label}: sample {sample}/3 ---")
                status, output, rc, elapsed = run_streamed([str(binary), "--benchmark"], log,
                                                           args.timeout, env)
                perf, avg, steps = PERF_RE.search(output), AVG_RE.search(output), STEPS_RE.search(output)
                if status == "ok" and not perf:
                    status = "process_failure"
                row = {"gpu_key": gpu_key(gpu["name"]), "case": case.label,
                       "collision": case.collision, "layout": case.layout, "sample": sample,
                       "mlups": perf.group(1) if perf else "", "median_mlups": "",
                       "average_timestep_ms": avg.group(1) if avg else "",
                       "steps_per_second": steps.group(1) if steps else "", "status": status,
                       "warmup_steps": 100, "measured_steps": 250, "date": timestamp,
                       "protocol": "100+250", "log": str(log.relative_to(run_dir))}
                rows.append(row); case_rows.append(row)
                write_csv(run_dir / "results.csv", rows)
                if status != "ok":
                    break
                if sample < 3 and args.cooldown_run:
                    time.sleep(args.cooldown_run)
            valid = [float(r["mlups"]) for r in case_rows if r["status"] == "ok"]
            if len(valid) == 3:
                value = f"{median(valid):.3f}"
                for row in case_rows:
                    row["median_mlups"] = value
            write_csv(run_dir / "results.csv", rows)
    except KeyboardInterrupt:
        interrupted = True
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGINT)
            try:
                ACTIVE_PROCESS.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
        print("\nInterrupted; packaging partial results.", file=sys.stderr)
    finally:
        metadata["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        metadata["interrupted"] = interrupted
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if not (run_dir / "results.csv").exists():
            write_csv(run_dir / "results.csv", rows)
        write_summary(run_dir, metadata, rows)
        if not args.dry_run:
            try:
                python = plotting_python(args.no_bootstrap)
                if python == Path(sys.executable):
                    render_plots(run_dir, gpu_key(gpu["name"]))
                else:
                    subprocess.check_call([str(python), str(Path(__file__).resolve()), "--plot-only", str(run_dir)])
            except Exception as exc:
                metadata["plot_error"] = str(exc)
                (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                print(f"Plotting failed: {exc}", file=sys.stderr)
        archive = run_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(run_dir, arcname=run_dir.name)
        print(f"Results: {run_dir}\nBundle:  {archive}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"benchmark.py: {exc}", file=sys.stderr)
        raise SystemExit(2)
