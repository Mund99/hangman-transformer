"""
Real-time training progress monitor.

Automatically finds the latest run under logs/latest/ and tails whichever
stage is currently active (supervised → rl). Replays existing content first,
then streams new lines live.

Usage
─────
    python scripts/watch_progress.py                  # latest run, active stage
    python scripts/watch_progress.py --stage sup      # supervised log only
    python scripts/watch_progress.py --stage rl       # rl log only
    python scripts/watch_progress.py --run 20260514_1106  # specific run
    python scripts/watch_progress.py --list           # list all runs
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RED    = "\033[91m"
WHITE  = "\033[97m"
MAGENTA = "\033[95m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}"

def _header(text: str, width: int = 64) -> str:
    pad = max(0, width - len(text) - 4)
    return _c(BOLD, f"┌── {text} {'─' * pad}┐")

def _footer(width: int = 64) -> str:
    return _c(BOLD, "└" + "─" * (width - 1) + "┘")


# ── line patterns ─────────────────────────────────────────────────────────────
RE_STEP_SUP = re.compile(r"\[step\s+(\d+)/(\d+)\].*loss=([\d.]+).*elapsed=([\d.]+)s")
RE_STEP_RL  = re.compile(r"\[rl step\s+(\d+)/(\d+)\].*win\(batch\)=([\d.]+)%.*win_ma=([\d.]+)%")
RE_EVAL     = re.compile(r"\[eval.*\]\s+val_win_rate=([\d.]+)%")
RE_SAVED    = re.compile(r"→ saved .+\(val=([\d.]+)%\)")
RE_DONE     = re.compile(r"\[done.*\] best val win-rate: ([\d.]+)%")
RE_DEMO_HDR = re.compile(r"\[demo step=\d+\]")
RE_DEMO_LN  = re.compile(r"\[demo\]")


def _strip_ts(line: str) -> str:
    m = re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| \w+ \| (.+)$", line)
    return m.group(1) if m else line


def format_line(raw: str) -> str | None:
    line = _strip_ts(raw.rstrip())

    if RE_DEMO_HDR.search(line):
        return _c(MAGENTA + BOLD, f"\n{line}")
    if RE_DEMO_LN.search(line):
        return _c(MAGENTA, line)
    if RE_SAVED.search(line):
        return _c(GREEN + BOLD, f"  {line}")
    if RE_DONE.search(line):
        return _c(GREEN + BOLD, line)
    if RE_EVAL.search(line):
        wr = float(RE_EVAL.search(line).group(1))
        colour = GREEN if wr >= 60 else CYAN if wr >= 40 else YELLOW
        return _c(colour, line)
    if RE_STEP_RL.search(line):
        return _c(BLUE, line)
    if RE_STEP_SUP.search(line):
        return _c(DIM, line)
    if re.search(r"\[(device|data|model|init)\]", line):
        return _c(WHITE, line)
    if "ERROR" in line:
        return _c(RED, line)
    if not line.strip():
        return ""
    return _c(DIM, line)


# ── per-stage summary ─────────────────────────────────────────────────────────
class StageSummary:
    def __init__(self, stage: str):
        self.stage = stage          # "supervised" | "rl"
        self.step = 0
        self.total = 0
        self.best_val = 0.0
        self.last_val = 0.0
        self.last_loss = 0.0
        self.last_win_ma = 0.0
        self.checkpoints = 0
        self.done = False

    def ingest(self, line: str):
        line = _strip_ts(line)
        m = RE_STEP_SUP.search(line)
        if m:
            self.step, self.total = int(m.group(1)), int(m.group(2))
            self.last_loss = float(m.group(3))
            return
        m = RE_STEP_RL.search(line)
        if m:
            self.step, self.total = int(m.group(1)), int(m.group(2))
            self.last_win_ma = float(m.group(4))
            return
        m = RE_EVAL.search(line)
        if m:
            self.last_val = float(m.group(1))
            self.best_val = max(self.best_val, self.last_val)
            return
        if RE_SAVED.search(line):
            self.checkpoints += 1
        if RE_DONE.search(line):
            self.done = True

    def render(self) -> str:
        label = "Stage 1 — Supervised" if self.stage == "supervised" else "Stage 2 — RL"
        status = _c(GREEN, "done") if self.done else _c(YELLOW, "running")
        lines = [f"{_c(BOLD, label)}  [{status}]"]

        if self.total > 0:
            pct = self.step / self.total * 100
            filled = int(28 * pct / 100)
            bar = "█" * filled + "░" * (28 - filled)
            lines.append(f"  Progress  {bar} {pct:5.1f}%  ({self.step:,}/{self.total:,})")

        if self.best_val > 0:
            lines.append(f"  Best val  {_c(GREEN, f'{self.best_val:.2f}%')}  "
                         f"(last: {self.last_val:.2f}%)")

        if self.stage == "supervised" and self.last_loss > 0:
            lines.append(f"  Loss      {self.last_loss:.4f}")
        elif self.stage == "rl" and self.last_win_ma > 0:
            lines.append(f"  Win MA    {self.last_win_ma:.1f}%")

        if self.checkpoints:
            lines.append(f"  Saved     {self.checkpoints} checkpoint(s)")

        return "\n".join(lines)


# ── run discovery ─────────────────────────────────────────────────────────────
def _all_runs(logs: Path) -> list[Path]:
    return sorted(
        [d for d in logs.iterdir() if d.is_dir() and re.match(r"\d{8}_\d{4}", d.name)],
        key=lambda d: d.name,
        reverse=True,
    )


def _latest_run(logs: Path) -> Path | None:
    latest = logs / "latest"
    if latest.is_symlink() and latest.exists():
        return latest.resolve()
    runs = _all_runs(logs)
    return runs[0] if runs else None


def _active_log(run_dir: Path, stage: str | None) -> Path | None:
    """Return the log to tail: honour --stage, otherwise pick the active one."""
    sup = run_dir / "supervised.log"
    rl  = run_dir / "rl.log"

    if stage == "sup":
        return sup if sup.exists() else None
    if stage == "rl":
        return rl if rl.exists() else None

    # auto: prefer rl if it exists, else supervised
    if rl.exists():
        return rl
    if sup.exists():
        return sup
    return None


# ── tailer ────────────────────────────────────────────────────────────────────
def tail_log(path: Path, summary: StageSummary) -> None:
    with open(path) as f:
        for raw in f:
            summary.ingest(raw)
            out = format_line(raw)
            if out is not None:
                print(out)

        print(_c(BOLD, f"\n{'─'*64}  LIVE\n"))

        while True:
            raw = f.readline()
            if raw:
                summary.ingest(raw)
                out = format_line(raw)
                if out is not None:
                    print(out, flush=True)
                if summary.done:
                    break
            else:
                time.sleep(0.5)


def show_run(run_dir: Path, stage: str | None) -> None:
    run_id = run_dir.name

    # print run_info if present
    info_file = run_dir / "run_info.txt"
    print(_header(f"Run: {run_id}"))
    if info_file.exists():
        for ln in info_file.read_text().splitlines():
            print(f"  {_c(DIM, ln)}")
    print(_footer())
    print()

    sup_log = run_dir / "supervised.log"
    rl_log  = run_dir / "rl.log"

    # if we're watching RL, first print a summary of supervised
    watching_rl = (stage == "rl") or (stage is None and rl_log.exists())
    if watching_rl and sup_log.exists():
        sup_summary = StageSummary("supervised")
        with open(sup_log) as f:
            for raw in f:
                sup_summary.ingest(raw)
        print(sup_summary.render())
        print()

    target = _active_log(run_dir, stage)
    if target is None:
        print(_c(YELLOW, f"Waiting for logs in {run_dir} …"), flush=True)
        while True:
            target = _active_log(run_dir, stage)
            if target:
                break
            time.sleep(1)

    log_stage = "rl" if "rl" in target.name else "supervised"
    active_summary = StageSummary(log_stage)

    print(_header(f"Stage {'2 — RL' if log_stage == 'rl' else '1 — Supervised'}  │  {target.name}"))
    print()

    try:
        tail_log(target, active_summary)
    except KeyboardInterrupt:
        pass

    print(_c(DIM, "\n[watch] stopped."))
    print()
    print(active_summary.render())


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Watch hangman_rl training progress.")
    ap.add_argument("--run",   default=None, help="Run ID, e.g. 20260514_1106")
    ap.add_argument("--stage", default=None, choices=["sup", "rl"],
                    help="Which stage log to tail (default: auto)")
    ap.add_argument("--list",  action="store_true", help="List all training runs and exit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    logs = Path("logs")

    if args.list:
        runs = _all_runs(logs)
        if not runs:
            print(_c(YELLOW, "No runs found in logs/"))
            return
        print(_c(BOLD, f"\n{'Run ID':<20}  {'Stages present':<28}  Info"))
        print("─" * 70)
        for r in runs:
            stages = "  ".join(
                s for s, f in [("supervised", r/"supervised.log"), ("rl", r/"rl.log")]
                if f.exists()
            )
            info = ""
            ri = r / "run_info.txt"
            if ri.exists():
                for ln in ri.read_text().splitlines():
                    if ln.startswith("started"):
                        info = ln.split(":", 1)[1].strip()
            marker = " ← latest" if (logs/"latest").is_symlink() and (logs/"latest").resolve() == r else ""
            print(f"  {r.name:<20}  {stages:<28}  {_c(DIM, info)}{_c(CYAN, marker)}")
        print()
        return

    if args.run:
        run_dir = logs / args.run
        if not run_dir.exists():
            print(_c(RED, f"Run not found: {run_dir}"))
            sys.exit(1)
    else:
        run_dir = _latest_run(logs)
        if run_dir is None:
            print(_c(RED, "No runs found. Start training with:"))
            print(_c(DIM, "  bash scripts/run_training.sh &"))
            sys.exit(1)

    show_run(run_dir, args.stage)


if __name__ == "__main__":
    main()
