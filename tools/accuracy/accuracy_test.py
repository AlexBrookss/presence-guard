"""Accuracy test harness for the presence / badge detectors.

Two modes
---------
``record``  Capture labelled samples from the live camera (interactive).
``evaluate`` Re-run the detector over a labelled directory and print a
             confusion matrix, precision/recall and (optionally) a
             threshold sweep that suggests better cut-offs.

Dataset layout::

    dataset/
      badge/          badge is worn (lanyard + card)
      strap_only/     lanyard worn, card removed
      no_badge/       nothing worn
      no_person/      empty chair (camera sees the room, no person)
      dark/           lens covered / very dark (must be "unknown")

Only ``badge/`` and ``strap_only/`` are considered "positive" for the
``strap`` mode; in ``card`` mode only ``badge/`` is positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from presence_guard.config import Config, ConfigError, load_config  # noqa: E402
from presence_guard.types import BadgeDecision  # noqa: E402
from presence_guard.vision import VisionPipeline  # noqa: E402

CATEGORIES = ("badge", "strap_only", "no_badge", "no_person", "dark")
POSITIVE_FOR = {
    "strap": {"badge", "strap_only"},
    "card": {"badge"},
}


@dataclass
class Sample:
    path: Path
    label: str          # directory name
    decision: str = ""  # PRESENT / ABSENT / UNKNOWN
    valid: bool = False
    present_votes: int = 0
    valid_frames: int = 0
    largest: int = 0
    ratio: float = 0.0
    mean_v: float = 0.0
    face_width: float = 0.0
    reason: str = ""
    error: str = ""


@dataclass
class Result:
    samples: list[Sample] = field(default_factory=list)
    mode: str = "strap"

    # ------------------------------------------------------------------ stats
    def count(self, label: Optional[str] = None,
              decision: Optional[str] = None) -> int:
        return sum(1 for s in self.samples
                   if (label is None or s.label == label)
                   and (decision is None or s.decision == decision))

    def positive_labels(self) -> set[str]:
        return POSITIVE_FOR[self.mode]

    def confusion(self) -> dict[str, dict[str, int]]:
        """label -> decision -> count."""

        table: dict[str, dict[str, int]] = {}
        for sample in self.samples:
            table.setdefault(sample.label, {})
            table[sample.label][sample.decision] = (
                table[sample.label].get(sample.decision, 0) + 1)
        return table

    def metrics(self) -> dict[str, float]:
        tp = sum(1 for s in self.samples
                 if s.label in self.positive_labels()
                 and s.decision == BadgeDecision.PRESENT.value)
        fn = sum(1 for s in self.samples
                 if s.label in self.positive_labels()
                 and s.decision != BadgeDecision.PRESENT.value)
        fp = sum(1 for s in self.samples
                 if s.label not in self.positive_labels()
                 and s.label != "dark"      # dark is UNKNOWN by design
                 and s.decision == BadgeDecision.PRESENT.value)
        tn = sum(1 for s in self.samples
                 if s.label not in self.positive_labels()
                 and s.label != "dark"
                 and s.decision != BadgeDecision.PRESENT.value)

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and not np.isnan(precision) else float("nan"))
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall,
                "f1": f1, "accuracy": accuracy}

    def unknowns_on_dark(self) -> tuple[int, int]:
        total = self.count(label="dark")
        unknown = self.count(label="dark", decision=BadgeDecision.UNKNOWN.value)
        return unknown, total

    def presence_metrics(self) -> dict[str, float]:
        """person vs empty-chair, using presence.present."""

        tp = sum(1 for s in self.samples
                 if s.label != "no_person" and s.label != "dark" and s.valid)
        fn = sum(1 for s in self.samples
                 if s.label != "no_person" and s.label != "dark" and not s.valid)
        fp = self.count(label="no_person", decision=BadgeDecision.PRESENT.value)
        tn = self.count(label="no_person", decision=BadgeDecision.ABSENT.value)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall}


# ---------------------------------------------------------------- service
def _service_state() -> str:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "presence-guard"],
            capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _service(action: str) -> None:
    try:
        subprocess.run(["systemctl", "--user", action, "presence-guard"],
                       capture_output=True, text=True, timeout=20, check=False)
    except Exception as exc:
        print(f"  (无法 {action} presence-guard: {exc})")


# ---------------------------------------------------------------- recording
def record_camera(cfg: Config, output: Path, category: str, count: int,
                  interval: float, burst_frames: int) -> int:
    from presence_guard.camera import CameraCapture, CameraError

    if category not in CATEGORIES:
        raise SystemExit(f"unknown category {category!r}; choose from {CATEGORIES}")

    target = output / category
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    camera = CameraCapture(cfg.camera)

    print(f"\n=== 采集类别: {category}")
    print(f"    目录: {target}")
    print(f"    数量: {count} 张，间隔 {interval}s")
    if category == "badge":
        print("    请佩戴好工牌（挂绳 + 卡片都在脖子上）")
    elif category == "strap_only":
        print("    请只戴挂绳、把卡片取下/翻到背面")
    elif category == "no_badge":
        print("    请把工牌和挂绳都取下（正常坐着看屏幕）")
    elif category == "no_person":
        print("    请离开座位（画面里不要有人）")
    elif category == "dark":
        print("    请用手遮住摄像头镜头")
    print("    按 Ctrl-C 可提前结束\n")

    saved = 0
    try:
        while saved < count:
            try:
                frames = camera.capture_burst(burst_frames)
            except CameraError as exc:
                print(f"  摄像头错误: {exc}")
                time.sleep(2)
                continue
            for frame in frames:
                if saved >= count:
                    break
                saved += 1
                stamp = time.strftime("%Y%m%d-%H%M%S")
                path = target / f"{category}-{stamp}-{saved:03d}.jpg"
                cv2.imwrite(str(path), frame)
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
                print(f"  [{saved:3d}/{count}] {path.name}")
                if interval > 0:
                    time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  已中断")
    print(f"  完成，共 {saved} 张")
    return saved


# --------------------------------------------------------------- evaluation
def _analyse_file(vision: VisionPipeline, path: Path,
                  repeat_frames: int) -> Sample:
    sample = Sample(path=path, label=path.parent.name)
    frame = cv2.imread(str(path))
    if frame is None:
        sample.error = "cannot read image"
        sample.decision = BadgeDecision.UNKNOWN.value
        return sample

    presence = vision.analyze_presence([frame], person_fallback=True)
    sample.valid = presence.present

    if sample.label == "no_person":
        sample.decision = (BadgeDecision.PRESENT.value if presence.present
                           else BadgeDecision.ABSENT.value)
        sample.reason = "person detected" if presence.present else "empty chair"
        return sample

    if sample.label == "dark":
        # A covered lens must not yield a confident verdict.
        if presence.error:
            sample.decision = BadgeDecision.UNKNOWN.value
            sample.reason = presence.error
        else:
            sample.reason = "frame usable"
            sample.decision = (BadgeDecision.PRESENT.value if presence.present
                               else BadgeDecision.ABSENT.value)
        return sample

    if not presence.present:
        sample.decision = BadgeDecision.ABSENT.value
        sample.reason = "no person in frame"
        return sample

    badge = vision.analyze_badge([frame] * max(1, repeat_frames), presence)
    sample.decision = badge.decision.value
    sample.present_votes = badge.present_votes
    sample.valid_frames = badge.valid_frames
    sample.reason = badge.reason
    valid_stats = [s for s in badge.stats if s.valid]
    if valid_stats:
        sample.largest = max(s.largest_area for s in valid_stats)
        sample.ratio = max(s.blue_ratio for s in valid_stats)
        sample.mean_v = max(s.mean_v for s in valid_stats)
        sample.face_width = max(s.face_width for s in valid_stats)
    return sample


def evaluate(cfg: Config, dataset: Path, mode: str,
             repeat_frames: int) -> Result:
    cfg.badge.mode = mode
    vision = VisionPipeline(cfg)
    result = Result(mode=mode)

    for category in CATEGORIES:
        folder = dataset / category
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            continue
        print(f"  分析 {category}: {len(files)} 张 ...")
        for path in files:
            result.samples.append(_analyse_file(vision, path, repeat_frames))
    return result


def suggest_thresholds(result: Result, cfg: Config) -> list[str]:
    """Report the separation between positive and negative samples."""

    positives = [s.largest for s in result.samples
                 if s.label in result.positive_labels() and s.valid
                 and s.decision != BadgeDecision.UNKNOWN.value]
    negatives = [s.largest for s in result.samples
                 if s.label not in result.positive_labels()
                 and s.label not in ("dark", "no_person") and s.valid]
    if not positives or not negatives:
        return ["样本不足，无法给出阈值建议（正/负样本都需要至少 1 个有效样本）"]

    lines = [
        f"正样本 largest_area: min={min(positives)} "
        f"中位={int(np.median(positives))} max={max(positives)}",
        f"负样本 largest_area: min={min(negatives)} "
        f"中位={int(np.median(negatives))} max={max(negatives)}",
    ]
    gap_low, gap_high = max(negatives), min(positives)
    if gap_low < gap_high:
        suggested = int((gap_low + gap_high) / 2)
        lines.append(
            f"当前 min_blue_pixels={cfg.badge.min_blue_pixels}；"
            f"建议设在 {max(gap_low + 1, 1)}..{gap_high - 1} 之间（中值 {suggested}）")
    else:
        lines.append(
            "正负样本在 largest_area 上重叠，单靠面积阈值分不开；"
            "请检查光照/距离，或使用 mode=\"card\" / 调整 HSV 范围")
    return lines


def print_report(result: Result, cfg: Config) -> None:
    print("\n" + "=" * 72)
    print(f"工牌判定精准度报告  (mode = {result.mode}, "
          f"min_blue_pixels = {cfg.badge.min_blue_pixels})")
    print("=" * 72)

    table = result.confusion()
    decisions = ("present", "absent", "unknown")
    header = f"{'样本类别':<12}" + "".join(f"{d:>10}" for d in decisions) + f"{'合计':>8}"
    print(header)
    print("-" * len(header))
    for label in CATEGORIES:
        if label not in table:
            continue
        row = table[label]
        total = sum(row.values())
        print(f"{label:<12}" + "".join(f"{row.get(d, 0):>10}" for d in decisions)
              + f"{total:>8}")

    metrics = result.metrics()
    print("\n-- 工牌检测（present 视为阳性）--")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  精准率 precision = {metrics['precision']:.3f}")
    print(f"  召回率 recall    = {metrics['recall']:.3f}")
    print(f"  F1               = {metrics['f1']:.3f}")
    print(f"  准确率 accuracy  = {metrics['accuracy']:.3f}")

    presence = result.presence_metrics()
    print("\n-- 在座检测（person vs 空椅子）--")
    print(f"  TP={presence['tp']}  FP={presence['fp']}  "
          f"FN={presence['fn']}  TN={presence['tn']}")
    print(f"  精准率 precision = {presence['precision']:.3f}")
    print(f"  召回率 recall    = {presence['recall']:.3f}")

    unknown, total = result.unknowns_on_dark()
    if total:
        print(f"\n-- 遮镜头/过暗 --\n  {unknown}/{total} 正确判为 unknown"
              f"（不会锁屏、不会提醒）")

    print("\n-- 阈值建议 --")
    for line in suggest_thresholds(result, cfg):
        print("  " + line)

    failures = [s for s in result.samples if _is_failure(s, result.mode)]
    if failures:
        print(f"\n-- 判定错误 ({len(failures)}) --")
        for s in failures[:20]:
            print(f"  {s.label:<12} {s.decision:<8} largest={s.largest:<5} "
                  f"ratio={s.ratio:.4f} V={s.mean_v:.0f} fw={s.face_width:.0f} "
                  f"| {s.path.name}")
            if s.reason:
                print(f"      {s.reason}")
    else:
        print("\n-- 没有判定错误 --")


def _is_failure(sample: Sample, mode: str) -> bool:
    positive = sample.label in POSITIVE_FOR[mode]
    if sample.label == "dark":
        return sample.decision != BadgeDecision.UNKNOWN.value
    if sample.label == "no_person":
        return sample.decision == BadgeDecision.PRESENT.value
    expected = (BadgeDecision.PRESENT.value if positive
                else BadgeDecision.ABSENT.value)
    return sample.decision != expected


def write_outputs(result: Result, cfg: Config, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    report_json = [{
        "file": str(s.path),
        "label": s.label,
        "decision": s.decision,
        "largest": s.largest,
        "ratio": round(s.ratio, 4),
        "mean_v": round(s.mean_v, 1),
        "face_width": round(s.face_width, 1),
        "votes": f"{s.present_votes}/{s.valid_frames}",
        "reason": s.reason,
        "error": s.error,
    } for s in result.samples]
    (output / "report.json").write_text(
        json.dumps({
            "mode": result.mode,
            "metrics": result.metrics(),
            "presence_metrics": result.presence_metrics(),
            "confusion": result.confusion(),
            "samples": report_json,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    with (output / "report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "label", "decision", "largest", "ratio",
                         "mean_v", "face_width", "votes", "reason"])
        for s in result.samples:
            writer.writerow([s.path.name, s.label, s.decision, s.largest,
                             round(s.ratio, 4), round(s.mean_v, 1),
                             round(s.face_width, 1),
                             f"{s.present_votes}/{s.valid_frames}", s.reason])
    print(f"\n报告已写入: {output / 'report.json'} / report.csv")


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accuracy-test",
        description="采集各状态样本并统计 presence-guard 的识别精准度",
    )
    p.add_argument("--config", help="配置文件路径")
    p.add_argument("--camera", help="摄像头设备")
    p.add_argument("--dataset", default="~/.cache/presence-guard/accuracy",
                   help="样本集目录（默认 ~/.cache/presence-guard/accuracy）")

    sub = p.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="从摄像头采集带标签的样本")
    rec.add_argument("--category", required=True,
                     choices=tuple(CATEGORIES) + ("all",))
    rec.add_argument("--keep-service", action="store_true",
                     help="不要自动停止/恢复 presence-guard 服务")
    rec.add_argument("--count", type=int, default=20, help="采集张数（默认 20）")
    rec.add_argument("--interval", type=float, default=0.6,
                     help="每张之间的间隔秒数（默认 0.6）")
    rec.add_argument("--burst-frames", type=int, default=1,
                     help="每次打开摄像头抓几帧（默认 1）")

    ev = sub.add_parser("evaluate", help="对已有样本集做精准度评估")
    ev.add_argument("--mode", choices=["strap", "card"],
                    help="工牌判定模式（默认取配置）")
    ev.add_argument("--repeat-frames", type=int, default=5,
                    help="单张图片重复几帧以满足多帧投票（默认 5）")
    ev.add_argument("--output", default="",
                    help="报告输出目录（默认 <dataset>/report）")

    sub.add_parser("list", help="列出样本集统计")
    clean = sub.add_parser("clean", help="清理样本/报告")
    clean.add_argument("--all", action="store_true", help="同时删除所有样本图片")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    if args.camera:
        cfg.camera.device = args.camera

    dataset = Path(args.dataset).expanduser()

    if args.command == "record":
        was_active = _service_state() == "active"
        if was_active and not args.keep_service:
            print("==> 暂停 presence-guard 服务，释放摄像头")
            _service("stop")
            time.sleep(1.5)
        try:
            categories = (CATEGORIES if args.category == "all"
                          else (args.category,))
            total = 0
            for category in categories:
                if args.category == "all":
                    input(f"\n准备采集 [{category}]，摆好姿势后按回车开始…")
                total += record_camera(cfg, dataset, category, args.count,
                                       args.interval, args.burst_frames)
        finally:
            if was_active and not args.keep_service:
                print("==> 恢复 presence-guard 服务")
                _service("start")
        return 0 if total else 1

    if args.command == "clean":
        removed = 0
        if args.all:
            for category in CATEGORIES:
                folder = dataset / category
                if folder.is_dir():
                    removed += sum(1 for _ in folder.iterdir())
                    shutil.rmtree(folder)
        report = dataset / "report"
        if report.is_dir():
            removed += sum(1 for _ in report.rglob("*") if _.is_file())
            shutil.rmtree(report)
        print(f"已清理 {removed} 个文件" + ("（含所有样本）" if args.all else "（保留样本集）"))
        return 0

    if args.command == "list":
        for category in CATEGORIES:
            folder = dataset / category
            count = (len([p for p in folder.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
                     if folder.is_dir() else 0)
            print(f"{category:<12} {count:>4} 张")
        return 0

    mode = args.mode or cfg.badge.mode
    if not dataset.is_dir():
        print(f"样本集不存在: {dataset}", file=sys.stderr)
        print("请先运行: accuracy-test record --category badge", file=sys.stderr)
        return 2

    print(f"样本集: {dataset}")
    print(f"模式: {mode}")
    result = evaluate(cfg, dataset, mode, args.repeat_frames)
    if not result.samples:
        print("样本集为空", file=sys.stderr)
        return 2

    print_report(result, cfg)
    output = Path(args.output).expanduser() if args.output else dataset / "report"
    write_outputs(result, cfg, output)

    metrics = result.metrics()
    # Non-zero exit when precision/recall drop below a usable bar.
    ok = ((np.isnan(metrics["precision"]) or metrics["precision"] >= 0.9)
          and (np.isnan(metrics["recall"]) or metrics["recall"] >= 0.9))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
