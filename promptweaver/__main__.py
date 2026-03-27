"""CLI entry point for Prompt Weaver."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def main() -> None:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="promptweaver",
        description="Combinatorial prompt generator that never repeats",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--db",
        type=Path,
        metavar="PATH",
        help="Custom database path (default: ~/.local/share/promptweaver/prompts.db)",
    )

    # ── operation modes (mutually exclusive) ──────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="Export prompts to a JSON file and exit",
    )
    mode.add_argument(
        "--batch",
        type=int,
        metavar="N",
        help="Generate N unique prompts headlessly",
    )
    mode.add_argument(
        "--random",
        action="store_true",
        help="Generate a single prompt and print to stdout",
    )
    mode.add_argument(
        "--stats",
        action="store_true",
        help="Show generation statistics",
    )

    # ── visual mode ──────────────────────────────────────────────────
    parser.add_argument(
        "--hyper",
        action="store_true",
        help="Launch in Hyperobject Mode (3D ASCII viewport active)",
    )

    # ── modifiers ─────────────────────────────────────────────────────
    parser.add_argument(
        "--template",
        type=str,
        metavar="ID",
        help="Filter by template ID (for --batch/--random)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format for --batch (default: text)",
    )
    parser.add_argument(
        "--favorites-only",
        action="store_true",
        help="Export only favorited prompts (with --export)",
    )

    args = parser.parse_args()

    if args.batch is not None and args.batch < 1:
        parser.error("--batch requires a positive integer")

    if args.export:
        _cmd_export(args)
    elif args.batch:
        _cmd_batch(args)
    elif args.random:
        _cmd_random(args)
    elif args.stats:
        _cmd_stats(args)
    else:
        _cmd_tui(args)


# ── commands ──────────────────────────────────────────────────────────────


def _cmd_export(args: argparse.Namespace) -> None:
    from .store import PromptStore

    store = PromptStore(db_path=args.db)
    try:
        if args.favorites_only:
            prompts = store.get_favorites()
            label = "favorite "
        else:
            prompts = store.get_all()
            label = ""

        with open(args.export, "w") as f:
            json.dump(
                [p.model_dump(mode="json") for p in prompts],
                f,
                indent=2,
                default=str,
            )
        print(f"Exported {len(prompts)} {label}prompts to {args.export}")
    finally:
        store.close()


def _validate_template(
    engine: object, template_id: Optional[str]
) -> None:
    """Exit with an error if the template ID doesn't exist."""
    if template_id is None:
        return
    from .engine import CombinatorialEngine

    assert isinstance(engine, CombinatorialEngine)
    if template_id not in engine.templates:
        print(f"Error: unknown template '{template_id}'", file=sys.stderr)
        print(f"Available: {', '.join(engine.template_ids)}", file=sys.stderr)
        sys.exit(1)


def _cmd_batch(args: argparse.Namespace) -> None:
    from .engine import CombinatorialEngine
    from .store import PromptStore

    engine = CombinatorialEngine()
    store = PromptStore(db_path=args.db)
    _validate_template(engine, args.template)

    try:
        prompts = []
        for _ in range(args.batch):
            prompt = engine.generate_unique(
                store.seen_hashes,
                template_id=args.template,
            )
            store.save(prompt)
            prompts.append(prompt)

        if args.output_format == "json":
            json.dump(
                [p.model_dump(mode="json") for p in prompts],
                sys.stdout,
                indent=2,
                default=str,
            )
            print()
        else:
            for p in prompts:
                print(p.positive)

        print(
            f"\n// {len(prompts)} prompts generated ({store.count} total)",
            file=sys.stderr,
        )
    finally:
        store.close()


def _cmd_random(args: argparse.Namespace) -> None:
    from .engine import CombinatorialEngine
    from .store import PromptStore

    engine = CombinatorialEngine()
    store = PromptStore(db_path=args.db)
    _validate_template(engine, args.template)

    try:
        prompt = engine.generate_unique(
            store.seen_hashes,
            template_id=args.template,
        )
        store.save(prompt)
        print(prompt.positive)
    finally:
        store.close()


def _cmd_stats(args: argparse.Namespace) -> None:
    from .engine import CombinatorialEngine
    from .store import PromptStore

    engine = CombinatorialEngine()
    store = PromptStore(db_path=args.db)

    try:
        stats = store.get_stats()

        print("promptweaver stats")
        print("\u2500" * 40)
        print(f"  total prompts:  {stats['total']:>10,}")
        print(f"  favorites:      {stats['favorites']:>10,}")
        print(f"  templates used: {len(stats['by_template']):>10}")
        print()

        if stats["by_template"]:
            print("  by template:")
            for tid, count in stats["by_template"].items():
                print(f"    {tid:<24} {count:>6,}")
            print()

        if stats["first_generated"]:
            print(f"  first generated: {stats['first_generated']}")
            print(f"  last generated:  {stats['last_generated']}")
            print()

        total_combos = engine.total_combinations
        print(f"  combinatorial space: ~{total_combos:,}")
        if stats["total"] > 0:
            pct = stats["total"] / total_combos * 100
            print(f"  coverage: {pct:.8f}%")
    finally:
        store.close()


def _cmd_tui(args: argparse.Namespace) -> None:
    try:
        from .app import PromptWeaverApp

        app = PromptWeaverApp(db_path=args.db, hyper=args.hyper)
        app.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
