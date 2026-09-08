"""Trusted one-off entrypoint for synchronizing a pinned RAG release bundle."""

# ruff: file-ignore[T201] - CLI output is the metadata-only sync report

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from app.corpus_sync import (
    BundleValidationError,
    ConcurrentInventoryError,
    CorpusBundle,
    DeletionCapExceededError,
    EmbeddingNotReadyError,
    RetrievalSmokeError,
    SyncPlan,
    apply_plan,
    build_plan,
    load_bundle_path,
    synchronize,
)
from app.data.connection import Database
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

__all__ = ["BundleValidationError", "main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="report only (the default)"
    )
    mode.add_argument("--apply", action="store_true", help="mutate the document index")
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="DESTRUCTIVE: delete every document and cascade-delete every chunk",
    )
    parser.add_argument(
        "--max-deletions",
        type=int,
        help="required approval cap for ordinary apply mode",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-bundle-sha256")
    parser.add_argument("--expected-deployment-identity")
    return parser


def _print_plan(bundle: object, plan: object) -> None:
    if not isinstance(bundle, CorpusBundle) or not isinstance(plan, SyncPlan):
        raise TypeError("unexpected sync object")
    print(f"source_commit={bundle.source_commit}")
    print(f"raw_bundle_sha256={bundle.raw_bundle_sha256}")
    print(f"manifest_sha256={bundle.manifest_sha256}")
    print(f"add={len(plan.add)} names={','.join(entry.name for entry in plan.add)}")
    print(
        f"update={len(plan.update)} names={','.join(entry.name for entry in plan.update)}"
    )
    print(f"unchanged={len(plan.unchanged)} names={','.join(plan.unchanged)}")
    print(f"delete={len(plan.delete)} names={','.join(plan.delete)}")
    print(
        f"legacy_delete={len(plan.legacy_delete)} names={','.join(plan.legacy_delete)}"
    )
    print(f"preserved={len(plan.preserved)} names={','.join(plan.preserved)}")


async def _run(arguments: argparse.Namespace) -> int:
    if arguments.replace_all and not arguments.apply:
        raise ValueError("--replace-all requires --apply")
    if (
        arguments.apply
        and arguments.max_deletions is None
        and not arguments.replace_all
    ):
        raise ValueError("--apply requires an explicit --max-deletions")
    if arguments.apply and (
        arguments.expected_source_commit is None
        or arguments.expected_bundle_sha256 is None
    ):
        raise ValueError(
            "--apply requires --expected-source-commit and --expected-bundle-sha256"
        )
    configured_source_commit = os.environ.get("RAG_SYNC_EXPECTED_SOURCE_COMMIT", "")
    configured_bundle_sha256 = os.environ.get("RAG_SYNC_EXPECTED_BUNDLE_SHA256", "")
    if arguments.apply and (
        not configured_source_commit
        or arguments.expected_source_commit != configured_source_commit
        or not configured_bundle_sha256
        or arguments.expected_bundle_sha256 != configured_bundle_sha256
    ):
        raise ValueError("approved source commit and bundle SHA-256 assertion failed")
    bundle = load_bundle_path(
        arguments.bundle,
        expected_bundle_sha256=arguments.expected_bundle_sha256,
    )
    if (
        arguments.expected_source_commit is not None
        and arguments.expected_source_commit != bundle.source_commit
    ):
        raise BundleValidationError(
            "source commit does not match the approved expectation"
        )
    if arguments.apply and not arguments.expected_deployment_identity:
        raise ValueError("--apply requires --expected-deployment-identity")
    configured_identity = os.environ.get("RAG_SYNC_DEPLOYMENT_IDENTITY", "")
    if arguments.apply and (
        not configured_identity
        or arguments.expected_deployment_identity != configured_identity
    ):
        raise ValueError("deployment identity assertion failed")
    if arguments.timeout_seconds < 0:
        raise ValueError("--timeout-seconds must not be negative")

    settings = Settings()
    database = Database(
        settings.DATABASE_URL,
        min_size=settings.DATABASE_POOL_MIN_SIZE,
        max_size=settings.DATABASE_POOL_MAX_SIZE,
    )
    http_initialized = False
    try:
        if arguments.apply:
            init_http_client()
            http_initialized = True
        await database.init()
        if arguments.apply:
            result = await synchronize(
                database,
                bundle,
                apply=True,
                max_deletions=arguments.max_deletions or 0,
                timeout_seconds=arguments.timeout_seconds,
                expected_source_commit=arguments.expected_source_commit,
                expected_bundle_sha256=arguments.expected_bundle_sha256,
                plan_report=lambda plan: _print_plan(bundle, plan),
                replace_all=arguments.replace_all,
            )
        else:
            plan = await build_plan(database, bundle)
            _print_plan(bundle, plan)
            result = await apply_plan(database, bundle, plan, apply=False)
        print(f"mode={'apply' if not result.dry_run else 'dry-run'}")
        print(f"deleted={len(result.deleted)}")
    finally:
        await database.disconnect()
        if http_initialized:
            await close_http_client()
    return 0


def main() -> int:
    arguments = _parser().parse_args()
    try:
        return asyncio.run(_run(arguments))
    except (
        BundleValidationError,
        ConcurrentInventoryError,
        DeletionCapExceededError,
        EmbeddingNotReadyError,
        RetrievalSmokeError,
        ValueError,
    ) as exc:
        return _parser().error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
