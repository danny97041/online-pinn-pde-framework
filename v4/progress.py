"""Scoped console instrumentation; no additional model or gradient evaluation."""

from contextlib import contextmanager, ExitStack
from functools import wraps
import threading
import time
from pathlib import Path
import shutil
from unittest.mock import patch

_LOCK = threading.Lock()


@contextmanager
def training_progress(log=print, interval_seconds=15.0):
    """Instrument synchronous Colab training and restore hooks even on interruption.

    Hooks are process-wide while active, so concurrent instrumented runs are
    rejected. Numerical runner/runtime source and checkpoint identity stay intact.
    """
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("다른 학습의 진행 표시가 실행 중입니다.")
    try:
        from scipy import optimize
        from . import runner
        from .runtime import Runtime
        from .diagnostics import observe
        from . import diagnostics
        from .storage import write_json, sha_file

        original_lbfgs = Runtime.lbfgs
        original_evaluate = runner.evaluate
        original_pack = runner.pack
        original_resume = runner.resume_slot
        original_trial = runner.run_trial
        resume_depth = 0

        def emit(message):
            log(message)

        def label(runtime):
            direction = "역문제" if runtime.settings.inverse else "순방향"
            return f"{runtime.settings.equation} [{direction}] {runtime.arm}"

        @wraps(original_trial)
        def trial(root, settings, arm, seed, *args, **kwargs):
            direction = "역문제" if settings.inverse else "순방향"
            emit(f"실험 확인: {settings.equation} · {direction} · {arm} · 시드 {seed}")
            # Diagnostic provenance is separate from immutable training identity.
            source = Path(diagnostics.__file__)
            checksum = sha_file(source)
            destination = Path(root) / "validation" / "observer"
            destination.mkdir(parents=True, exist_ok=True)
            stem = f"{diagnostics.REVISION}_{checksum[:12]}"
            shutil.copyfile(source, destination / (stem + ".py"))
            write_json(
                destination / (stem + ".json"),
                {
                    "revision": diagnostics.REVISION,
                    "diagnostic_source_sha256": checksum,
                    "training_source_sha": runner.training_source_digest(),
                    "training_protocol_changed": False,
                    "historical_diagnostics_rewritten": False,
                },
            )
            return original_trial(root, settings, arm, seed, *args, **kwargs)

        @wraps(original_lbfgs)
        def lbfgs(runtime, tensors, completed):
            trial_label = label(runtime)
            started = time.perf_counter()
            emit(f"{trial_label} L-BFGS 시작 · 최대 {runtime.settings.lbfgs_maxiter}회")
            original_minimize = optimize.minimize

            @wraps(original_minimize)
            def minimize(fun, *args, **kwargs):
                if kwargs.get("method") != "L-BFGS-B":
                    return original_minimize(fun, *args, **kwargs)
                calls = 0
                accepted = 0
                last_value = None
                last_message = started
                callback = kwargs.get("callback")

                def report(force=False):
                    nonlocal last_message
                    now = time.perf_counter()
                    if force or now - last_message >= interval_seconds:
                        value = "—" if last_value is None else f"{last_value:.6g}"
                        emit(
                            f"{trial_label} L-BFGS 진행 · 반복 {accepted}/{runtime.settings.lbfgs_maxiter}"
                            f" · 손실·기울기 평가 {calls}회 · 최근 목적함수 {value}"
                            f" · 경과 {now - started:.0f}초"
                        )
                        last_message = now

                @wraps(fun)
                def objective(*position_args, **position_kwargs):
                    nonlocal calls, last_value
                    result = fun(*position_args, **position_kwargs)
                    calls += 1
                    last_value = float(
                        result[0] if isinstance(result, tuple) else result
                    )
                    report(force=calls == 1)
                    return result

                def accepted_step(xk):
                    nonlocal accepted
                    accepted += 1
                    report()
                    if callback is not None:
                        return callback(xk)

                kwargs["callback"] = accepted_step
                result = original_minimize(objective, *args, **kwargs)
                report(force=True)
                return result

            with patch.object(optimize, "minimize", minimize):
                result = original_lbfgs(runtime, tensors, completed)
            emit(
                f"{trial_label} L-BFGS 종료 · 반복 {result['iterations']}회"
                f" · 평가 {result['function_evaluations']}회"
                f" · {time.perf_counter() - started:.1f}초 · {result['message']}"
            )
            return result

        @wraps(original_evaluate)
        def evaluate(runtime, evaluation, physics, iteration, stage, *args, **kwargs):
            trial_label = label(runtime)
            if stage == "adam":
                emit(f"{trial_label} Adam 학습 종료 · {iteration}회 · 최종 평가 시작")
                if not runtime.settings.lbfgs_maxiter:
                    emit(f"{trial_label} L-BFGS 생략 · 설정 0회")
            else:
                emit(f"{trial_label} {stage} 최종 평가 시작")
            started = time.perf_counter()
            result = original_evaluate(
                runtime, evaluation, physics, iteration, stage, *args, **kwargs
            )
            emit(
                f"{trial_label} {stage} 최종 평가 완료 · {time.perf_counter() - started:.1f}초"
            )
            return result

        def save_with_notice(original, label, *args, **kwargs):
            started = time.perf_counter()
            emit(f"{label} 저장·무결성 검사 시작")
            result = original(*args, **kwargs)
            emit(f"{label} 저장 완료 · {time.perf_counter() - started:.1f}초")
            return result

        @wraps(original_pack)
        def pack(*args, **kwargs):
            if resume_depth:
                return original_pack(*args, **kwargs)
            return save_with_notice(original_pack, "결과 ZIP", *args, **kwargs)

        @wraps(original_resume)
        def resume(*args, **kwargs):
            nonlocal resume_depth
            resume_depth += 1
            try:
                return save_with_notice(original_resume, "재개 ZIP", *args, **kwargs)
            finally:
                resume_depth -= 1

        with ExitStack() as stack:
            stack.enter_context(patch.object(Runtime, "observe", observe))
            stack.enter_context(patch.object(runner, "run_trial", trial))
            stack.enter_context(patch.object(Runtime, "lbfgs", lbfgs))
            stack.enter_context(patch.object(runner, "evaluate", evaluate))
            stack.enter_context(patch.object(runner, "pack", pack))
            stack.enter_context(patch.object(runner, "resume_slot", resume))
            yield
    finally:
        _LOCK.release()
