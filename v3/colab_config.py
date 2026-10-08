"""Non-blocking initial selection; detailed controls live in the bottom panel."""

import asyncio
import json
import math
import time
import uuid

V3_DEFAULT_FOLDER = "/content/drive/MyDrive/PINN"
V3_SELECTION_TIMEOUT = 60


def v3_initial_setup(
    timeout_seconds=60, clock=time.monotonic, show=True, kernel=None, frontend=None
):
    """Finalize once; inactivity always chooses read-only with automatic storage.

    Colab uses browser -> kernel polling, not an idle Python loop timer.
    Other notebooks use a running asyncio loop or the existing kernel IOLoop.
    No new dormant loop or worker-thread model/Drive calls are created.
    Tests may inject a clock and call expire_if_due without real-time waiting.
    """
    import ipywidgets as w
    from IPython.display import display

    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("선택 대기 시간은 양수여야 합니다.")
    actions = {
        "results": "저장 결과 보기",
        "predict": "좌표 예측",
        "agent": "자연어 질문",
        "evaluate": "기능 검사",
        "train": "추가 학습",
    }
    style = {"description_width": "100px"}
    action = w.Dropdown(
        description="사용할 기능",
        options=[(label, key) for key, label in actions.items()],
        value="results",
        style=style,
        layout=w.Layout(width="100%"),
    )
    storage = w.Dropdown(
        description="파일 위치",
        options=[
            ("자동 · 연결된 Drive 또는 로컬", "auto"),
            ("Google Drive · 연결 요청", "drive"),
            ("로컬 · 업로드한 ZIP", "local"),
        ],
        value="auto",
        style=style,
        layout=w.Layout(width="100%"),
    )
    button = w.Button(description="선택 적용", button_style="primary")
    notice = w.HTML()
    dashboard = w.VBox(
        [
            w.HTML("<h3>최초 설정</h3>"),
            w.GridBox(
                [action, storage],
                layout=w.Layout(
                    grid_template_columns="repeat(auto-fit, minmax(280px, 1fr))",
                    grid_gap="8px",
                    width="100%",
                ),
            ),
            button,
            notice,
        ]
    )
    state = {"status": "pending", "choice": None, "deadline": clock() + timeout_seconds}
    listeners = []
    task = None
    schedule = None
    cancel_handle = None
    frontend_callback = None
    if frontend is None and show:
        try:
            from google.colab import output as frontend
        except ImportError:
            pass
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if kernel is None:
            from IPython import get_ipython

            kernel = getattr(get_ipython(), "kernel", None)
        loop = getattr(kernel, "io_loop", None)
        if loop is not None and callable(getattr(loop, "call_later", None)):
            schedule = loop.call_later
            cancel_handle = loop.remove_timeout
            state["timer_backend"] = "kernel_ioloop"
    else:
        schedule = loop.call_later
        cancel_handle = lambda handle: handle.cancel()
        state["timer_backend"] = "asyncio"
    state.setdefault("timer_backend", "manual")

    def stop_timer(remove_frontend=False):
        nonlocal task, frontend_callback
        if task is not None:
            cancel_handle(task)
            task = None
        if remove_frontend and frontend_callback is not None:
            name, frontend_callback = frontend_callback, None
            frontend.unregister_callback(name)

    def remaining():
        return max(0, math.ceil(state["deadline"] - clock()))

    def update_notice():
        if state["status"] == "pending":
            if state["timer_backend"] == "manual":
                notice.value = (
                    "<p>자동 타이머를 연결하지 못했습니다. ‘선택 적용’ 버튼으로 바로 진행하세요. "
                    "학습·검사는 자동 실행하지 않습니다.</p>"
                )
                return
            notice.value = (
                f"<p>입력이 없으면 {remaining()}초 후 저장 결과 조회로 진행합니다. "
                "선택 변경 시 대기 시간이 다시 시작됩니다. "
                "상세 설정과 실행 버튼은 맨 아래에 있습니다.</p>"
            )

    def settle(choice, status):
        if state["status"] != "pending":
            return False
        state.update(status=status, choice=dict(choice))
        action.disabled = storage.disabled = button.disabled = True
        suffix = " · 무입력 기본값" if status == "timed_out" else ""
        notice.value = (
            f"<p>{actions[choice['action']]}{suffix}. "
            "검사·학습·언어 모델은 자동 실행하지 않습니다. 상세 설정은 맨 아래 실행 패널에서 변경합니다.</p>"
        )
        stop_timer()
        callbacks = list(listeners)
        listeners.clear()
        for callback in callbacks:
            callback(dict(state["choice"]))
        return True

    def confirm(_=None):
        return settle({"action": action.value, "storage": storage.value}, "selected")

    def open_defaults(_=None):
        # Recovery ignores unconfirmed training; automatic storage may ask for Drive consent.
        return settle({"action": "results", "storage": "auto"}, "selected")

    def expire_if_due():
        if state["status"] == "pending" and clock() >= state["deadline"]:
            return settle({"action": "results", "storage": "auto"}, "timed_out")
        return False

    def touch(change):
        if state["status"] != "pending":
            return
        state["deadline"] = clock() + timeout_seconds
        update_notice()

    def on_ready(callback):
        if state["status"] == "cancelled":
            return
        if state["choice"] is not None:
            callback(dict(state["choice"]))
        else:
            listeners.append(callback)
            expire_if_due()

    def cancel():
        state["status"] = "cancelled"
        listeners.clear()
        stop_timer(remove_frontend=True)
        dashboard.close()

    def countdown():
        nonlocal task
        task = None
        if state["status"] != "pending":
            return
        expire_if_due()
        update_notice()
        if state["status"] == "pending":
            task = schedule(
                min(1.0, max(0.001, state["deadline"] - clock())), countdown
            )

    def frontend_tick():
        from IPython.display import JSON

        state["frontend_ticks"] = state.get("frontend_ticks", 0) + 1
        expire_if_due()
        update_notice()
        if state["status"] != "pending":
            stop_timer(remove_frontend=True)
        return JSON({"status": state["status"], "remaining": remaining()})

    action.observe(touch, names="value")
    storage.observe(touch, names="value")
    button.on_click(confirm)
    use_frontend = frontend is not None and all(
        callable(getattr(frontend, method, None))
        for method in ("register_callback", "unregister_callback", "eval_js")
    )
    if use_frontend:
        state["timer_backend"] = "colab_frontend"
    elif schedule is None:
        notice.value = (
            "<p>자동 타이머를 연결하지 못했습니다. ‘선택 적용’ 버튼으로 바로 진행하세요. "
            "학습·검사는 자동 실행하지 않습니다.</p>"
        )
    update_notice()
    if not use_frontend and schedule is not None:
        task = schedule(min(1.0, timeout_seconds), countdown)
    if show:
        try:
            from google.colab import output as colab_output

            colab_output.enable_custom_widget_manager()
        except ImportError:
            pass
        display(dashboard)
    if use_frontend:
        frontend_callback = "v3.selection." + uuid.uuid4().hex
        script = r"""(() => {
          const callback = CALLBACK_NAME;
          const info = document.createElement('p');
          info.textContent = '자동 대기 연결 중…';
          document.body.appendChild(info);
          async function tick() {
            try {
              const reply = await google.colab.kernel.invokeFunction(callback, [], {});
              const data = reply.data && reply.data['application/json'];
              if (!data) throw new Error('No timer status');
              if (data.status !== 'pending') {
                info.textContent = '최초 설정 완료 · 맨 아래 실행 패널을 확인하세요.';
                return;
              }
              info.textContent = '무입력 기본 조회까지 ' + data.remaining + '초';
              setTimeout(tick, 1000);
            } catch (error) {
              info.textContent = '자동 대기 연결이 종료됐습니다. 하단의 ‘기본 조회로 바로 진행’을 누르세요.';
            }
          }
          setTimeout(tick, 1000);
          return true;
        })()""".replace("CALLBACK_NAME", json.dumps(frontend_callback))
        try:
            frontend.register_callback(frontend_callback, frontend_tick)
            frontend.eval_js(script, ignore_result=True)
        except Exception:
            if frontend_callback is not None:
                try:
                    stop_timer(remove_frontend=True)
                except ValueError:
                    frontend_callback = None
            state["timer_backend"] = "manual"
            update_notice()
    return {
        "dashboard": dashboard,
        "action": action,
        "storage": storage,
        "button": button,
        "state": state,
        "confirm": confirm,
        "open_defaults": open_defaults,
        "on_ready": on_ready,
        "expire_if_due": expire_if_due,
        "cancel": cancel,
        "timer_task": task,
        "notice": notice,
    }
