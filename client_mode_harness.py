"""Exercise Ray's real client-mode state machine without a C++ build.

`ray._raylet` is stubbed out and the client `Worker` is replaced with a
deterministic fake, so the *actual* Python logic that this change touches
(the `is_client_mode_enabled` flag and the `client_mode_hook` dispatch
decision) runs for real.

Usage:
    python client_mode_harness.py            # current working tree
    git stash && python client_mode_harness.py && git stash pop   # baseline

Run it from a Ray source checkout that is pip-installed (editable) so that
`import ray` resolves to the repo's python/ directory.
"""

import sys
import types

# --------------------------------------------------------------------------
# 1. Stub the compiled extension so pure-Python client logic is importable.
# --------------------------------------------------------------------------
class _AnyMeta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any


class _Any(metaclass=_AnyMeta):
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()

    def __bool__(self):
        return False


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any


sys.modules["ray._raylet"] = _StubModule("ray._raylet")

# --------------------------------------------------------------------------
# 2. Import the real Ray client modules.
# --------------------------------------------------------------------------
import ray  # noqa: E402
import ray.util.client as client_mod  # noqa: E402
import ray.util.client.worker as client_worker_mod  # noqa: E402
from ray._private import client_mode_hook as cmh  # noqa: E402
from ray.util.client import _ClientContext  # noqa: E402
from ray.util.client import ray as ray_client  # noqa: E402

PY_VER = ".".join(str(v) for v in sys.version_info[:3])


# --------------------------------------------------------------------------
# 3. Deterministic fake transport.
# --------------------------------------------------------------------------
class FakeWorker:
    """Stands in for ray.util.client.worker.Worker."""

    fail_mode = None  # None | "init" | "server_init" | "conn_info"

    def __init__(self, conn_str, **kwargs):
        self.conn_str = conn_str
        self.closed = False
        self.failed = FakeWorker.fail_mode == "init"
        if self.failed:
            raise ConnectionError("simulated connect failure")

    def _server_init(self, job_config=None, ray_init_kwargs=None):
        if FakeWorker.fail_mode == "server_init":
            raise ConnectionError("simulated server_init failure")

    def connection_info(self):
        if FakeWorker.fail_mode == "conn_info":
            raise ConnectionError("simulated connection_info failure")
        return {
            "num_clients": 1,
            "python_version": PY_VER,
            "ray_version": ray.__version__,
            "ray_commit": ray.__commit__,
        }

    def is_connected(self):
        return not self.closed and not self.failed

    def close(self):
        self.closed = True


client_worker_mod.Worker = FakeWorker


def state():
    return {
        "is_client_mode_enabled": cmh.is_client_mode_enabled,
        "should_convert": cmh.client_mode_should_convert(),
    }


def report(name, s):
    print(f"  {name}: {s}")


# --------------------------------------------------------------------------
# 4. Scenarios.
# --------------------------------------------------------------------------
def scenario_failed_connect():
    print("\n[S1] Failed connect via RayAPIStub (the bug being fixed)")
    FakeWorker.fail_mode = "init"
    report("before", state())
    err = "?"
    try:
        ray_client.connect("localhost:50051")
        print("  ERROR: expected failure")
    except Exception as e:
        err = type(e).__name__
    report("after ", state())
    print(f"  raised: {err}")
    FakeWorker.fail_mode = None


def scenario_success_then_failure():
    print("\n[S2] Successful connect, THEN a second failing connect")
    FakeWorker.fail_mode = None
    ctx1 = _ClientContext()
    try:
        ctx1.connect("localhost:50051")
    except Exception as e:
        print(f"  unexpected failure: {e}")
    report("after ctx1 success", state())

    FakeWorker.fail_mode = "init"
    ctx2 = _ClientContext()
    try:
        ctx2.connect("localhost:99999")
    except Exception:
        pass
    after = state()
    report("after ctx2 failure", after)
    print(f"  ctx1 still connected: {ctx1.is_connected()}")
    print(
        "  => client mode preserved for the still-live ctx1? "
        f"{after['is_client_mode_enabled']}"
    )
    FakeWorker.fail_mode = None
    ctx1.disconnect()


def scenario_failure_then_success():
    print("\n[S3] Failed connect, THEN a successful connect")
    FakeWorker.fail_mode = "init"
    try:
        ray_client.connect("localhost:50051")
    except Exception:
        pass
    report("after failure", state())

    FakeWorker.fail_mode = None
    ctx = _ClientContext()
    try:
        ctx.connect("localhost:50051")
    except Exception as e:
        print(f"  unexpected failure: {e}")
    report("after success", state())
    ctx.disconnect()


def scenario_inside_client_test():
    print("\n[S4] Failing connect with _inside_client_test=True (test harness path)")
    cmh._explicitly_enable_client_mode()
    report("before", state())
    ctx = _ClientContext()
    ctx._inside_client_test = True
    FakeWorker.fail_mode = "init"
    try:
        ctx.connect("localhost:50051")
    except Exception:
        pass
    report("after ", state())
    print(
        "  => harness-enabled client mode clobbered by a connect that never "
        "enabled it? "
        f"{not cmh.is_client_mode_enabled}"
    )
    FakeWorker.fail_mode = None
    cmh._explicitly_disable_client_mode()


def scenario_failure_inside_enable_cm():
    print("\n[S5] Failing connect inside enable_client_mode() context manager")
    from ray._private.client_mode_hook import enable_client_mode

    FakeWorker.fail_mode = "init"
    with enable_client_mode():
        ctx = _ClientContext()
        try:
            ctx.connect("localhost:50051")
        except Exception:
            pass
        report("inside cm", state())
    report("after cm ", state())
    FakeWorker.fail_mode = None


def scenario_allow_multiple():
    """Mirrors client_builder.py allow_multiple=True: set_context(None) then connect."""
    print("\n[S6] allow_multiple=True style: context A live, context B connect fails")
    FakeWorker.fail_mode = None

    ray_client.set_context(None)
    ctx_a = ray_client.get_context()
    try:
        ray_client.connect("localhost:50051")
    except Exception as e:
        print(f"  unexpected failure: {e}")
    report("after A connect", state())
    print(f"  contexts registered: {len(client_mod._all_contexts)}")

    old = ray_client.set_context(None)
    FakeWorker.fail_mode = "init"
    try:
        ray_client.connect("localhost:99999")
    except Exception:
        pass
    report("after B failure", state())
    print(f"  A still connected: {ctx_a.is_connected()}")
    print(
        "  => A's client-mode dispatch still works? "
        f"{cmh.client_mode_should_convert()}"
    )
    FakeWorker.fail_mode = None
    ray_client.set_context(old)
    ctx_a.disconnect()
    cmh._explicitly_disable_client_mode()


if __name__ == "__main__":
    print(f"python: {sys.version.split()[0]}")
    scenario_failed_connect()
    scenario_success_then_failure()
    scenario_failure_then_success()
    scenario_inside_client_test()
    scenario_failure_inside_enable_cm()
    scenario_allow_multiple()
