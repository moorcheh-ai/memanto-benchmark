from __future__ import annotations

import asyncio
import copy
from decimal import Decimal
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

# The unmodified upstream provider source is embedded below when this file is built.
# Its source URL and SHA256 are included in identity(); it is loaded only at execution.
MEMANTO_PROVIDER_URL = "https://github.com/moorcheh-ai/memanto/blob/main/integrations/hermes-agents/hermes_memanto/provider.py"


def digest(value):
    """Hash JSON evidence deterministically without accepting non-finite numbers."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Publish complete JSON atomically; leave prior evidence intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot record {type(value).__name__}")


def agent_cost(events, rates):
    """Price disjoint Hermes usage buckets using explicit USD-per-million overrides."""
    if rates is None:
        return None
    fields = {"input": "input_tokens", "cached_input": "cache_read_tokens",
              "cache_write": "cache_write_tokens", "output": "output_tokens"}
    for field in fields:
        number = rates.get(field)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
            raise ValueError(f"Invalid USD-per-million rate: {field}")
    total = 0.0
    for event in events:
        usage = event["assistant"].get("usage")
        if not isinstance(usage, dict):
            raise ValueError("Missing per-response API usage")
        if any(type(usage.get(field)) is not int or usage[field] < 0 for field in fields.values()):
            raise ValueError("Missing or invalid normalized usage buckets")
        if usage.get("prompt_tokens") != sum(usage[field] for field in ("input_tokens", "cache_read_tokens", "cache_write_tokens")):
            raise ValueError("OpenAI prompt total does not reconcile")
        total += sum(usage[field] * rates[kind] for kind, field in fields.items()) / 1_000_000
    return total


def provider_source(options=None):
    """Read the installed plugin when configured; never silently fall back."""
    path = (options or {}).get("memanto_plugin_path")
    if path:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Installed Memanto plugin not found: {path}")
        return path.read_text(encoding="utf-8"), str(path)
    return MEMANTO_PROVIDER_SOURCE, MEMANTO_PROVIDER_URL


def provider_module(options=None):
    """Load the installed plugin as a package, supporting relative imports."""
    import importlib.util
    import types

    name = "dolphinbench_native_memanto"
    plugin_path = (options or {}).get("memanto_plugin_path")

    if plugin_path:
        path = Path(plugin_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Memanto plugin not found: {path}")
        spec = importlib.util.spec_from_file_location(
            name, path, submodule_search_locations=[str(path.parent)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Memanto plugin: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
            raise
        return module

    module = types.ModuleType(name)
    module.__file__ = "<embedded upstream Memanto Hermes provider>"
    sys.modules[name] = module
    exec(compile(MEMANTO_PROVIDER_SOURCE, module.__file__, "exec"),
         module.__dict__)
    return module


class ProviderSession:
    """Run upstream Memanto hooks, record writes, and deny mutation during tests."""

    def __init__(self, harness, request, directory):
        self.harness, self.request, self.directory = harness, request, directory
        self.testing = request.phase == "tests"
        self.errors, self.receipts = [], []
        self.lock = threading.RLock()
        self.receipt_lock = threading.Lock()
        self.provider = None

    def start(self):
        native = provider_module(self.harness.options)
        directory = self.directory / "provider"
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
        configuration = {
            "agent_id": self.harness.agent_id(self.request.persona),
            "pattern": "tool", "auto_recall": False,
            "auto_capture": False, "auto_create": not self.testing,
            "mirror_memory_writes": False,
            "max_recall_results": self.harness.options["top_k"],
        }
        save(directory / "memanto.json", configuration)
        # Upstream gives this env var precedence over its configuration file.
        # Do not permit a shell-wide smoke-test namespace to override the run.
        previous = os.environ.pop("MEMANTO_AGENT_ID", None)
        try:
            self.provider = native.MemantoMemoryProvider()
            self.provider.initialize(self.directory.name, hermes_home=str(directory),
                                     agent_identity=self.request.persona, agent_context="primary")
        finally:
            if previous is not None:
                os.environ["MEMANTO_AGENT_ID"] = previous
        self._drain()
        provider = self.provider
        if not provider._active or provider._client is None:
            raise RuntimeError("Native Memanto provider did not activate")
        provider._client.ensure_session()
        if provider._client.agent_id != self.harness.agent_id(self.request.persona):
            raise RuntimeError("Memanto namespace binding mismatch")
        threshold = self.harness.options.get("recall_min_similarity")
        if threshold is not None:
            limit = self.harness.adaptive_recall_limit
            sdk = provider._client._client
            original_recall = sdk.recall
            if "min_similarity" not in inspect.signature(original_recall).parameters:
                raise RuntimeError("Installed Memanto SDK does not expose recall min_similarity")

            def adaptive_recall(**arguments):
                arguments.pop("type", None)
                if limit is None:
                    arguments.pop("limit", None)
                else:
                    arguments["limit"] = limit
                arguments["min_similarity"] = threshold
                return original_recall(**arguments)

            sdk.recall = adaptive_recall

        original = provider._client.remember

        def remember(**arguments):
            if self.testing:
                raise PermissionError("Memory writes are blocked during evaluation")
            receipt = {"arguments": arguments, "status": "started"}
            with self.receipt_lock:
                self.receipts.append(receipt)
                self._save_receipts()
            try:
                result = original(**arguments)
                receipt["response"] = result
                if result.get("status") not in {"queued", "success", "ok"} or not result.get("memory_id"):
                    raise RuntimeError("Memanto returned no acknowledged memory ID")
                self._wait_indexed(result["memory_id"], arguments["content"])
                receipt["status"] = "indexed"
                return result
            except BaseException as exc:
                receipt.update(status="uncertain", error=f"{type(exc).__name__}: {exc}")
                self.errors.append(receipt["error"])
                raise
            finally:
                with self.receipt_lock:
                    self._save_receipts()

        provider._client.remember = remember
        if self.testing:
            provider._write_enabled = False
            provider._auto_capture = False
            provider._mirror_memory_writes = False
        return self

    def _save_receipts(self):
        save(self.directory / "memory-writes.json", self.receipts)

    def _drain(self):
        if self.provider is None:
            return
        deadline = time.monotonic() + self.harness.options["write_timeout"]
        for attribute in ("_warmup_thread", "_sync_thread", "_write_thread"):
            thread = getattr(self.provider, attribute, None)
            if thread is not None:
                thread.join(max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    raise TimeoutError(f"Memanto background operation did not finish: {attribute}")
        if self.errors:
            raise RuntimeError("Memanto write failed; evidence retained: " + self.errors[0])

    def _wait_indexed(self, memory_id, content):
        """
        Require the written ID to be observable through semantic search, not just queued.
        """
        sdk = self.provider._client._client
        namespace = sdk.get_agent(self.harness.agent_id(self.request.persona))["namespace"]
        backend = sdk._get_moorcheh()
        deadline = time.monotonic() + self.harness.options["write_timeout"]
        while True:
            result = backend.similarity_search.query(namespaces=[namespace], query=content, top_k=100)
            if any(str(item.get("id")) == str(memory_id) for item in result.get("results", [])):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Written memory not observed in semantic search: {memory_id}")
            time.sleep(1)

    def schemas(self):
        schemas = self.provider.get_tool_schemas()
        # Preserve a single declared retrieval mode for comparable experiments.
        interface = self.harness.options.get("memory_interface", "answer")
        operation = "memanto_" + interface
        allowed = {operation} if self.testing else {"memanto_remember"}
        selected = copy.deepcopy([schema for schema in schemas if schema["name"] in allowed])
        if interface == "recall":
            for schema in selected:
                if schema["name"] == operation:
                    parameters = schema["parameters"]
                    properties = parameters.setdefault("properties", {})
                    if self.harness.options.get("recall_min_similarity") is not None:
                        properties.pop("limit", None)
                        if "required" in parameters:
                            parameters["required"] = [
                                name for name in parameters["required"] if name != "limit"
                            ]
                    else:
                        properties["limit"] = {
                            "type": "integer", "enum": [self.harness.options["top_k"]],
                            "description": "Fixed retrieval budget for this experiment."}
        return selected

    def call(self, operation, arguments):
        """Permit only the upstream provider's declared tools and normal turn hooks."""
        with self.lock:
            self._drain()  # Never let an upstream thread reference be overwritten while live.
            if operation == "prefetch":
                if not self.testing:
                    return ""
                return self.provider.prefetch(arguments["query"], session_id=arguments.get("session_id", ""))
            if operation == "sync_turn":
                return None
            if operation not in {schema["name"] for schema in self.schemas()}:
                raise PermissionError(f"Memory operation is not permitted: {operation}")
            if (operation == "memanto_recall"
                    and self.harness.options.get("memory_interface") == "recall"
                    and self.harness.options.get("recall_min_similarity") is None):
                arguments = dict(arguments)
                limit = arguments.get("limit", self.harness.options["top_k"])
                if type(limit) is not int or limit != self.harness.options["top_k"]:
                    return json.dumps({"error": "Recall limit must equal the configured top_k",
                                       "required_limit": self.harness.options["top_k"]})
                arguments["limit"] = limit
            result = self.provider.handle_tool_call(operation, arguments)
            self._drain()
            return result

    def close(self):
        with self.lock:
            self._drain()
            if self.provider is not None:
                self.provider.shutdown()



def _original_user_boundary(messages, original):
    # Preserve the runtime instruction in the transcript; never rewrite its role.
    users = [(i, m.get("content")) for i, m in enumerate(messages) if m.get("role") == "user"]
    if [content for _, content in users] == [original]:
        return True
    marker = "You've reached the maximum number of tool-calling iterations allowed. Please provide a final response summarizing what you've found and accomplished so far, without calling any more tools."
    return (len(users) == 2 and users[0][1] == original
            and users[1][1] == marker and users[1][0] == len(messages) - 2
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls"))


def worker():
    """Run the unchanged native Hermes conversation loop in its installed interpreter."""
    protocol = sys.stdout
    sys.stdout = sys.stderr
    # Import BEFORE reading stdin: Hermes can re-exec to activate its installed runtime.
    from run_agent import AIAgent
    from tools.registry import registry
    from toolsets import create_custom_toolset
    from agent.memory_manager import MemoryManager
    from agent.memory_provider import MemoryProvider
    spec = json.loads(sys.stdin.readline())
    sys.path.insert(0, spec["benchmark_source"])
    from harness.submission_capture import install, read_recording
    capture_recorder = install(Path(spec["recording_directory"]))

    # Observe the existing summary request without altering its input, output or retries.
    import agent.chat_completion_helpers as _summary_helpers
    from harness.submission_capture import chat_messages as _capture_chat, build_rollout as _capture_build
    from harness.submission import SETTINGS_FIELDS as _capture_fields
    from agent.usage_pricing import normalize_usage as _capture_usage
    _original_summary_builder = _summary_helpers._SUMMARY_ATTEMPT_BUILDERS["codex_responses"]
    def _observed_summary_builder(agent, api_messages, api_request_id):
        attempt = _original_summary_builder(agent, api_messages, api_request_id)
        def observed_attempt(retry_count):
            original_call = agent._interruptible_api_call
            def observed_call(kwargs):
                started = time.monotonic()
                response = original_call(kwargs)
                try:
                    # This is the provider's real response usage, not aggregate subtraction.
                    raw_usage = getattr(response, "usage", None)
                    if raw_usage is None:
                        raise ValueError("Iteration-summary provider response has no usage")
                    usage = _capture_usage(raw_usage, provider=agent.provider, api_mode=agent.api_mode)
                    normalized = agent._get_transport().normalize_response(response)
                    content = (normalized.content or "").strip()
                    import re as _capture_re
                    visible = _capture_re.sub(r'<think>.*?</think>\s*', '', content, flags=_capture_re.DOTALL).strip()
                    settings = {k: json_value(v) for k, v in kwargs.items()
                                if k in _capture_fields and v is not None}
                    settings["model"] = agent.model
                    if isinstance(kwargs.get("reasoning"), dict) and "effort" in kwargs["reasoning"]:
                        settings["reasoning_effort"] = kwargs["reasoning"]["effort"]
                    buckets = {name: getattr(usage, name) for name in
                               ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")}
                    if any(type(v) is not int or v < 0 for v in buckets.values()):
                        raise ValueError("Incomplete iteration-summary response usage")
                    event = {"settings": settings, "messages": _capture_chat(api_messages),
                             "assistant": {"role": "assistant", "content": visible,
                                           "usage": buckets, "duration_ms": (time.monotonic() - started) * 1000},
                             "capture_role": "iteration_summary", "retry_count": retry_count,
                             "omit_from_rollout": not bool(visible)}
                    capture_recorder.write(spec["session_id"], event)
                except Exception as exc:
                    # Do not change Hermes behavior, but refuse a misleading export later.
                    capture_recorder.write(spec["session_id"], {"error": "Iteration summary capture: " + str(exc)})
                return response
            agent._interruptible_api_call = observed_call
            try:
                return attempt(retry_count)
            finally:
                agent._interruptible_api_call = original_call
        return observed_attempt
    _summary_helpers._SUMMARY_ATTEMPT_BUILDERS["codex_responses"] = _observed_summary_builder
    _original_read_recording = read_recording
    def read_recording(path):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # Empty summary retries are still retained and priced in the raw recording.
        # Hermes does not append an empty response to its native conversation.
        for event in events:
            if event.get("error"):
                raise RuntimeError(event["error"])
        return _capture_build([event for event in events if not
                              (event.get("capture_role") == "iteration_summary" and event.get("omit_from_rollout"))])

    pending, pending_lock, output_lock = {}, threading.Lock(), threading.Lock()
    hook_errors = []

    def emit(value):
        with output_lock:
            protocol.write(json.dumps(json_value(value), allow_nan=False) + "\n")
            protocol.flush()

    def receive():
        try:
            for line in sys.stdin:
                answer = json.loads(line)
                with pending_lock:
                    mailbox = pending.get(answer.get("id"))
                if mailbox is None:
                    raise RuntimeError("Unmatched controller response")
                mailbox.put(answer)
        finally:
            with pending_lock:
                for mailbox in pending.values():
                    mailbox.put({"error": "Controller pipe closed"})

    def rpc(operation, arguments, kind="memory"):
        identifier, mailbox = uuid4().hex, queue.Queue()
        with pending_lock:
            pending[identifier] = mailbox
        try:
            emit({"event": "request", "id": identifier, "kind": kind,
                  "operation": operation, "arguments": arguments})
            answer = mailbox.get(timeout=spec["timeout"] + spec["write_timeout"])
            if answer.get("error"):
                raise RuntimeError(answer["error"])
            return answer.get("value")
        finally:
            with pending_lock:
                pending.pop(identifier, None)

    class BoundMemory(MemoryProvider):
        @property
        def name(self):
            return "memanto"

        def is_available(self):
            return True

        def initialize(self, session_id, **kwargs):
            pass  # Controller initializes the actual upstream provider.

        def system_prompt_block(self):
            return spec["memory_prompt"]

        def get_tool_schemas(self):
            return spec["memory_schemas"]

        def handle_tool_call(self, tool_name, args, **kwargs):
            return rpc(tool_name, args)

        def prefetch(self, query, *, session_id=""):
            return rpc("prefetch", {"query": query, "session_id": session_id})

        def sync_turn(self, user_content, assistant_content, *, session_id="", **kwargs):
            try:
                return rpc("sync_turn", {"user_content": user_content,
                                        "assistant_content": assistant_content, "session_id": session_id})
            except Exception as exc:
                hook_errors.append(str(exc))
                raise

    try:
        # Process-local configuration only. Never load personal memory, skills, MCPs,
        # filesystem tools, or real app connectors from the user's normal Hermes profile.
        import hermes_cli.config as config
        isolated = {"memory": {"memory_enabled": False, "user_profile_enabled": False,
                               "enabled": False, "provider": "builtin"},
                    "agent": {"api_max_retries": spec["api_attempts"], "auto_recovery_cycles": 0,
                              "retry_rate_limits_until_success": False},
                    "compression": {"enabled": False}, "plugins": {}, "mcp_servers": {}}
        config.load_config = lambda *args, **kwargs: copy.deepcopy(isolated)
        config.load_config_readonly = lambda *args, **kwargs: copy.deepcopy(isolated)
        # The benchmark's provider is injected below; unrelated host plugins must
        # not inject private context or install tools into this fresh conversation.
        import hermes_cli.plugins as host_plugins
        import hermes_cli.left_core_migration as host_migration
        host_plugins.discover_plugins = lambda *args, **kwargs: None
        host_migration.recover_at_startup = lambda *args, **kwargs: False
        if spec.get("check_only"):
            import socket
            def deny_network(*args, **kwargs):
                raise RuntimeError("Network disabled during offline preflight")
            socket.socket.connect = deny_network
            socket.socket.connect_ex = deny_network
        threading.Thread(target=receive, daemon=True).start()
        allowed = {schema["name"] for schema in spec["app_schemas"]}
        for schema in spec["app_schemas"]:
            def handler(arguments, _name=schema["name"], **kwargs):
                return rpc(_name, arguments, "app")
            registry.register(name=schema["name"], toolset="dolphinbench-bound",
                              schema=schema, handler=handler)
        create_custom_toolset("dolphinbench-bound", "DolphinBench supplied apps", tools=sorted(allowed))
        manager = MemoryManager()
        manager.add_provider(BoundMemory())
        manager.initialize_all(spec["session_id"])
        agent = AIAgent(
            model=spec["model"], provider="openai-api", api_mode="codex_responses",
            api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1",
            enabled_toolsets=["dolphinbench-bound", "memory"], memory_manager=manager,
            max_iterations=spec["max_turns"], run_budget_seconds=spec["timeout"],
            reasoning_config={"effort": spec["reasoning"]}, session_id=spec["session_id"],
            quiet_mode=True, skip_context_files=True, load_soul_identity=False,
            skip_memory=False, skip_background_review=True, fallback_model={},
        )
        # Tool-format adapter only: preserve native loop/tool dispatch and every tool's
        # schema. Direct tools avoid the demonstrated batched tool_call history rewrite.
        agent.tools = [{"type": "function", "function": schema} for schema in
                       spec["app_schemas"] + spec["memory_schemas"]]
        agent.valid_tool_names = {tool["function"]["name"] for tool in agent.tools}
        agent._skip_mcp_refresh = True  # App MCP is exclusively owned by the template/controller.
        # Hermes's general plugin payload sanitizer truncates deep/large schemas.
        # Supply complete permitted settings to the observer without touching the API request.
        from harness.submission import SETTINGS_FIELDS
        agent._api_request_payload_for_hook = lambda payload: {"method": "POST", "body": {
            key: json_value(value) for key, value in (payload or {}).items()
            if key in SETTINGS_FIELDS or key == "reasoning"}}
        emit({"event": "runtime", "python": sys.version, "dependencies": {
            name: importlib.metadata.version(name) for name in ("openai", "httpx")}})
        if agent._memory_store is not None or agent._memory_manager is not manager:
            raise RuntimeError("Unexpected personal memory store or inactive Memanto manager")
        if spec.get("check_only"):
            emit({"event": "preflight", "status": "passed", "tools": sorted(agent.valid_tool_names)})
            return
        emit({"event": "stage", "stage": "native_conversation_starting"})
        result = agent.run_conversation(spec["message"], system_message=spec["system_prompt"],
                                        conversation_history=[])
        emit({"event": "native_result", "value": result})
        if not manager.flush_pending(timeout=spec["write_timeout"]):
            raise TimeoutError("Native Hermes memory-hook queue did not drain")
        if hook_errors:
            raise RuntimeError("Native Memanto sync hook failed: " + hook_errors[0])
        agent.shutdown_memory_provider(result.get("messages", []))
        if manager.shutdown_drain_state.get("status") != "drained":
            raise RuntimeError("Native Hermes memory shutdown did not drain")
        path = Path(spec["recording_directory"]) / f"{spec['session_id']}.jsonl"
        rollout = read_recording(path)  # No fabricated or rewritten fallback on incompatibility.
        # Use Hermes's own cache-aware accounting code and its recorded pricing source.
        # This is a model-cost estimate, not a provider invoice or full-run charge.
        native_costs = []
        try:
            from agent.usage_pricing import CanonicalUsage, estimate_usage_cost
            for event in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
                usage = event["assistant"]["usage"]
                fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
                if any(type(usage.get(field)) is not int or usage[field] < 0 for field in fields):
                    raise ValueError("Incomplete normalized provider usage for native pricing")
                canonical = CanonicalUsage(**{field: usage[field] for field in fields}, raw_usage=usage)
                cost = estimate_usage_cost(event["settings"]["model"], canonical,
                                           provider="openai-api", base_url="https://api.openai.com/v1")
                native_costs.append({"amount_usd": float(cost.amount_usd) if cost.amount_usd is not None else None,
                                     "status": cost.status, "source": cost.source,
                                     "pricing_version": cost.pricing_version, "notes": list(cost.notes)})
        except Exception as exc:
            native_costs = [{"amount_usd": None, "status": "unavailable", "reason": str(exc)}]
        emit({"event": "complete", "rollout": rollout, "native_costs": native_costs})
    except BaseException as exc:
        emit({"event": "failure", "type": type(exc).__name__, "error": str(exc)})
        raise


D = Decimal
API_RATE = D('0.00005')  # $0.05 / 1,000 non-generation requests
ANSWER_RATE = D('0.01')  # per model-tier unit per answer
INGESTION_RATE = D('0.01')  # per confirmed billable 10-MB block


def number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a cost/usage amount')
    result = D(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('Usage and cost must be finite and nonnegative')
    return result


def count(value):
    result = number(value)
    if result != result.to_integral_value():
        raise ValueError('Request counts must be integers')
    return result


def pricing_read(path):
    return json.loads(path.read_text())


def observed(root, phase):
    """Worker events are a diagnostic proxy, NOT backend billing telemetry."""
    files = sorted((root / 'adapter' / phase).glob('*/*/worker-events.json'))
    answers = 0
    for path in files:
        events = pricing_read(path)
        identifiers = set()
        for event in events:
            if (event.get('event') == 'request' and event.get('kind') == 'memory'
                    and event.get('operation') == 'memanto_answer'):
                identifier = event.get('id')
                if not identifier:
                    raise ValueError(f'Memory event without correlation ID: {path}')
                identifiers.add(identifier)
        answers += len(identifiers)
    return answers, len(files)


def memory_pricing_report(root, phase, tier, billing):
    records = sorted((root / 'adapter' / phase).glob('*/*/cost.json'))
    costs = [pricing_read(p).get('agent_usd') for p in records]
    known = sum((number(v) for v in costs if v is not None), D(0))
    agent = known if costs and all(v is not None for v in costs) else None
    answers, event_files = observed(root, phase)
    phase_usage = billing.get('phases', {}).get(phase)
    memory = None
    breakdown = None
    if phase_usage is not None:
        if not phase_usage.get('source'):
            raise ValueError('Each billing phase needs an evidence source')
        # Caller supplies provider-confirmed billable counts, including retries,
        # setup, polling and checkpoint reads attributable to this phase.
        by_tier = phase_usage['answer_calls_by_tier']
        if set(by_tier) != {'1', '2', '3'}:
            raise ValueError('answer_calls_by_tier must contain exactly 1, 2, 3')
        generation = sum((count(by_tier[str(i)]) * i * ANSWER_RATE
                          for i in (1, 2, 3)), D(0))
        api = count(phase_usage['non_generation_requests']) * API_RATE
        ingestion = number(phase_usage['billable_ingestion_10mb_units']) * INGESTION_RATE
        memory = generation + api + ingestion
        breakdown = {'answer_usd': float(generation), 'non_generation_usd': float(api),
                     'ingestion_usd': float(ingestion), 'source': phase_usage['source']}
    return {
        'phase': phase, 'agent_cost_records': len(records),
        'recorded_agent_usd': None if agent is None else float(agent),
        'known_agent_subtotal_usd': float(known),
        'observed_answer_tool_requests': answers, 'worker_event_files': event_files,
        'answer_only_scenario_usd': float(answers * tier * ANSWER_RATE) if tier is not None else None,
        'answer_only_scenario_assumption': (f'Each observed tool request incurs one tier-{tier} answer charge; excludes backend retries and other operations' if tier else 'Set memory_answer_tier to the confirmed Memanto answering model tier; no tier is guessed'),
        'memory_list_price_usd': None if memory is None else float(memory),
        'memory_breakdown': breakdown,
        'combined_recorded_scope_usd': float(agent + memory) if agent is not None and memory is not None else None,
        'total_cost_usd': None,
        'status': 'partial: agent retries/auxiliary coverage remains unaudited; not a submission phase total',
    }



class BenchmarkHarness:
    """Implement DolphinBench's five-method contract for the declared pilot configuration."""

    def __init__(self, options, work_dir):
        self.options = copy.deepcopy(options)
        self.work_dir = Path(work_dir).resolve()
        defaults = {"model": "gpt-6-luna", "reasoning": "medium", "max_turns": 6,
                    "timeout": 120, "write_timeout": 120, "top_k": 10, "api_attempts": 3, "rates": None,
                    "hermes_source": "~/.hermes/hermes-agent", "hermes_home": "~/.hermes"}
        allowed = set(defaults) | {"hermes_python", "memory_run_dir", "pilot", "judge_model", "judge_reasoning", "memory_answer_tier", "memory_billing_usage", "memanto_plugin_path", "memory_interface", "recall_min_similarity"}
        if set(self.options) - allowed:
            raise ValueError("Unknown options (credentials must not be placed in run.yaml)")
        for key, value in defaults.items():
            self.options.setdefault(key, value)
        for key in ("hermes_source", "hermes_python", "hermes_home"):
            if key not in self.options:
                raise ValueError(f"options.{key} is required")
            path = Path(self.options[key]).expanduser()
            # Do not resolve venv/bin/python to the base interpreter.
            self.options[key] = str(
                path.absolute() if key == "hermes_python" else path.resolve()
            )
        for key in ("max_turns", "timeout", "write_timeout", "top_k", "api_attempts"):
            if type(self.options[key]) is not int or self.options[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.options["top_k"] > 100:
            raise ValueError("top_k must not exceed 100")
        if self.options.get("memory_run_dir"):
            self.options["memory_run_dir"] = str(Path(self.options["memory_run_dir"]).expanduser().resolve())
        tier = self.options.get("memory_answer_tier")
        if tier is not None and (type(tier) is not int or tier not in (1, 2, 3)):
            raise ValueError("memory_answer_tier must be the confirmed tier 1, 2, or 3")
        if self.options.get("memory_billing_usage"):
            self.options["memory_billing_usage"] = str(Path(self.options["memory_billing_usage"]).expanduser().resolve())
        if self.options.get("memory_interface", "answer") not in {"answer", "recall"}:
            raise ValueError("memory_interface must be answer or recall")
        if self.options.get("memanto_plugin_path"):
            self.options["memanto_plugin_path"] = str(Path(self.options["memanto_plugin_path"]).expanduser().resolve())
        threshold = self.options.get("recall_min_similarity")
        if threshold is not None:
            if self.options.get("memory_interface") != "recall":
                raise ValueError("recall_min_similarity requires recall mode")
            if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
                raise ValueError("recall_min_similarity must be between 0 and 1")
        self.adaptive_recall_limit = options.get("top_k")
        self.app_python = sys.executable

    def identity(self):
        """Record code/configuration locally; do not contact models or memory services."""
        source = Path(self.options["hermes_source"])
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        difference = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
        interpreter = Path(self.options["hermes_python"])
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError("Hermes interpreter is not executable")
        native_versions = json.loads(subprocess.check_output([
            str(interpreter), "-c", "import sys,json,importlib.metadata as m; "
            "print(json.dumps({'python':sys.version,'openai':m.version('openai'),'httpx':m.version('httpx')}))"
        ], text=True, timeout=30))
        dependencies = {}
        for name in ("memanto", "moorcheh-sdk", "mcp", "PyYAML"):
            try:
                dependencies[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                dependencies[name] = "not installed in controller"
        recall_policy = f"recall budget={self.options['top_k']}"
        if self.options.get("recall_min_similarity") is not None:
            cap = self.adaptive_recall_limit
            limit_source = "configured"
            if cap is None:
                from memanto.cli.config.manager import ConfigManager

                cap = ConfigManager().get_recall_config()["limit"]
                limit_source = "SDK default"
            recall_policy = (f"recall min_similarity={self.options['recall_min_similarity']}, "
                             f"{limit_source} limit={cap}, no type filter")
        return {"harness": "dolphinbench-hermes-memanto-pilot", "options": self.options,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "hermes_revision": revision, "hermes_diff_sha256": hashlib.sha256(difference.encode()).hexdigest(),
                "controller_dependencies": dependencies, "hermes_dependencies": native_versions,
                "memanto_provider_source": provider_source(self.options)[1],
                "memanto_provider_sha256": hashlib.sha256(provider_source(self.options)[0].encode()).hexdigest(),
                "tool_presentation": "direct schemas; native Hermes loop and autonomous tool selection",
                "memory_policy": f"single retrieval interface memanto_{self.options.get('memory_interface', 'answer')}; {recall_policy}; ingestion memanto_remember only; auto_recall=false; no built-in personal memory",
                "internal_model_calls": "Memanto answer uses its configured native backend; its model context/usage are not captured",
                "ingestion_policy": "ordered history; fresh Hermes conversation per message; memanto_remember only; automatic capture disabled; writes verified before completion",
                "agent_controls": "configured iteration/time/API-attempt limits; automatic recovery and background review disabled",
                "completion_policy": "join provider threads and observe each acknowledged ID in semantic search",
                "cost_scope": "native Hermes recorded-response estimates plus published-rate Moorcheh accounting; unknown request coverage stays null",
                "packaging": "disabled: pilot judge and cost scope differ from submission requirements"}

    def agent_id(self, persona):
        if self.options.get("memory_run_dir"):
            checkpoint = read(Path(self.options["memory_run_dir"]) / "checkpoints" / f"{persona}.json")
            return checkpoint["agent_id"]
        return f"db-{hashlib.sha256(str(self.work_dir).encode()).hexdigest()[:16]}-{persona}"

    def run_interaction(self, request):
        return asyncio.run(self._run(request))

    async def _run(self, request):
        if request.phase == "ingestion":
            return await self.run_agent(request, [], None)
        from examples.mcp_connection import connect_apps
        async with connect_apps(request.apps) as apps:
            tools = (await apps.list_tools()).tools
            return await self.run_agent(request, tools, apps.call_tool)

    async def run_agent(self, request, tools, call_app):
        """Run one fresh native Hermes conversation with phase-bound Memanto hooks."""
        from harness.adapter import InteractionRecord
        if request.phase not in {"ingestion", "tests"}:
            raise ValueError("Unknown interaction phase")
        if request.phase == "ingestion" and self.options.get("memory_run_dir"):
            raise ValueError("Adopted frozen memory is retrieval-only; do not ingest into it")
        checkpoint = self.work_dir / "checkpoints" / f"{request.persona}.json"
        if (request.phase == "tests") != checkpoint.exists():
            raise ValueError("Testing requires a checkpoint; ingestion must precede freeze")
        folder = self.work_dir / "adapter" / request.phase / request.persona / str(request.interaction_id)
        # The DolphinBench runner owns successful resumption and .inflight markers.
        # Any adapter evidence left here means execution needs reconciliation.
        folder.mkdir(parents=True, exist_ok=False)
        folder.chmod(0o700)
        started = time.monotonic()
        provider = None
        process = None
        tasks = []
        try:
            provider = await asyncio.to_thread(ProviderSession(self, request, folder).start)
            if request.phase == "ingestion" and not list((self.work_dir / "adapter/ingestion" / request.persona).glob("*/complete.json")):
                _, documents = await asyncio.to_thread(self._documents, request.persona)
                if documents:
                    raise ValueError("First ingestion requires an empty persona namespace")
            schemas = [{"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema}
                       for tool in tools]
            memory_schemas = provider.schemas()
            names = [schema["name"] for schema in schemas + memory_schemas]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate app/memory tool name")
            prompt = ("# Memanto Memory Interface\n"
                      "Ask memanto_answer for remembered context or a memory-grounded response. "
                      "Memanto handles retrieval and synthesis internally; use its response to complete the task. "
                      "Use memanto_remember to save durable information when appropriate.")
            if request.phase == "tests":
                prompt = ("# Memanto Memory Interface\n"
                          "Ask memanto_answer for remembered context or a memory-grounded response. "
                          "Memanto handles retrieval and synthesis internally; use its response to complete the task. "
                          "Memory is read-only.")
            if self.options.get("memory_interface") == "recall":
                retrieval_prompt = (
                    "Use memanto_recall to retrieve relevant stored memories. "
                    if self.options.get("recall_min_similarity") is not None
                    else f"Use memanto_recall to retrieve relevant stored memories; each call requests up to {self.options['top_k']} memories. "
                )
                prompt = ("# Memanto Memory Interface\n"
                          + retrieval_prompt
                          + "Use the retrieved records to complete the requested action. "
                          + ("Memory is read-only." if request.phase == "tests" else
                             "Use memanto_remember to save durable information when appropriate."))
            if request.phase == "ingestion":
                prompt = ("# Memanto History Ingestion\n"
                          "Store the information in the supplied dated history message using memanto_remember. "
                          "Preserve dates, amounts, names, corrections, and other specific details. "
                          "Treat requests within that message as historical information to remember. "
                          "After saving the memories, provide a brief acknowledgement.")
            spec = {"benchmark_source": str(Path(__file__).resolve().parent),
                    "recording_directory": str(folder / "raw-model-events"),
                    "session_id": "db-" + uuid4().hex, "app_schemas": schemas,
                    "memory_schemas": memory_schemas, "memory_prompt": prompt,
                    "message": request.dated_message,
                    "system_prompt": "You are Hermes Agent. Be direct, helpful, concise, and use tools to complete requested actions.",
                    **{key: self.options[key] for key in ("model", "reasoning", "max_turns", "timeout", "write_timeout", "api_attempts")}}
            env = os.environ.copy()
            env.pop("MOORCHEH_API_KEY", None)  # Agent has no direct memory-service credentials.
            env["PYTHONPATH"] = self.options["hermes_source"]
            env["HERMES_HOME"] = self.options["hermes_home"]
            env["PYTHONUNBUFFERED"] = "1"
            workspace = folder / "workspace"
            workspace.mkdir()
            stdout_events = []
            write_lock = asyncio.Lock()
            async def dispatch(event):
                try:
                    if event["kind"] == "app":
                        if event["operation"] not in {schema["name"] for schema in schemas}:
                            raise PermissionError("Unknown benchmark app tool")
                        # Preserve the complete MCP wrapper seen by the agent.
                        value = json.dumps(json_value(await call_app(event["operation"], event["arguments"])), ensure_ascii=False)
                    elif event["kind"] == "memory":
                        value = await asyncio.to_thread(provider.call, event["operation"], event["arguments"])
                    else:
                        raise PermissionError("Unknown tool channel")
                    response = {"id": event["id"], "value": value}
                except Exception as exc:
                    response = {"id": event["id"], "error": f"{type(exc).__name__}: {exc}"}
                async with write_lock:
                    process.stdin.write((json.dumps(response) + "\n").encode())
                    await process.stdin.drain()
            with (folder / "stderr.log").open("w") as stderr:
                process = await asyncio.create_subprocess_exec(
                    self.options["hermes_python"], str(Path(__file__).resolve()), "_worker",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=stderr, env=env, cwd=workspace, limit=32 * 1024 * 1024)
                process.stdin.write((json.dumps(spec) + "\n").encode())
                await process.stdin.drain()
                rollout, native_costs = None, []
                async with asyncio.timeout(self.options["timeout"] + self.options["write_timeout"] + 30):
                    while line := await process.stdout.readline():
                        event = json.loads(line)
                        stdout_events.append(event)
                        save(folder / "worker-events.json", stdout_events)
                        if event["event"] == "request":
                            print(f"Hermes: {event['operation']}", flush=True)
                            tasks.append(asyncio.create_task(dispatch(event)))
                        elif event["event"] == "native_result":
                            save(folder / "native-result.json", event["value"])
                        elif event["event"] == "complete":
                            rollout = event["rollout"]
                            native_costs = event.get("native_costs", [])
                        elif event["event"] == "failure":
                            raise RuntimeError(f"Hermes {event['type']}: {event['error']}; inspect {folder}")
                    if tasks:
                        await asyncio.gather(*tasks)
                    code = await process.wait()
                    if code or rollout is None:
                        raise RuntimeError(f"Hermes exited {code}; inspect {folder / 'stderr.log'}")
            await asyncio.to_thread(provider.close)
            if request.phase == "ingestion" and not provider.receipts:
                raise RuntimeError("History message produced no acknowledged memory writes; inspect the saved recording")
            record = InteractionRecord(settings=rollout["settings"], messages=rollout["messages"],
                                       duration_ms=(time.monotonic() - started) * 1000)
            from harness.submission import check_messages, check_settings
            check_messages(record.messages, "native Hermes conversation")
            check_settings(record.settings, "native Hermes settings")
            if not _original_user_boundary(record.messages, request.dated_message):
                raise ValueError("Hermes changed the original dated user-message boundary")
            events_path = folder / "raw-model-events" / f"{spec['session_id']}.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            if self.options["rates"] is not None:
                amount = agent_cost(events, self.options["rates"])
                basis = "configured cache-aware rate overrides"
            else:
                amount = math.fsum(item["amount_usd"] for item in native_costs) if native_costs and all(
                    item.get("amount_usd") is not None for item in native_costs) else None
                basis = "native Hermes pricing estimates for recorded API responses"
            save(folder / "cost.json", {"agent_usd": amount, "native_costs": native_costs,
                                       "basis": basis, "moorcheh_usd": None, "scope": "recorded agent API responses",
                                       "retry_and_auxiliary_coverage": "not audited; raw requests retained"})
            save(folder / "complete.json", {"session_id": spec["session_id"], "phase": request.phase,
                                            "persona": request.persona, "memory_agent_id": self.agent_id(request.persona)})
            return record
        except BaseException as exc:
            save(folder / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                          "replay": "delivery uncertain; inspect evidence before retry"})
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            if provider is not None:
                await asyncio.to_thread(provider.close)

    def _sdk(self, persona):
        from memanto.cli.client.sdk_client import SdkClient
        sdk = SdkClient(api_key=os.environ["MOORCHEH_API_KEY"])
        sdk.activate_agent(self.agent_id(persona))
        return sdk

    def _documents(self, persona):
        """Enumerate every page of the actual memory store and reject ambiguous pagination."""
        sdk = self._sdk(persona)
        namespace = sdk.get_agent(self.agent_id(persona))["namespace"]
        documents, seen, token = {}, set(), None
        while True:
            arguments = {"namespace_name": namespace, "limit": 100}
            if token:
                arguments["next_token"] = token
            page = sdk._get_moorcheh().documents.fetch_text_data(**arguments)
            items = page.get("items")
            if not isinstance(items, list):
                raise ValueError("Unexpected Moorcheh enumeration response")
            for item in items:
                identifier = str(item["id"])
                if identifier in documents:
                    raise ValueError("Duplicate memory ID in enumeration")
                documents[identifier] = item
            pagination = page.get("pagination")
            if pagination is None:
                if len(items) >= 100:
                    raise ValueError("Full memory page without pagination metadata")
                break
            if type(pagination.get("has_more")) is not bool:
                raise ValueError("Invalid pagination metadata")
            if not pagination["has_more"]:
                break
            token = pagination.get("next_token")
            if not token or token in seen:
                raise ValueError("Missing or repeated memory pagination token")
            seen.add(token)
        return namespace, [documents[key] for key in sorted(documents)]

    def freeze(self, persona):
        """Preserve indexed acknowledged writes and a durable, stable live-store snapshot."""
        completed = list((self.work_dir / "adapter/ingestion" / persona).glob("*/complete.json"))
        if not completed:
            raise ValueError("No completed ingestion interactions")
        receipts = []
        for path in (self.work_dir / "adapter/ingestion" / persona).glob("*/memory-writes.json"):
            receipts.extend(read(path))
        if any(receipt.get("status") != "indexed" for receipt in receipts):
            raise ValueError("Ingestion has unconfirmed memory writes")
        namespace, documents = self._documents(persona)
        second_namespace, second = self._documents(persona)
        if namespace != second_namespace or digest(documents) != digest(second):
            raise ValueError("Memory changed during freeze")
        if not {str(receipt["response"]["memory_id"]) for receipt in receipts} <= {str(item["id"]) for item in documents}:
            raise ValueError("An acknowledged write is absent from the snapshot")
        snapshot = self.work_dir / "checkpoints" / f"{persona}.memories.json"
        save(snapshot, documents)
        return {"persona": persona, "agent_id": self.agent_id(persona), "namespace": namespace,
                "memory_count": len(documents), "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "content_sha256": digest(documents), "completion": "each acknowledged ID observed in semantic search"}

    def verify_checkpoint(self, persona, checkpoint):
        """Compare durable snapshot and live memory; never trust the checkpoint label alone."""
        root = Path(self.options.get("memory_run_dir", self.work_dir))
        snapshot = root / "checkpoints" / f"{persona}.memories.json"
        if checkpoint["persona"] != persona or checkpoint["agent_id"] != self.agent_id(persona):
            raise ValueError("Checkpoint belongs to another persona or memory store")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != checkpoint["snapshot_sha256"]:
            raise ValueError("Durable memory snapshot changed")
        namespace, documents = self._documents(persona)
        expected = read(snapshot)
        if namespace != checkpoint["namespace"] or len(documents) != checkpoint["memory_count"] or digest(documents) != digest(expected):
            raise ValueError("Live memory differs from the completed snapshot")

    def memory_cost_report(self, phase):
        """Price durable records; never rerun the agent or call a memory service.

        Optional billing usage contains provider-confirmed attributable request counts,
        not guesses derived from tool calls. Without it only an answer-call scenario
        can be estimated. No subscription, grading or infrastructure charge is added.
        """
        billing = read(Path(self.options["memory_billing_usage"])) if self.options.get("memory_billing_usage") else {}
        value = memory_pricing_report(self.work_dir, phase, self.options.get("memory_answer_tier"), billing)
        value.update(pricing_source="https://docs.moorcheh.ai/guides/pricing",
                     pricing_reviewed="2026-10-29", currency="USD",
                     basis="published-rate estimates, not actual invoiced charges")
        save(self.work_dir / "costs" / f"{phase}-memory-pricing.json", value)
        return value

    def total_cost_usd(self, phase):
        """Return the pilot's recorded agent cost; Moorcheh is explicitly excluded.

        This approved pilot scope is not the full-cost submission contract. Missing
        rates raise; the pilot runner preserves a null cost instead of inventing zero.
        """
        if phase not in {"ingestion", "tests"}:
            raise ValueError("Unknown cost phase")
        records = [read(path) for path in (self.work_dir / "adapter" / phase).glob("*/*/cost.json")]
        if not records or any(record["agent_usd"] is None for record in records):
            raise CostUnavailable("Agent pricing is unavailable; provider usage has been preserved")
        return math.fsum(record["agent_usd"] for record in records)


class CostUnavailable(ValueError):
    pass


def configure_judge(model, reasoning, directory):
    """Adapt only the judge transport/settings; keep DolphinBench's prompts and verdict logic."""
    import urllib.request
    from graders import llm_judge
    from graders.judge_recording import record_response
    import harness.runner as benchmark_runner
    benchmark_runner.JUDGE = {"model": model, "reasoning_effort": reasoning}
    lock = threading.Lock()
    llm_judge.BACKEND = "openai"
    llm_judge.OPENAI_DEFAULT_MODEL = model
    llm_judge.JUDGE_REASONING_EFFORT = reasoning

    def call(system_prompt, user_prompt, config, timeout=90):
        body = {"model": model, "reasoning_effort": reasoning,
                "max_completion_tokens": llm_judge.DEFAULT_MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}]}
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                     "Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        response = llm_judge._urlopen_json(request, timeout, model=model)
        duration = (time.monotonic() - started) * 1000
        record_response({key: value for key, value in body.items() if key != "messages"},
                        body["messages"], response, duration)
        with lock:
            path = directory / "judge-events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as output:
                output.write(json.dumps({"request": body, "response": response, "duration_ms": duration}) + "\n")
                output.flush()
                os.fsync(output.fileno())
        text = response["choices"][0]["message"].get("content")
        if not text:
            raise RuntimeError("Judge returned no verdict; do not count this as a task verdict")
        return llm_judge._extract_json(text)

    # Some retained task criteria explicitly select the Azure backend. Both names
    # route to the approved OpenAI transport, without changing criterion prompts.
    llm_judge._call_openai = call
    llm_judge._call_azure = call


def credentials():
    """Read only the two approved credential variables; never serialize their values."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        dotenv_values = None
    if dotenv_values:
        for path in (Path.cwd() / ".env", Path.home() / ".hermes/.env"):
            if path.is_file():
                values = dotenv_values(path)
                for key in ("OPENAI_API_KEY", "MOORCHEH_API_KEY"):
                    if values.get(key):
                        os.environ.setdefault(key, values[key])
    for key in ("OPENAI_API_KEY", "MOORCHEH_API_KEY"):
        if not os.environ.get(key):
            raise ValueError(f"Set {key} in your environment or existing .env; never in run.yaml")


def pilot_runner_class():
    """Reuse Runner execution/grading; override only explicitly partial pilot accounting."""
    from harness.runner import Runner

    class PilotRunner(Runner):
        def _saved_cost(self, phase):
            path = self.directory / "costs" / f"{phase}.json"
            if not path.exists():
                raise ValueError(f"Missing cost status for {phase}; resume that stage")
            return read(path).get("agent_usd")

        def _collect_cost(self, phase):
            path = self.directory / "costs" / f"{phase}.json"
            try:
                amount = self.adapter.total_cost_usd(phase)
            except CostUnavailable as exc:
                amount, reason = None, str(exc)
            else:
                reason = "Moorcheh cost unavailable; retries/auxiliary coverage not audited"
            memory = self.adapter.memory_cost_report(phase)
            save(path, {"agent_usd": amount, "total_cost_usd": None,
                        "moorcheh_usd": memory["memory_list_price_usd"],
                        "answer_only_estimate_usd": memory["answer_only_scenario_usd"],
                        "observed_answer_tool_requests": memory["observed_answer_tool_requests"],
                        "combined_recorded_scope_usd": memory["combined_recorded_scope_usd"],
                        "memory_pricing": memory,
                        "status": "partial" if amount is not None else "unavailable",
                        "reason": "Published-rate memory estimates; complete backend usage requires provider-confirmed counts. Agent retries/auxiliary coverage remains unaudited."})
            return amount

        def package(self, *args, **kwargs):
            raise ValueError("Packaging is disabled for this changed-judge, partial-cost pilot")

    return PilotRunner


def write_summary(directory, release):
    """Summarize saved verdicts and median runner latency without any service calls."""
    rows = []
    for persona, data in release.items():
        for spec in data["tests"]:
            item = str(spec["id"]).zfill(3)
            execution = directory / "tests" / persona / f"{item}.json"
            grade = directory / "grades" / persona / f"{item}.json"
            checks = read(grade)["checks"] if grade.exists() else None
            duration = read(execution)["row"]["duration_ms"] if execution.exists() else None
            rows.append({"persona": persona, "task_id": item, "duration_ms": duration,
                         "passed": all(check["passed"] for check in checks) if checks else None,
                         "status": "graded" if checks else "not graded"})
    graded = [row for row in rows if row["passed"] is not None]
    latencies = [row["duration_ms"] for row in rows if row["duration_ms"] is not None]
    costs = {phase: read(directory / "costs" / f"{phase}.json")
             for phase in ("ingestion", "tests") if (directory / "costs" / f"{phase}.json").exists()}
    result = {"selected_tasks": len(rows), "graded_tasks": len(graded),
              "passed_tasks": sum(row["passed"] for row in graded),
              "task_accuracy": sum(row["passed"] for row in graded) / len(graded) if graded else None,
              "accuracy_denominator": "graded tasks; incomplete runs are not final benchmark results",
              "median_task_latency_ms": statistics.median(latencies) if latencies else None,
              "costs": costs, "tasks": rows,
              "pilot_adaptations": ["OpenAI agent and judge models recorded in run identity and grade settings", "direct function tool presentation",
                                    "single Memanto retrieval interface; see run identity for selected mode and retrieval budget",
                                    "Memanto-only native provider hooks; auto_recall=false", "published-rate memory accounting; incomplete costs remain null; packaging disabled"]}
    save(directory / "pilot_results.json", result)
    return result


def runtime_check(harness):
    """Construct native Hermes offline; prove neither conversation nor provider hooks are run."""
    native = provider_module(harness.options)
    session = ProviderSession(harness, type("Request", (), {"phase": "tests"})(), Path("unused"))
    session.provider = native.MemantoMemoryProvider()
    schemas = session.schemas()
    with tempfile.TemporaryDirectory(prefix="dolphinbench-runtime-check-") as temporary:
        spec = {"check_only": True, "benchmark_source": str(Path(__file__).resolve().parent),
                "recording_directory": str(Path(temporary) / "recordings"),
                "session_id": "db-runtime-check", "app_schemas": [],
                "memory_schemas": schemas, "memory_prompt": "Memanto runtime check",
                "message": "Not sent", "system_prompt": "Not sent",
                **{key: harness.options[key] for key in ("model", "reasoning", "max_turns", "timeout", "write_timeout", "api_attempts")}}
        env = os.environ.copy()
        env["PYTHONPATH"] = harness.options["hermes_source"]
        env["HERMES_HOME"] = harness.options["hermes_home"]
        env["OPENAI_API_KEY"] = "offline-not-a-real-key"
        env.pop("MOORCHEH_API_KEY", None)
        result = subprocess.run([harness.options["hermes_python"], str(Path(__file__).resolve()), "_worker"],
                                input=json.dumps(spec) + "\n", text=True, capture_output=True,
                                env=env, cwd=temporary, timeout=60)
        if result.returncode:
            raise RuntimeError("Native Hermes preflight failed:\n" + result.stderr + "\n" + result.stdout)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        if not any(event.get("event") == "preflight" for event in events):
            raise RuntimeError("Worker returned no preflight evidence")
        print(json.dumps(events, indent=2))
        print("PASS: native Hermes construction only. No conversation, memory operation, or grading.")


def main():
    """Thin pilot entry point; all benchmark execution and grading stay in DolphinBench Runner."""
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "runtime-check", "ingest", "evaluate", "summary", "self-test"))
    parser.add_argument("--config", type=Path, default=Path("run.yaml"))
    parser.add_argument("--confirm-paid-calls", action="store_true")
    args = parser.parse_args()
    if args.stage == "self-test":
        return self_test()
    configuration = yaml.safe_load(args.config.read_text())
    if set(configuration) != {"release", "output", "adapter", "options"}:
        raise ValueError("run.yaml requires release, output, adapter, and options")
    expected_adapter = f"{Path(__file__).stem}:BenchmarkHarness"
    if configuration["adapter"] != expected_adapter:
        raise ValueError(f"Keep adapter: {expected_adapter}")
    base = args.config.resolve().parent
    root, output = (base / configuration["release"]).resolve(), (base / configuration["output"]).resolve()
    options = copy.deepcopy(configuration["options"])
    if options.get("memory_run_dir"):
        options["memory_run_dir"] = str((base / options["memory_run_dir"]).resolve())
    harness = BenchmarkHarness(options, output)
    if args.stage == "runtime-check":
        runtime_check(harness)
        return
    from harness.runner import read_release, run_lock
    complete_release = read_release(root)  # Original release validation and hashes.
    pilot = options.get("pilot", {"personas": ["alex"], "tasks": ["001"]})
    personas = pilot.get("personas", ["alex"])
    tasks = [str(item).zfill(3) for item in pilot.get("tasks", ["001"])]
    if not personas or len(personas) != len(set(personas)) or any(persona not in complete_release for persona in personas):
        raise ValueError("Invalid persona selection")
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in {f"{i:03d}" for i in range(1, 201)} for task in tasks):
        raise ValueError("Tasks must be unique IDs from 001 through 200")
    release = {persona: copy.deepcopy(complete_release[persona]) for persona in personas}
    for data in release.values():
        data["tests"] = [spec for spec in data["tests"] if str(spec["id"]).zfill(3) in tasks]
        if len(data["tests"]) != len(tasks):
            raise ValueError("Selected tasks are missing from the release")
    if args.stage == "summary":
        print(json.dumps(write_summary(output, release), indent=2))
        return
    if args.stage in {"ingest", "evaluate"}:
        if not args.confirm_paid_calls:
            raise ValueError("Execution requires --confirm-paid-calls")
        credentials()
    identity = {"config": configuration, "adapter": harness.identity(),
                "pilot_selection": {"personas": personas, "tasks": tasks}}
    # Record framework source locally. No dataset or grading evidence crosses to the agent.
    identity["framework"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for name in ("harness", "graders", "mock_mcp")
                             for path in sorted((root / name).rglob("*.py"))}
    with run_lock(output):
        # Intentional pilot-only change: fixed-Azure admission is bypassed, not spoofed.
        # Runner still owns history delivery, app state, evidence validation, grading,
        # completed-interaction reuse and .inflight handling.
        runner = pilot_runner_class()(release, output, harness, identity=identity, release_root=root, allow_paid=False)
        runner.allow_paid = bool(args.confirm_paid_calls)
        if args.stage == "prepare":
            print(json.dumps({"prepared": str(output), "model_calls": 0,
                              "selected_tasks": tasks, "personas": personas,
                              "cost_scope": "partial/unavailable; not submission totals"}, indent=2))
            return
        configure_judge(options.get("judge_model", "gpt-6-luna"), options.get("judge_reasoning", "medium"), output)
        if args.stage == "evaluate" and options.get("memory_run_dir"):
            # Diagnostic adoption is explicitly different from native-agent ingestion.
            memory_root = Path(options["memory_run_dir"])
            for persona in personas:
                source = memory_root / "checkpoints" / f"{persona}.json"
                checkpoint = read(source)
                snapshot = memory_root / "checkpoints" / f"{persona}.memories.json"
                if hashlib.sha256(snapshot.read_bytes()).hexdigest() != checkpoint["snapshot_sha256"]:
                    raise ValueError("Adopted snapshot does not match its checkpoint")
                target = output / "checkpoints" / f"{persona}.json"
                if target.exists() and read(target) != checkpoint:
                    raise ValueError("Adopted checkpoint changed")
                save(target, checkpoint)
            cost_path = output / "costs/ingestion.json"
            if not cost_path.exists():
                save(cost_path, {"agent_usd": None, "total_cost_usd": None, "moorcheh_usd": None,
                                 "status": "unavailable", "reason": "Adopted prior memory; ingestion accounting not established"})
            save(output / "memory_provenance.json", {"source": str(memory_root),
                 "scope": "diagnostic reuse; not evidence of native Hermes ingestion"})
        try:
            getattr(runner, args.stage)()
            if args.stage == "evaluate":
                for persona in personas:
                    harness.verify_checkpoint(persona, read(output / "checkpoints" / f"{persona}.json"))
        finally:
            write_summary(output, release)
        print(f"{args.stage} completed: {output}")


def self_test():
    """Exercise pricing, persistence, permissions and provider-hook draining without credentials."""
    import unittest
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    class Checks(unittest.TestCase):
        def test_openai_cache_is_not_double_counted(self):
            event = {"assistant": {"usage": {"prompt_tokens": 1000, "input_tokens": 5,
                      "cache_write_tokens": 195, "cache_read_tokens": 800, "output_tokens": 100}}}
            self.assertAlmostEqual(agent_cost([event], {"input": 2, "cached_input": .5, "cache_write": 2.5, "output": 8}), .0016975)

        def test_unknown_cost_is_not_zero(self):
            self.assertIsNone(agent_cost([], None))
            with self.assertRaises(ValueError):
                agent_cost([{"assistant": {}}], {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1})

        def test_bad_usage_rejected(self):
            for usage in ({"prompt_tokens": 1, "output_tokens": -1},
                          {"prompt_tokens": 1, "output_tokens": 1, "cache_read_tokens": 2}):
                with self.assertRaises(ValueError):
                    agent_cost([{"assistant": {"usage": usage}}], {"input": 1, "cached_input": 1, "cache_write": 1, "output": 1})

        def test_memory_published_rate_accounting(self):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                folder = root / "adapter/tests/alex/001"
                folder.mkdir(parents=True)
                save(folder / "cost.json", {"agent_usd": 0.001})
                event = {"event": "request", "kind": "memory", "operation": "memanto_answer", "id": "one"}
                save(folder / "worker-events.json", [event, event])
                result = memory_pricing_report(root, "tests", 3, {})
                self.assertEqual(result["observed_answer_tool_requests"], 1)
                self.assertEqual(result["answer_only_scenario_usd"], 0.03)
                self.assertIsNone(result["memory_list_price_usd"])
                self.assertIsNone(memory_pricing_report(root, "tests", None, {})["answer_only_scenario_usd"])
                billing = {"phases": {"tests": {"source": "confirmed usage",
                    "answer_calls_by_tier": {"1": 0, "2": 0, "3": 1},
                    "non_generation_requests": 20, "billable_ingestion_10mb_units": 1}}}
                result = memory_pricing_report(root, "tests", 3, billing)
                self.assertEqual(result["memory_list_price_usd"], 0.041)
                self.assertIsNone(result["total_cost_usd"])
                self.assertIsNone(memory_pricing_report(root, "ingestion", 3, {})["recorded_agent_usd"])

        def test_atomic_json(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "evidence.json"
                save(path, {"ok": True})
                with self.assertRaises(ValueError):
                    save(path, {"bad": math.nan})
                self.assertEqual(read(path), {"ok": True})

        def test_provider_is_upstream_source(self):
            module = provider_module()
            self.assertEqual({schema["name"] for schema in module.MemantoMemoryProvider().get_tool_schemas()},
                             {"memanto_remember", "memanto_recall", "memanto_answer"})

        def test_read_only_dispatch_and_capture(self):
            request = SimpleNamespace(phase="tests")
            harness = SimpleNamespace(options={"write_timeout": 1})
            session = ProviderSession(harness, request, Path("unused"))
            provider = Mock()
            provider.get_tool_schemas.return_value = [{"name": "memanto_remember"}, {"name": "memanto_recall"}, {"name": "memanto_answer"}]
            for attribute in ("_warmup_thread", "_sync_thread", "_write_thread"):
                setattr(provider, attribute, None)
            session.provider = provider
            with self.assertRaises(PermissionError):
                session.call("memanto_remember", {"content": "blocked"})
            with self.assertRaises(PermissionError):
                session.call("delete_memory", {})
            session.call("sync_turn", {"user_content": "test", "assistant_content": "answer"})
            provider.sync_turn.assert_not_called()
            self.assertEqual([s["name"] for s in session.schemas()], ["memanto_answer"])
            with self.assertRaises(PermissionError):
                session.call("memanto_recall", {"query": "question"})
            session.call("memanto_answer", {"question": "question"})
            provider.handle_tool_call.assert_called_once_with("memanto_answer", {"question": "question"})
            session.testing = False
            self.assertEqual({s["name"] for s in session.schemas()}, {"memanto_remember"})

        def test_ingestion_blocks_reads_and_automatic_capture(self):
            provider = Mock()
            provider.get_tool_schemas.return_value = [
                {"name": name} for name in ("memanto_remember", "memanto_recall", "memanto_answer")
            ]
            for attribute in ("_warmup_thread", "_sync_thread", "_write_thread"):
                setattr(provider, attribute, None)
            session = ProviderSession(
                SimpleNamespace(options={"write_timeout": 1, "memory_interface": "recall"}),
                SimpleNamespace(phase="ingestion"), Path("unused"),
            )
            session.provider = provider
            self.assertEqual([schema["name"] for schema in session.schemas()], ["memanto_remember"])
            for name in ("memanto_recall", "memanto_answer", "delete_memory"):
                with self.assertRaises(PermissionError):
                    session.call(name, {"query": "history"})
            self.assertEqual(session.call("prefetch", {"query": "history"}), "")
            session.call("sync_turn", {"user_content": "history", "assistant_content": "stored"})
            provider.prefetch.assert_not_called()
            provider.sync_turn.assert_not_called()
            session.call("memanto_remember", {"content": "dated history"})
            provider.handle_tool_call.assert_called_once_with(
                "memanto_remember", {"content": "dated history"},
            )

        def test_ingestion_runs_without_apps_and_keeps_request(self):
            from unittest.mock import AsyncMock
            harness = BenchmarkHarness({"hermes_python": sys.executable}, Path("unused"))
            request = SimpleNamespace(phase="ingestion", persona="riley", dated_message="[date] history")
            record = object()
            with patch.object(harness, "run_agent", new=AsyncMock(return_value=record)) as run:
                self.assertIs(asyncio.run(harness._run(request)), record)
                run.assert_awaited_once_with(request, [], None)

        def test_persona_stores_are_separate(self):
            harness = BenchmarkHarness({"hermes_python": sys.executable}, Path("unused"))
            identifiers = [harness.agent_id(persona) for persona in ("alex", "morgan", "riley")]
            self.assertEqual(len(set(identifiers)), 3)
            self.assertTrue(identifiers[1].endswith("-morgan"))
            self.assertTrue(identifiers[2].endswith("-riley"))

        def test_background_errors_are_not_swallowed(self):
            session = ProviderSession(SimpleNamespace(options={"write_timeout": 1}),
                                      SimpleNamespace(phase="ingestion"), Path("unused"))
            session.provider = SimpleNamespace()
            session.errors.append("service write failed")
            with self.assertRaisesRegex(RuntimeError, "service write failed"):
                session._drain()

        def test_native_write_is_confirmed_before_success(self):
            native = provider_module()
            backend = SimpleNamespace(similarity_search=SimpleNamespace(query=Mock(return_value={"results": [{"id": "m-1"}]})))
            sdk = SimpleNamespace(get_agent=lambda identifier: {"namespace": "ns"}, _get_moorcheh=lambda: backend)
            class Client:
                agent_id = "agent-alex"
                _client = sdk
                def __init__(self, *args, **kwargs):
                    pass
                def ensure_session(self):
                    pass
                def set_profile_path(self, path):
                    pass
                def remember(self, **kwargs):
                    return {"memory_id": "m-1", "status": "queued"}
            native._MemantoClient = Client
            harness = SimpleNamespace(options={"write_timeout": 1, "top_k": 10}, agent_id=lambda persona: "agent-alex")
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MOORCHEH_API_KEY": "offline"}), patch(
                    __name__ + ".provider_module", return_value=native):
                session = ProviderSession(harness, SimpleNamespace(phase="ingestion", persona="alex"), Path(directory)).start()
                self.assertEqual(session.provider.prefetch("Do not append to the dated user input"), "")
                result = json.loads(session.call("memanto_remember", {"content": "A durable verification fact."}))
                self.assertTrue(result["saved"])
                self.assertEqual(read(Path(directory) / "memory-writes.json")[0]["status"], "indexed")
                backend.similarity_search.query.assert_called_once()
                session.close()

        def test_native_write_guard_even_if_tool_is_called_internally(self):
            native = provider_module()
            class Client:
                agent_id = "agent-alex"
                def __init__(self, *args, **kwargs):
                    pass
                def ensure_session(self):
                    pass
                def set_profile_path(self, path):
                    pass
                def remember(self, **kwargs):
                    raise AssertionError("Backend write must not be reached")
            native._MemantoClient = Client
            harness = SimpleNamespace(options={"write_timeout": 1, "top_k": 10}, agent_id=lambda persona: "agent-alex")
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MOORCHEH_API_KEY": "offline"}), patch(
                    __name__ + ".provider_module", return_value=native):
                session = ProviderSession(harness, SimpleNamespace(phase="tests", persona="alex"), Path(directory)).start()
                self.assertNotIn("memanto_remember", {schema["name"] for schema in session.schemas()})
                with self.assertRaises(PermissionError):
                    session.provider._client.remember(content="blocked")
                session.close()

        def test_freeze_and_actual_checkpoint_verification(self):
            with tempfile.TemporaryDirectory() as directory:
                harness = BenchmarkHarness({"hermes_python": sys.executable}, Path(directory))
                evidence = Path(directory) / "adapter/ingestion/alex/000001/complete.json"
                save(evidence, {"complete": True})
                save(evidence.with_name("memory-writes.json"), [{"status": "indexed", "response": {"memory_id": "m-1"}}])
                documents = [{"id": "m-1", "text": "original"}]
                with patch.object(harness, "_documents", return_value=("ns", documents)):
                    checkpoint = harness.freeze("alex")
                    harness.verify_checkpoint("alex", checkpoint)
                with patch.object(harness, "_documents", return_value=("ns", [{"id": "m-1", "text": "changed"}])):
                    with self.assertRaises(ValueError):
                        harness.verify_checkpoint("alex", checkpoint)

        def test_credentials_cannot_enter_identity_options(self):
            with self.assertRaises(ValueError):
                BenchmarkHarness({"hermes_python": sys.executable, "api_key": "not-allowed"}, Path("unused"))

        @staticmethod
        def recorded_events():
            usage = {"input_tokens": 10, "output_tokens": 5, "prompt_tokens": 10, "cache_read_tokens": 0}
            original = [{"role": "system", "content": "Offline system"},
                        {"role": "user", "content": "[2026-10-26T12:00:00Z] Offline task"}]
            assistant = {"role": "assistant", "tool_calls": [
                {"id": "memory-1", "name": "memanto_answer", "arguments": {"question": "Find context"}},
                {"id": "app-1", "name": "send_slack_dm", "arguments": {"recipient": "hema", "message": "Offline"}}],
                "usage": usage}
            tool_results = [{"role": "tool", "tool_call_id": "memory-1", "content": '{"answer":"context"}'},
                            {"role": "tool", "tool_call_id": "app-1", "content": '{"ok":true}'}]
            settings = {"model": "offline-contract-fixture"}
            first = {"settings": settings, "messages": original, "assistant": assistant}
            without_usage = {key: value for key, value in assistant.items() if key != "usage"}
            second = {"settings": settings, "messages": original + [without_usage] + tool_results,
                      "assistant": {"role": "assistant", "content": "Completed offline", "usage": usage}}
            return [first, second]

        def test_actual_dolphinbench_recorder_accepts_parallel_direct_tools(self):
            try:
                from harness.submission_capture import build_rollout
                from harness.submission import check_messages
            except ModuleNotFoundError:
                self.skipTest("Run from DolphinBench root to exercise the real recorder")
            rollout = build_rollout(self.recorded_events())
            calls = check_messages(rollout["messages"], "offline parallel direct-tools regression")
            self.assertEqual([call["tool"] for call in calls], ["memanto_answer", "send_slack_dm"])

        def test_actual_dolphinbench_runner_grades_and_resumes_without_reexecution(self):
            try:
                from harness.runner import Runner
                from harness.adapter import InteractionRecord
                from harness.submission_capture import build_rollout
            except ModuleNotFoundError:
                self.skipTest("Run from DolphinBench root to exercise the real runner")
            rollout = build_rollout(self.recorded_events())
            adapter = Mock()
            adapter.run_interaction.return_value = InteractionRecord(rollout["settings"], rollout["messages"])
            spec = {"id": "001", "test": "Offline task", "narrative_anchor_date": "2026-10-26T12:00:00Z",
                    "mock_state": {}, "grade": {"type": "tool_trace", "config": {"required_tool": "send_slack_dm"}}}
            with tempfile.TemporaryDirectory() as directory:
                runner = Runner({}, Path(directory), adapter, identity={"offline": True})
                evidence = runner._execute("tests", "alex", spec)
                grade = runner._grade(spec, evidence)
                self.assertTrue(all(check["passed"] for check in grade["checks"]))
                runner._execute("tests", "alex", spec)
                adapter.run_interaction.assert_called_once()

        def test_hermes_interpreter_symlink_is_preserved(self):
            with tempfile.TemporaryDirectory() as directory:
                interpreter = Path(directory) / "venv/bin/python"
                interpreter.parent.mkdir(parents=True)
                interpreter.symlink_to(sys.executable)
                harness = BenchmarkHarness(
                    {"hermes_python": str(interpreter)}, Path(directory) / "run"
                )
                self.assertEqual(
                    harness.options["hermes_python"], str(interpreter.absolute())
                )
                self.assertNotEqual(
                    harness.options["hermes_python"], str(interpreter.resolve())
                )

        def test_constructor_is_local(self):
            with patch("socket.socket.connect", side_effect=AssertionError("network")):
                harness = BenchmarkHarness({"hermes_python": sys.executable}, Path("unused"))
                self.assertEqual(harness.options["model"], "gpt-6-luna")

        def test_missing_cost_records_raise(self):
            with tempfile.TemporaryDirectory() as directory:
                harness = BenchmarkHarness({"hermes_python": sys.executable}, Path(directory))
                with self.assertRaises(CostUnavailable):
                    harness.total_cost_usd("tests")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("OFFLINE CHECKS ONLY. Native Hermes execution, service responses and live grading remain unverified.")


# Embedded source placeholder is replaced with exact reviewed upstream bytes.
MEMANTO_PROVIDER_SOURCE = '"""Memanto memory-agent provider for the Hermes agent.\n\nMemanto (https://memanto.ai) is a *memory agent*: typed long-term memory with\nconfidence + provenance, semantic recall, and RAG-style answers, backed by the\nMoorcheh vector platform. Unlike a passive "memory layer", every namespace in\nMemanto is a first-class **agent** (``memanto agent create/activate``), so this\nprovider maps one Hermes identity to one Memanto agent.\n\nWhat it gives Hermes:\n  * Auto-recall — relevant memories injected before each turn (prefetch).\n  * Turn capture — conversation turns persisted as ``event`` memories.\n  * Explicit tools — ``memanto_remember`` / ``memanto_recall`` / ``memanto_answer``.\n  * Built-in memory mirroring — Hermes ``memory`` writes echoed into Memanto.\n\nThis module is the single source of truth for the provider. The installer\n(``hermes-memanto-install``) copies it verbatim into\n``$HERMES_HOME/plugins/memanto/__init__.py`` so Hermes discovers it as a\ndirectory plugin. The ``agent`` / ``tools`` imports below resolve against the\nhost Hermes at runtime; when imported standalone (e.g. unit tests, packaging)\nthey fall back to minimal stand-ins so the module still imports.\n\nThe Memanto SDK (``memanto``) is imported lazily so this module loads even when\nthe package isn\'t installed; ``is_available()`` then returns False and the\nprovider stays inert.\n"""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nimport logging\nimport os\nimport re\nimport threading\nimport time\nfrom collections.abc import Callable\nfrom pathlib import Path\nfrom typing import Any, TypeVar\n\ntry:  # resolved against the host Hermes at runtime\n    from agent.memory_provider import MemoryProvider\nexcept Exception:  # pragma: no cover - running outside a Hermes runtime\n\n    class MemoryProvider:  # type: ignore[no-redef]\n        """Minimal stand-in so this module imports without a Hermes install."""\n\n\ntry:  # resolved against the host Hermes at runtime\n    from tools.registry import tool_error\nexcept Exception:  # pragma: no cover - running outside a Hermes runtime\n\n    def tool_error(message: str) -> str:  # type: ignore[misc]\n        return json.dumps({"error": message})\n\n\nlogger = logging.getLogger(__name__)\n\n# -- Defaults -----------------------------------------------------------------\n\n_DEFAULT_AGENT_ID = "hermes-{identity}"\n_DEFAULT_PATTERN = "tool"\n_VALID_PATTERNS = ("support", "project", "tool")\n_DEFAULT_MAX_RECALL_RESULTS = 10\n_DEFAULT_CONFIDENCE = 0.85\n_CAPTURE_CONFIDENCE = 0.6\n_MIN_CAPTURE_LENGTH = 10\n_MAX_TITLE_LENGTH = 100\n_MAX_AGENT_ID_LENGTH = 64\n_ACTIVATION_RETRY_COOLDOWN = 60.0\n\n\ndef _sanitize_agent_id(raw: str) -> str:\n    """Sanitize charset and append a stable hash if over 64 chars."""\n    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)\n    if len(sanitized) > _MAX_AGENT_ID_LENGTH:\n        suffix = hashlib.sha256(raw.encode()).hexdigest()[:8]\n        sanitized = sanitized[: _MAX_AGENT_ID_LENGTH - len(suffix) - 1] + "-" + suffix\n    return sanitized\n\n\n# Memory taxonomy mirrored from memanto.app.constants.VALID_MEMORY_TYPES so the\n# tool schema matches what the backend validates. Kept as a literal list to\n# avoid importing memanto at module-import time.\n_VALID_MEMORY_TYPES = (\n    "fact",\n    "preference",\n    "goal",\n    "decision",\n    "artifact",\n    "learning",\n    "event",\n    "instruction",\n    "relationship",\n    "context",\n    "observation",\n    "commitment",\n    "error",\n)\n\n_TRIVIAL_RE = re.compile(\n    r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\\.?$",\n    re.IGNORECASE,\n)\n_MEMORY_OPEN_TAG = r"<\\s*memanto-memory(?:\\s+[^>]*)?\\s*>"\n_MEMORY_CLOSE_TAG = r"<\\s*/\\s*memanto-memory\\s*>"\n_CONTEXT_STRIP_RE = re.compile(\n    rf"{_MEMORY_OPEN_TAG}[\\s\\S]*?{_MEMORY_CLOSE_TAG}\\s*",\n    re.IGNORECASE,\n)\n_RECALL_TAG_RE = re.compile(\n    rf"{_MEMORY_OPEN_TAG}|{_MEMORY_CLOSE_TAG}",\n    re.IGNORECASE,\n)\n_REFRESH_THROTTLE_SECONDS = 5.0\n_TOKEN_FILE_NAME = ".memanto_session_token"\n\n_T = TypeVar("_T")\n\n\ndef _resolve_hermes_home() -> str:\n    """Best-effort Hermes home: ask the host, else ``$HERMES_HOME``, else ~/.hermes."""\n    try:\n        from hermes_constants import get_hermes_home\n\n        return str(get_hermes_home())\n    except Exception:\n        return os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")\n\n\ndef _default_config() -> dict:\n    return {\n        "agent_id": _DEFAULT_AGENT_ID,\n        "pattern": _DEFAULT_PATTERN,\n        "auto_recall": True,\n        "auto_capture": True,\n        "auto_create": True,\n        "mirror_memory_writes": True,\n        "max_recall_results": _DEFAULT_MAX_RECALL_RESULTS,\n        "min_confidence": None,\n        "session_duration_hours": None,\n    }\n\n\ndef _as_bool(value: Any, default: bool) -> bool:\n    if isinstance(value, bool):\n        return value\n    if isinstance(value, str):\n        lowered = value.strip().lower()\n        if lowered in {"true", "1", "yes", "y", "on"}:\n            return True\n        if lowered in {"false", "0", "no", "n", "off"}:\n            return False\n    return default\n\n\ndef _detect_memory_type(text: str) -> str:\n    lowered = text.lower()\n    if re.search(r"prefer|like|love|hate|want", lowered):\n        return "preference"\n    if re.search(r"decided|will use|going with|chose", lowered):\n        return "decision"\n    if re.search(r"\\bis\\b|\\bare\\b|\\bhas\\b|\\bhave\\b", lowered):\n        return "fact"\n    return "observation"\n\n\ndef _load_memanto_config(hermes_home: str) -> dict:\n    config = _default_config()\n    config_path = Path(hermes_home) / "memanto.json"\n    if config_path.exists():\n        try:\n            raw = json.loads(config_path.read_text(encoding="utf-8"))\n            if isinstance(raw, dict):\n                config.update({k: v for k, v in raw.items() if v is not None})\n        except Exception:\n            logger.debug("Failed to parse %s", config_path, exc_info=True)\n\n    # agent_id is kept raw here — the {identity} template is resolved (and the\n    # result sanitized) in initialize() once agent_identity is known.\n    config["agent_id"] = (\n        str(config.get("agent_id") or _DEFAULT_AGENT_ID).strip() or _DEFAULT_AGENT_ID\n    )\n    pattern = str(config.get("pattern") or _DEFAULT_PATTERN).strip().lower()\n    config["pattern"] = pattern if pattern in _VALID_PATTERNS else _DEFAULT_PATTERN\n    config["auto_recall"] = _as_bool(config.get("auto_recall"), True)\n    config["auto_capture"] = _as_bool(config.get("auto_capture"), True)\n    config["auto_create"] = _as_bool(config.get("auto_create"), True)\n    config["mirror_memory_writes"] = _as_bool(config.get("mirror_memory_writes"), True)\n    try:\n        config["max_recall_results"] = max(\n            1,\n            min(\n                100, int(config.get("max_recall_results", _DEFAULT_MAX_RECALL_RESULTS))\n            ),\n        )\n    except Exception:\n        config["max_recall_results"] = _DEFAULT_MAX_RECALL_RESULTS\n    if config.get("min_confidence") is not None:\n        try:\n            config["min_confidence"] = max(\n                0.0, min(1.0, float(config["min_confidence"]))\n            )\n        except Exception:\n            config["min_confidence"] = None\n    if config.get("session_duration_hours") is not None:\n        try:\n            config["session_duration_hours"] = max(\n                1, min(24 * 30, int(config["session_duration_hours"]))\n            )\n        except Exception:\n            config["session_duration_hours"] = None\n    return config\n\n\ndef _save_memanto_config(values: dict, hermes_home: str) -> None:\n    config_path = Path(hermes_home) / "memanto.json"\n    existing: dict = {}\n    if config_path.exists():\n        try:\n            raw = json.loads(config_path.read_text(encoding="utf-8"))\n            if isinstance(raw, dict):\n                existing = raw\n        except Exception:\n            existing = {}\n    existing.update(values)\n    config_path.write_text(\n        json.dumps(existing, indent=2, sort_keys=True) + "\\n", encoding="utf-8"\n    )\n\n\ndef _clean_text_for_capture(text: str) -> str:\n    return _CONTEXT_STRIP_RE.sub("", text or "").strip()\n\n\ndef _is_trivial_message(text: str) -> bool:\n    return bool(_TRIVIAL_RE.match((text or "").strip()))\n\n\ndef _memory_score(raw: dict) -> float | None:\n    """Coalesce the score key Memanto uses across endpoints."""\n    score = raw.get("score")\n    if score is None:\n        score = raw.get("similarity_score")\n    try:\n        return float(score) if score is not None else None\n    except (TypeError, ValueError):\n        return None\n\n\ndef _format_recall_block(memories: list[dict], max_results: int) -> str:\n    """Render recall hits into a context block for prefetch injection."""\n    lines = []\n    for item in (memories or [])[:max_results]:\n        content = (item.get("content") or item.get("title") or "").strip()\n        content = _RECALL_TAG_RE.sub("", content).strip()\n        if not content:\n            continue\n        prefix_bits = []\n        mem_type = item.get("type")\n        if mem_type:\n            prefix_bits.append(f"[{mem_type}]")\n        score = _memory_score(item)\n        if score is not None:\n            prefix_bits.append(f"[{round(score * 100)}%]")\n        prefix = " ".join(prefix_bits)\n        lines.append(f"- {prefix} {content}".strip())\n    if not lines:\n        return ""\n    intro = (\n        "Background from your Memanto memory agent. Use it silently when relevant; "\n        "do not force memories into the conversation."\n    )\n    body = "## Relevant Memories\\n" + "\\n".join(lines)\n    return f"<memanto-memory>\\n{intro}\\n\\n{body}\\n</memanto-memory>"\n\n\nclass _MemantoClient:\n    """Thin wrapper over Memanto\'s ``SdkClient`` pinned to one agent.\n\n    Mirrors the MCP integration\'s lifecycle: ensure the agent exists (creating\n    it on first use when ``auto_create``) and activate a session once, lazily,\n    guarded by a lock. After a failed activation we back off for a short\n    cooldown (``_ACTIVATION_RETRY_COOLDOWN``) so a transient backend blip does\n    not poison the client for the whole run, while still avoiding a network hit\n    on every turn.\n    """\n\n    def __init__(\n        self,\n        api_key: str,\n        agent_id: str,\n        *,\n        pattern: str = "tool",\n        auto_create: bool = True,\n        session_duration_hours: int | None = None,\n    ):\n        from memanto.cli.client.sdk_client import SdkClient\n\n        self._agent_id = agent_id\n        self._pattern = pattern\n        self._auto_create = auto_create\n        self._session_duration_hours = session_duration_hours\n        self._client = SdkClient(api_key=api_key)\n        self._lock = threading.RLock()\n        self._ready = False\n        self._retry_after = 0.0\n        self._profile_path: Path | None = None\n        self._token_file: Path | None = None\n        self._last_refresh = 0.0\n\n    @property\n    def agent_id(self) -> str:\n        return self._agent_id\n\n    def set_profile_path(self, profile_path: str) -> None:\n        self._profile_path = Path(profile_path)\n        self._token_file = self._profile_path / _TOKEN_FILE_NAME\n\n        # Load any existing persisted token on startup\n        token = self.load_token()\n        if token:\n            self._client.session_token = token\n            self._client.agent_id = self._agent_id\n            self._ready = True\n\n    def save_token(self, token: str) -> None:\n        if self._token_file:\n            try:\n                self._token_file.parent.mkdir(parents=True, exist_ok=True)\n                fd = os.open(\n                    str(self._token_file),\n                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,\n                    0o600,\n                )\n                with os.fdopen(fd, "w", encoding="utf-8") as handle:\n                    handle.write(token)\n                if os.name != "nt":\n                    os.chmod(self._token_file, 0o600)\n            except Exception:\n                logger.debug("Failed to save token to file", exc_info=True)\n\n    def load_token(self) -> str | None:\n        if self._token_file and self._token_file.exists():\n            try:\n                return self._token_file.read_text(encoding="utf-8").strip()\n            except Exception:\n                logger.debug("Failed to load token from file", exc_info=True)\n        return None\n\n    def auto_refresh(self) -> bool:\n        try:\n            with self._lock:\n                now = time.monotonic()\n                if now - self._last_refresh < _REFRESH_THROTTLE_SECONDS:\n                    return self._ready\n                self._last_refresh = now\n                self._ready = False\n                self._retry_after = 0.0\n                self.ensure_session()\n                return self._ready\n        except Exception:\n            logger.debug("Auto-refresh activation failed", exc_info=True)\n            return False\n\n    def _is_refreshable_auth_error(self, error: Exception) -> bool:\n        try:\n            from memanto.app.utils.errors import (\n                AuthenticationError,\n                AuthorizationError,\n                InvalidSessionTokenError,\n                SessionExpiredError,\n            )\n\n            return isinstance(\n                error,\n                (\n                    AuthenticationError,\n                    AuthorizationError,\n                    SessionExpiredError,\n                    InvalidSessionTokenError,\n                ),\n            )\n        except Exception:\n            return False\n\n    def _call_with_auth_retry(self, operation: str, call: Callable[[], _T]) -> _T:\n        self.ensure_session()\n        try:\n            return call()\n        except Exception as error:\n            if not self._is_refreshable_auth_error(error):\n                raise\n            logger.debug(\n                "Auth failure during %s, attempting auto-refresh",\n                operation,\n                exc_info=True,\n            )\n            if not self.auto_refresh():\n                raise\n            return call()\n\n    def ensure_session(self) -> None:\n        if self._ready:\n            return\n        with self._lock:\n            if self._ready:\n                return\n            if time.monotonic() < self._retry_after:\n                raise RuntimeError(\n                    "Memanto session activation is cooling down after a recent failure"\n                )\n            from memanto.app.utils.errors import (\n                AgentAlreadyExistsError,\n                AgentNotFoundError,\n            )\n\n            try:\n                try:\n                    self._client.get_agent(self._agent_id)\n                except AgentNotFoundError:\n                    if not self._auto_create:\n                        raise\n                    try:\n                        self._client.create_agent(\n                            self._agent_id,\n                            pattern=self._pattern,\n                            description="Created by Hermes",\n                        )\n                    except AgentAlreadyExistsError:\n                        pass  # race: another caller created it\n                res = self._client.activate_agent(\n                    self._agent_id,\n                    duration_hours=self._session_duration_hours,\n                )\n                if isinstance(res, dict):\n                    token = res.get("session_token")\n                    if token:\n                        self.save_token(token)\n                self._ready = True\n                self._retry_after = 0.0\n            except Exception:\n                self._retry_after = time.monotonic() + _ACTIVATION_RETRY_COOLDOWN\n                raise\n\n    def remember(\n        self,\n        *,\n        memory_type: str,\n        title: str,\n        content: str,\n        confidence: float,\n        tags: list[str] | None = None,\n        source: str = "hermes",\n        provenance: str = "explicit_statement",\n    ) -> dict:\n        return self._call_with_auth_retry(\n            "remember",\n            lambda: self._client.remember(\n                agent_id=self._agent_id,\n                memory_type=memory_type,\n                title=title,\n                content=content,\n                confidence=confidence,\n                tags=tags or [],\n                source=source,\n                provenance=provenance,\n            ),\n        )\n\n    def recall(\n        self,\n        query: str,\n        *,\n        limit: int,\n        type: list[str] | None = None,\n        min_confidence: float | None = None,\n    ) -> list[dict]:\n        result = self._call_with_auth_retry(\n            "recall",\n            lambda: self._client.recall(\n                agent_id=self._agent_id,\n                query=query,\n                limit=limit,\n                type=type,\n                min_confidence=min_confidence,\n            ),\n        )\n        return result.get("memories", [])\n\n    def answer(self, question: str, *, limit: int | None = None) -> dict:\n        return self._call_with_auth_retry(\n            "answer",\n            lambda: self._client.answer(\n                agent_id=self._agent_id,\n                question=question,\n                limit=limit,\n            ),\n        )\n\n\n# -- Tool schemas -------------------------------------------------------------\n\nREMEMBER_SCHEMA = {\n    "name": "memanto_remember",\n    "description": "Store a durable fact, preference, decision, goal, or instruction in the Memanto memory agent for future recall.",\n    "parameters": {\n        "type": "object",\n        "properties": {\n            "content": {\n                "type": "string",\n                "description": "The memory itself — one atomic, self-contained statement.",\n            },\n            "type": {\n                "type": "string",\n                "enum": list(_VALID_MEMORY_TYPES),\n                "description": "Semantic memory type. Use \'preference\' for likes/styles, \'fact\' for stable claims, \'decision\' for choices, \'goal\' for objectives, \'instruction\' for directives.",\n            },\n            "tags": {\n                "type": "array",\n                "items": {"type": "string"},\n                "description": "Optional lowercase tags for later filtering.",\n            },\n            "confidence": {\n                "type": "number",\n                "description": "How sure you are this is true, 0.0-1.0. Use 1.0 only for explicit statements.",\n            },\n        },\n        "required": ["content"],\n    },\n}\n\nRECALL_SCHEMA = {\n    "name": "memanto_recall",\n    "description": "Search the Memanto memory agent by semantic similarity. Use this FIRST before asking the user to repeat information.",\n    "parameters": {\n        "type": "object",\n        "properties": {\n            "query": {\n                "type": "string",\n                "description": "Natural-language search query.",\n            },\n            "limit": {\n                "type": "integer",\n                "description": "Maximum results to return, 1 to 100.",\n            },\n            "type": {\n                "type": "array",\n                "items": {"type": "string", "enum": list(_VALID_MEMORY_TYPES)},\n                "description": "Optional type filter, e.g. [\'preference\'].",\n            },\n        },\n        "required": ["query"],\n    },\n}\n\nANSWER_SCHEMA = {\n    "name": "memanto_answer",\n    "description": "Ask a natural-language question and get an answer grounded ONLY in the Memanto memory agent\'s stored memories (RAG). Prefer over memanto_recall when you need a synthesized answer rather than a ranked list.",\n    "parameters": {\n        "type": "object",\n        "properties": {\n            "question": {\n                "type": "string",\n                "description": "The question to answer from memory.",\n            },\n            "limit": {\n                "type": "integer",\n                "description": "Number of context memories to retrieve, 1 to 100.",\n            },\n        },\n        "required": ["question"],\n    },\n}\n\n\nclass MemantoMemoryProvider(MemoryProvider):\n    """Memanto-backed memory provider for Hermes."""\n\n    def __init__(self):\n        self._config = _default_config()\n        self._api_key = ""\n        self._client: _MemantoClient | None = None\n        self._agent_id = "hermes"\n        self._hermes_home = ""\n        self._auto_recall = True\n        self._auto_capture = True\n        self._mirror_memory_writes = True\n        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS\n        self._min_confidence: float | None = None\n        self._write_enabled = True\n        self._active = False\n        self._warmup_thread: threading.Thread | None = None\n        self._sync_thread: threading.Thread | None = None\n        self._write_thread: threading.Thread | None = None\n\n    @property\n    def name(self) -> str:\n        return "memanto"\n\n    def is_available(self) -> bool:\n        if not os.environ.get("MOORCHEH_API_KEY", "").strip():\n            return False\n        try:\n            __import__("memanto")\n            return True\n        except Exception:\n            return False\n\n    def get_config_schema(self):\n        return [\n            {\n                "key": "api_key",\n                "description": "Moorcheh API key (powers Memanto)",\n                "secret": True,\n                "required": True,\n                "env_var": "MOORCHEH_API_KEY",\n                "url": "https://console.moorcheh.ai/api-keys",\n            },\n            {\n                "key": "agent_id",\n                "description": "Memory agent id / namespace ({identity} expands to the profile name)",\n                "secret": False,\n                "required": False,\n                "default": _DEFAULT_AGENT_ID,\n            },\n        ]\n\n    def save_config(self, values, hermes_home):\n        sanitized = dict(values or {})\n        sanitized.pop("api_key", None)\n        # Keep the {identity} template intact; only sanitize concrete ids.\n\n        if "pattern" in sanitized:\n            pattern = str(sanitized["pattern"]).strip().lower()\n            sanitized["pattern"] = (\n                pattern if pattern in _VALID_PATTERNS else _DEFAULT_PATTERN\n            )\n        _save_memanto_config(sanitized, hermes_home)\n\n    def initialize(self, session_id: str, **kwargs) -> None:\n        self._hermes_home = kwargs.get("hermes_home") or _resolve_hermes_home()\n        self._config = _load_memanto_config(self._hermes_home)\n        self._api_key = os.environ.get("MOORCHEH_API_KEY", "").strip()\n\n        # Resolve the agent id: env override > config, with {identity} template.\n        identity = kwargs.get("agent_identity", "default") or "default"\n        safe_identity = _sanitize_agent_id(str(identity))\n        raw_id = (\n            os.environ.get("MEMANTO_AGENT_ID", "").strip() or self._config["agent_id"]\n        )\n        self._agent_id = _sanitize_agent_id(raw_id.replace("{identity}", identity))\n\n        self._auto_recall = self._config["auto_recall"]\n        self._auto_capture = self._config["auto_capture"]\n        self._mirror_memory_writes = self._config["mirror_memory_writes"]\n        self._max_recall_results = self._config["max_recall_results"]\n        self._min_confidence = self._config["min_confidence"]\n\n        agent_context = kwargs.get("agent_context", "")\n        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}\n\n        self._active = False\n        self._client = None\n        if not self._api_key:\n            return\n        try:\n            self._client = _MemantoClient(\n                self._api_key,\n                self._agent_id,\n                pattern=self._config["pattern"],\n                auto_create=self._config["auto_create"],\n                session_duration_hours=self._config["session_duration_hours"],\n            )\n            profile_dir = Path(self._hermes_home) / "profiles" / safe_identity\n            if hasattr(self._client, "set_profile_path"):\n                self._client.set_profile_path(str(profile_dir))\n            self._active = True\n        except Exception:\n            logger.warning("Memanto initialization failed", exc_info=True)\n            self._client = None\n            self._active = False\n            return\n\n        # Warm up the session in the background so the first turn\'s recall does\n        # not also pay agent-create + activate latency.\n        self._warmup_thread = threading.Thread(\n            target=self._warmup, daemon=True, name="memanto-warmup"\n        )\n        self._warmup_thread.start()\n\n    def _warmup(self) -> None:\n        try:\n            self._client.ensure_session()\n        except Exception:\n            logger.debug("Memanto warmup activation failed", exc_info=True)\n\n    def system_prompt_block(self) -> str:\n        if not self._active:\n            return ""\n        return (\n            "# Memanto Memory Agent\\n"\n            f"Active. Memory agent: {self._agent_id}.\\n"\n            "Use memanto_recall to look up stored memories, memanto_remember to save durable "\n            "facts/preferences/decisions/goals, and memanto_answer for a synthesized answer "\n            "grounded in memory."\n        )\n\n    def prefetch(self, query: str, *, session_id: str = "") -> str:\n        if (\n            not self._active\n            or not self._auto_recall\n            or not self._client\n            or not query.strip()\n        ):\n            return ""\n        try:\n            memories = self._client.recall(\n                query[:2000],\n                limit=self._max_recall_results,\n                min_confidence=self._min_confidence,\n            )\n            return _format_recall_block(memories, self._max_recall_results)\n        except Exception:\n            logger.debug("Memanto prefetch failed", exc_info=True)\n            return ""\n\n    def sync_turn(\n        self, user_content: str, assistant_content: str, *, session_id: str = ""\n    ) -> None:\n        if (\n            not self._active\n            or not self._auto_capture\n            or not self._write_enabled\n            or not self._client\n        ):\n            return\n        client = self._client\n        clean_user = _clean_text_for_capture(user_content)\n        clean_assistant = _clean_text_for_capture(assistant_content)\n        if (\n            len(clean_user) < _MIN_CAPTURE_LENGTH\n            or len(clean_assistant) < _MIN_CAPTURE_LENGTH\n        ):\n            return\n        if _is_trivial_message(clean_user):\n            return\n\n        title = (\n            clean_user[: _MAX_TITLE_LENGTH - 3] + "..."\n            if len(clean_user) > _MAX_TITLE_LENGTH\n            else clean_user\n        )\n        content = f"User: {clean_user}\\n\\nAssistant: {clean_assistant}"\n\n        def _run():\n            try:\n                client.remember(\n                    memory_type="event",\n                    title=title,\n                    content=content,\n                    confidence=_CAPTURE_CONFIDENCE,\n                    tags=["conversation"],\n                    source="hermes",\n                    provenance="observed",\n                )\n            except Exception:\n                logger.debug("Memanto sync_turn failed", exc_info=True)\n\n        if self._sync_thread and self._sync_thread.is_alive():\n            self._sync_thread.join(timeout=2.0)\n        self._sync_thread = threading.Thread(\n            target=_run, daemon=True, name="memanto-sync"\n        )\n        self._sync_thread.start()\n\n    def on_memory_write(\n        self,\n        action: str,\n        target: str,\n        content: str,\n        metadata: dict[str, Any] | None = None,\n    ) -> None:\n        if (\n            not self._active\n            or not self._write_enabled\n            or not self._mirror_memory_writes\n            or not self._client\n        ):\n            return\n        if action != "add" or not (content or "").strip():\n            return\n        client = self._client\n        clean = content.strip()\n        title = (\n            clean[: _MAX_TITLE_LENGTH - 3] + "..."\n            if len(clean) > _MAX_TITLE_LENGTH\n            else clean\n        )\n        mem_type = "preference" if target == "user" else _detect_memory_type(clean)\n\n        def _run():\n            try:\n                client.remember(\n                    memory_type=mem_type,\n                    title=title,\n                    content=clean,\n                    confidence=0.9,\n                    tags=["hermes-memory", target],\n                    source="hermes-memory",\n                    provenance="explicit_statement",\n                )\n            except Exception:\n                logger.debug("Memanto on_memory_write failed", exc_info=True)\n\n        if self._write_thread and self._write_thread.is_alive():\n            self._write_thread.join(timeout=2.0)\n        self._write_thread = threading.Thread(\n            target=_run, daemon=True, name="memanto-memory-write"\n        )\n        self._write_thread.start()\n\n    def shutdown(self) -> None:\n        for attr_name in ("_warmup_thread", "_sync_thread", "_write_thread"):\n            thread = getattr(self, attr_name, None)\n            if thread and thread.is_alive():\n                thread.join(timeout=5.0)\n            setattr(self, attr_name, None)\n\n    # -- Tools ----------------------------------------------------------------\n\n    def get_tool_schemas(self) -> list[dict[str, Any]]:\n        return [REMEMBER_SCHEMA, RECALL_SCHEMA, ANSWER_SCHEMA]\n\n    def _tool_remember(self, args: dict) -> str:\n        content = str(args.get("content") or "").strip()\n        if not content:\n            return tool_error("content is required")\n        mem_type = str(args.get("type") or "").strip()\n        if mem_type not in _VALID_MEMORY_TYPES:\n            mem_type = _detect_memory_type(content)\n        tags = args.get("tags") or []\n        if not isinstance(tags, list):\n            tags = []\n        try:\n            confidence = float(args.get("confidence", _DEFAULT_CONFIDENCE))\n        except (TypeError, ValueError):\n            confidence = _DEFAULT_CONFIDENCE\n        confidence = max(0.0, min(1.0, confidence))\n        title = (\n            content[: _MAX_TITLE_LENGTH - 3] + "..."\n            if len(content) > _MAX_TITLE_LENGTH\n            else content\n        )\n        try:\n            result = self._client.remember(\n                memory_type=mem_type,\n                title=title,\n                content=content,\n                confidence=confidence,\n                tags=[str(t) for t in tags],\n                source="hermes-tool",\n                provenance="explicit_statement",\n            )\n            return json.dumps(\n                {\n                    "saved": True,\n                    "memory_id": result.get("memory_id"),\n                    "agent_id": self._agent_id,\n                    "type": mem_type,\n                }\n            )\n        except Exception as exc:\n            return tool_error(f"Failed to store memory: {exc}")\n\n    def _tool_recall(self, args: dict) -> str:\n        query = str(args.get("query") or "").strip()\n        if not query:\n            return tool_error("query is required")\n        try:\n            limit = max(\n                1,\n                min(\n                    100,\n                    int(\n                        args.get("limit", self._max_recall_results)\n                        or self._max_recall_results\n                    ),\n                ),\n            )\n        except (TypeError, ValueError):\n            limit = self._max_recall_results\n        type_filter = args.get("type")\n        if type_filter is not None and not isinstance(type_filter, list):\n            type_filter = [type_filter]\n        try:\n            memories = self._client.recall(\n                query,\n                limit=limit,\n                type=type_filter,\n                min_confidence=self._min_confidence,\n            )\n            formatted = []\n            for item in memories:\n                entry: dict[str, Any] = {\n                    "id": item.get("id"),\n                    "type": item.get("type"),\n                    "content": item.get("content") or item.get("title", ""),\n                }\n                score = _memory_score(item)\n                if score is not None:\n                    entry["score"] = round(score * 100)\n                formatted.append(entry)\n            return json.dumps({"results": formatted, "count": len(formatted)})\n        except Exception as exc:\n            return tool_error(f"Recall failed: {exc}")\n\n    def _tool_answer(self, args: dict) -> str:\n        question = str(args.get("question") or "").strip()\n        if not question:\n            return tool_error("question is required")\n        limit = args.get("limit")\n        if limit is not None:\n            try:\n                limit = max(1, min(100, int(limit)))\n            except (TypeError, ValueError):\n                limit = None\n        try:\n            result = self._client.answer(question, limit=limit)\n            sources = result.get("sources", []) or []\n            return json.dumps(\n                {\n                    "answer": result.get("answer", ""),\n                    "sources": sources,\n                    "count": len(sources),\n                }\n            )\n        except Exception as exc:\n            return tool_error(f"Answer failed: {exc}")\n\n    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:\n        if not self._active or not self._client:\n            return tool_error("Memanto is not configured")\n        if tool_name == "memanto_remember":\n            return self._tool_remember(args)\n        if tool_name == "memanto_recall":\n            return self._tool_recall(args)\n        if tool_name == "memanto_answer":\n            return self._tool_answer(args)\n        return tool_error(f"Unknown tool: {tool_name}")\n\n\ndef register(ctx):\n    ctx.register_memory_provider(MemantoMemoryProvider())\n'

if __name__ == "__main__":
    if sys.argv[1:] == ["_worker"]:
        worker()
    else:
        main()



