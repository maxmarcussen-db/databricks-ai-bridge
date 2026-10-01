"""`agentbricks models` — bind the agent to a swappable model service, and upgrade its model.

The agent calls a user-owned Unity Catalog AI Gateway *model service* (``catalog.schema.name``)
declared in agent.toml's ``[model_service]`` table, instead of a hardcoded ``system.ai.*`` model.
`agentbricks deploy` creates the service (routed to the binding's default model) and grants the app
access; from then on the model behind it is repointed here, with no code change or redeploy:

- ``models upgrade`` uploads the project to the workspace and submits a serverless Databricks job
  that evaluates candidate models on the agent's own recent traces. The search never runs on this
  machine (it can take hours);
- ``models status`` shows the latest run and, once it finishes, its recommendation;
- ``models apply`` switches the service to that recommendation (quality / latency / cost
  weighted); ``models set`` switches it to a named model; ``models rollback`` undoes a switch.

Runs and switches are recorded in ``.agentbricks/model_upgrades.json``.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit.runtime.model_services import destination_model

_BIND_COMMAND = "agentbricks models bind <catalog>.<schema>.<name> --default system.ai.<model>"


def _source_option(function):
    return click.option(
        "--source",
        type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
        default=pathlib.Path("."),
        show_default=True,
        help="Agent Bricks project containing agent.toml.",
    )(function)


def _yes_option(function):
    return click.option(
        "--yes", "-y", is_flag=True, help="Switch the model without asking for confirmation."
    )(function)


def _bound_project(source: pathlib.Path):
    """The project at ``source``; errors when it has no model-service binding."""
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    if not project.model_service:
        raise AgentCliError(
            "This agent has no model service bound.",
            hint=f"Run `{_BIND_COMMAND}`, then `agentbricks deploy`.",
        )
    return project


def _current_model(client, service: str) -> str:
    try:
        resolved = client.get_model_service(service)
    except AgentCliError as exc:
        if exc.error_code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
            raise AgentCliError(
                f"Model service '{service}' doesn't exist yet.",
                hint="Run `agentbricks deploy` to create it from agent.toml.",
            ) from exc
        raise
    model = destination_model(resolved)
    if model is None:
        raise AgentCliError(f"Model service '{service}' has no foundation-model destination.")
    return model


def _switch(obj, project, service: str, previous: str, model: str, reason: str, report=None):
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    with render.status(f"Switching '{service}' to {model}…"):
        obj.client().set_model_service_model(service, model)
    model_upgrade.record_change(
        project.root,
        service=service,
        previous=previous,
        model=model,
        reason=reason,
        report=report,
    )


def _confirm_switch(obj, yes: bool, service: str, previous: str, model: str) -> bool:
    """True to go ahead. JSON output never prompts: it switches only with --yes."""
    if yes:
        return True
    if obj.output == "json":
        return False
    return click.confirm(f"Switch '{service}' from {previous} to {model}?", default=False)


@click.group()
def models() -> None:
    """Choose, evaluate, and upgrade the model behind your agent.

    Bind the agent to a model service it calls through the AI Gateway, then switch the model behind
    that service at any time — by name, or by letting `models upgrade` pick the best candidate on
    the agent's own production traces. Switches take effect without a code change or redeploy.
    """


@models.command("bind")
@click.argument("service")
@click.option(
    "--default",
    "default_model",
    default=None,
    help="system.ai.* model `agentbricks deploy` routes the service to when it creates it.",
)
@_source_option
@click.pass_obj
def models_bind(obj, service: str, default_model: Optional[str], source: pathlib.Path) -> None:
    """Bind model service SERVICE (catalog.schema.name) to the agent in agent.toml.

    This only edits agent.toml — it does not create the service. `agentbricks deploy` creates it if it
    doesn't exist (routed to --default), grants the app's service principal EXECUTE on it, and points
    the agent at it via AGENT_MODEL_SERVICE.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415
    from databricks_agentbricks.model_upgrade import system_ai_name  # noqa: PLC0415

    project = AgentProject.load(source)
    project.bind_model_service(service, system_ai_name(default_model) if default_model else None)
    project.write()
    if obj.output == "json":
        render.emit_json(
            {
                "model_service": service,
                "default": project.model_service_default,
                "manifest": str(project.path),
            }
        )
        return
    fields = {"agent.toml": str(project.path)}
    if project.model_service_default:
        fields["Default model"] = str(project.model_service_default)
    render.success(
        f"Bound model service '{service}'",
        fields=fields,
        next_steps=[
            ("agentbricks deploy <name>", "Create it if missing and grant the app access"),
            ("agentbricks models upgrade --candidates <model>", "Then evaluate other models"),
        ],
    )


@models.command("unbind")
@_source_option
@click.pass_obj
def models_unbind(obj, source: pathlib.Path) -> None:
    """Remove the model-service binding from agent.toml.

    The service itself is left in place. After the next deploy the agent calls its template default
    model directly again.
    """
    from databricks_agentbricks.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(source)
    removed = project.unbind_model_service()
    project.write()
    if obj.output == "json":
        render.emit_json({"model_service": None, "removed": removed})
        return
    render.success("Model service unbound" if removed else "No model service was bound")


@models.command("list")
@click.pass_obj
def models_list(obj) -> None:
    """List the system.ai.* chat models you can route the agent to."""
    names = obj.client().list_chat_model_services()
    if obj.output == "json":
        render.emit_json(names)
        return
    render.resource_table("AI Gateway Models · system.ai", [("Model", "left")], [[n] for n in names])


@models.command("status")
@_source_option
@click.pass_obj
def models_status(obj, source: pathlib.Path) -> None:
    """Show the bound model service, the model behind it now, and the latest upgrade run."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    service = str(project.model_service)
    current = _current_model(obj.client(), service)
    history = model_upgrade.read_history(project.root)
    last_change = history["changes"][-1] if history["changes"] else None
    run = _refresh_latest_run(obj, project) if history["runs"] else None
    if obj.output == "json":
        render.emit_json(
            {"model_service": service, "model": current, "last_change": last_change, "latest_run": run}
        )
        return
    fields = {"Model service": service, "Current model": current}
    if last_change:
        fields["Last switch"] = (
            f"{last_change.get('previous_model')} → {last_change.get('model')} "
            f"({last_change.get('reason')})"
        )
    if run:
        fields["Latest upgrade run"] = f"{run.get('state')} · {run.get('run_page_url') or run.get('run_id')}"
    render.success("Model status", fields=fields)
    report = run.get("report") if run else None
    if report:
        _render_report(model_upgrade.UpgradeReport.from_json(report), current)


@models.command("set")
@click.argument("model")
@_yes_option
@_source_option
@click.pass_obj
def models_set(obj, model: str, yes: bool, source: pathlib.Path) -> None:
    """Switch the bound model service to MODEL (a system.ai.* name)."""
    from databricks_agentbricks.model_upgrade import system_ai_name  # noqa: PLC0415

    project = _bound_project(source)
    service = str(project.model_service)
    model = system_ai_name(model)
    previous = _current_model(obj.client(), service)
    if previous == model:
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": model, "changed": False})
            return
        render.success(f"'{service}' already routes to {model}")
        return
    if not _confirm_switch(obj, yes, service, previous, model):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": previous, "changed": False})
            return
        raise click.Abort()
    _switch(obj, project, service, previous, model, reason="set")
    if obj.output == "json":
        render.emit_json(
            {"model_service": service, "previous_model": previous, "model": model, "changed": True}
        )
        return
    render.success(
        f"Switched '{service}' to {model}",
        fields={"Previous model": previous},
        next_steps=[("agentbricks models rollback", "Switch back")],
    )


@models.command("rollback")
@_yes_option
@_source_option
@click.pass_obj
def models_rollback(obj, yes: bool, source: pathlib.Path) -> None:
    """Switch the model service back to the model it used before its last switch."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    service = str(project.model_service)
    history = [
        h for h in model_upgrade.read_history(project.root)["changes"] if h.get("model_service") == service
    ]
    previous = history[-1].get("previous_model") if history else None
    if not previous:
        raise AgentCliError(f"No recorded switch of '{service}' to roll back.")
    current = _current_model(obj.client(), service)
    if not _confirm_switch(obj, yes, service, current, previous):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": current, "changed": False})
            return
        raise click.Abort()
    _switch(obj, project, service, current, previous, reason="rollback")
    if obj.output == "json":
        render.emit_json(
            {"model_service": service, "previous_model": current, "model": previous, "changed": True}
        )
        return
    render.success(f"Rolled '{service}' back to {previous}", fields={"Previous model": current})


def _parse_weights(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 3 or any(p < 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise click.BadParameter("use three non-negative numbers summing to 1, e.g. 0.7,0.2,0.1")
    return parts  # ty: ignore[invalid-return-type]


_TERMINAL_STATES = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}


def _state_name(value) -> Optional[str]:
    return getattr(value, "value", None) or (str(value) if value is not None else None)


def _run_state(run) -> tuple[Optional[str], Optional[str]]:
    """``(life_cycle_state, result_state)`` of a job run, as plain strings."""
    state = getattr(run, "state", None)
    return (
        _state_name(getattr(state, "life_cycle_state", None)),
        _state_name(getattr(state, "result_state", None)),
    )


def _refresh_latest_run(obj, project) -> Optional[dict]:
    """The latest recorded upgrade run, with its job state refreshed and its report fetched once done."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    runs = model_upgrade.read_history(project.root)["runs"]
    if not runs:
        return None
    entry = dict(runs[-1])
    if entry.get("report") or entry.get("state") in {"FAILED", "CANCELED", "TIMEDOUT"}:
        return entry
    life_cycle, result = _run_state(obj.client().get_run(int(entry["run_id"])))
    entry["state"] = result or life_cycle
    if life_cycle in _TERMINAL_STATES and result == "SUCCESS":
        report = model_upgrade.fetch_report(
            obj.profile, str(project.trace_experiment_name), entry["upgrade_id"]
        )
        if report is not None:
            entry["report"] = report.to_json()
    model_upgrade.update_run(
        project.root, entry["upgrade_id"], state=entry["state"], report=entry.get("report")
    )
    return entry


def _render_report(report, current: str) -> None:
    if report.model_scores:
        rows = [
            [model, f"{score:.3f}", "current" if model == current else ""]
            for model, score in sorted(report.model_scores.items(), key=lambda kv: -kv[1])
        ]
        render.resource_table(
            f"Candidate models · {report.val_records} eval records",
            [("Model", "left"), ("Best score", "right"), ("", "left")],
            rows,
        )
    render.success(
        "Recommendation",
        fields={
            "Recommended": report.recommended_model,
            "Score": f"{report.baseline_score:.3f} → {report.best_score:.3f}",
        },
        next_steps=[("agentbricks models apply", "Switch to it")] if report.changed else None,
    )


def _parse_weights(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(p) for p in value.split(","))
    except ValueError:
        parts = ()
    if len(parts) != 3 or any(p < 0 for p in parts) or abs(sum(parts) - 1.0) > 1e-6:
        raise click.BadParameter("use three non-negative numbers summing to 1, e.g. 0.7,0.2,0.1")
    return parts  # ty: ignore[invalid-return-type]


@models.command("upgrade")
@click.option(
    "--candidates",
    "-c",
    "candidates",
    multiple=True,
    required=True,
    help="system.ai.* model to evaluate (repeat, or comma-separate). The current model is always "
    "included as the baseline. See `agentbricks models list`.",
)
@click.option(
    "--traces",
    "trace_limit",
    type=click.IntRange(min=5),
    default=50,
    show_default=True,
    help="Recent traces to build the eval set from.",
)
@click.option(
    "--budget",
    type=click.IntRange(min=1),
    default=None,
    help="Evaluation budget (agent runs). Default: 4 x the number of eval records.",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model serving endpoint for the equivalence judge (default: databricks-claude-sonnet-4-6).",
)
@click.option(
    "--weights",
    default="0.7,0.2,0.1",
    show_default=True,
    help="Quality, latency, cost weights for picking the winner.",
)
@click.option(
    "--timeout-hours",
    type=click.FloatRange(min=0.1),
    default=6.0,
    show_default=True,
    help="Cancel the job if it runs longer than this.",
)
@click.option("--wait", is_flag=True, help="Wait for the job to finish and show its recommendation.")
@_source_option
@click.pass_obj
def models_upgrade(
    obj,
    candidates: tuple[str, ...],
    trace_limit: int,
    budget: Optional[int],
    judge_model: Optional[str],
    weights: str,
    timeout_hours: float,
    wait: bool,
    source: pathlib.Path,
) -> None:
    """Evaluate candidate models on the agent's own traces, as a Databricks job.

    Uploads the project to your workspace and submits a serverless one-time run. The job replays
    the agent's recent production requests through your agent code once per candidate, against a
    temporary copy of the model service (production traffic is untouched), and scores each answer
    against what production returned with an LLM judge. The winner balances quality, latency, and
    cost (--weights); at equal quality the cheaper model wins.

    The search runs only on Databricks, never on this machine, and can take hours: this command
    returns once the job is submitted. Check on it with `agentbricks models status`, then switch to
    its recommendation with `agentbricks models apply`. Nothing changes until you apply.
    """
    import uuid  # noqa: PLC0415

    from databricks_agentbricks import model_upgrade  # noqa: PLC0415
    from databricks_agentbricks.databricks_cli import _databricks  # noqa: PLC0415

    parsed_weights = _parse_weights(weights)
    project = _bound_project(source)
    if not project.trace_experiment_name:
        raise AgentCliError(
            "models upgrade builds its eval set from the agent's traces, but tracing isn't bound.",
            hint="Run `agentbricks tracing bind --experiment-name <path>` and deploy first.",
        )
    service = str(project.model_service)
    names = [
        model_upgrade.system_ai_name(name.strip())
        for value in candidates
        for name in value.split(",")
        if name.strip()
    ]
    client = obj.client()
    current = _current_model(client, service)  # fails fast if the service isn't deployed yet
    # A model service can only route to models its owner can execute; catch that now rather than
    # an hour into the job.
    denied = [name for name in names if client.can_execute_model(name) is False]
    if denied:
        raise AgentCliError(
            f"You can't route the model service to: {', '.join(denied)}.",
            hint="A model service can only route to models its owner holds EXECUTE on. Ask a "
            "workspace admin to grant EXECUTE on those system.ai models, or pick others "
            "(`agentbricks models list`).",
        )
    config = model_upgrade.JobConfig(
        upgrade_id=uuid.uuid4().hex[:12],
        service=service,
        framework=str(project.framework),
        candidates=names,
        trace_experiment=str(project.trace_experiment_name),
        trace_limit=trace_limit,
        budget=budget,
        judge_model=judge_model or model_upgrade.DEFAULT_JUDGE_MODEL,
        weights=parsed_weights,
    )
    ws_path = model_upgrade.workspace_project_path(
        client.current_user, str(project.deployment_name or project.root.name)
    )
    with render.status(f"Uploading the project to {ws_path}…"):
        _databricks(
            ["sync", str(project.root), ws_path, "--exclude", "uv.lock", "--exclude", ".venv"],
            obj.profile,
            capture=True,
            action="Could not upload the project for the upgrade job.",
        )
    with render.status("Submitting the upgrade job…"):
        run_id, run_url = model_upgrade.submit_upgrade_run(
            client, workspace_path=ws_path, config=config, timeout_hours=timeout_hours
        )
    model_upgrade.record_run(
        project.root,
        {
            "upgrade_id": config.upgrade_id,
            "run_id": run_id,
            "run_page_url": run_url,
            "model_service": service,
            "current_model": current,
            "candidates": names,
            "state": "PENDING",
        },
    )

    run = None
    if wait:
        import time  # noqa: PLC0415

        with render.progress("Waiting for the upgrade job (this can take hours; Ctrl-C detaches)…"):
            while True:
                life_cycle, _ = _run_state(client.get_run(run_id))
                if life_cycle in _TERMINAL_STATES:
                    break
                time.sleep(30)
        run = _refresh_latest_run(obj, project)

    if obj.output == "json":
        render.emit_json(
            {"upgrade_id": config.upgrade_id, "run_id": run_id, "run_page_url": run_url, "run": run}
        )
        return
    if run is None:
        render.success(
            "Submitted the model upgrade job",
            fields={"Run": run_url or str(run_id), "Candidates": ", ".join(names)},
            next_steps=[
                ("agentbricks models status", "Check on it and see its recommendation"),
                ("agentbricks models apply", "Switch to the recommendation once it's done"),
            ],
        )
        return
    if run.get("report"):
        _render_report(model_upgrade.UpgradeReport.from_json(run["report"]), current)
    else:
        raise AgentCliError(
            f"The upgrade job finished as {run.get('state')} without a recommendation.",
            hint=f"See the run: {run_url or run_id}",
        )


@models.command("apply")
@_yes_option
@_source_option
@click.pass_obj
def models_apply(obj, yes: bool, source: pathlib.Path) -> None:
    """Switch the model service to the latest finished upgrade run's recommendation."""
    from databricks_agentbricks import model_upgrade  # noqa: PLC0415

    project = _bound_project(source)
    service = str(project.model_service)
    run = _refresh_latest_run(obj, project)
    if run is None:
        raise AgentCliError(
            "No upgrade run to apply.", hint="Run `agentbricks models upgrade -c <model>` first."
        )
    if not run.get("report"):
        raise AgentCliError(
            f"The latest upgrade run is {run.get('state')}; there's no recommendation to apply yet.",
            hint="Check on it with `agentbricks models status`.",
        )
    report = model_upgrade.UpgradeReport.from_json(run["report"])
    current = _current_model(obj.client(), service)
    if report.recommended_model == current:
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": current, "changed": False})
            return
        render.success(f"'{service}' is already on the recommended model ({current})")
        return
    if not _confirm_switch(obj, yes, service, current, report.recommended_model):
        if obj.output == "json":
            render.emit_json({"model_service": service, "model": current, "changed": False})
            return
        raise click.Abort()
    _switch(obj, project, service, current, report.recommended_model, reason="upgrade", report=report)
    if obj.output == "json":
        render.emit_json(
            {
                "model_service": service,
                "previous_model": current,
                "model": report.recommended_model,
                "changed": True,
            }
        )
        return
    render.success(
        f"Upgraded '{service}' to {report.recommended_model}",
        fields={
            "Previous model": current,
            "Score": f"{report.baseline_score:.3f} → {report.best_score:.3f}",
        },
        next_steps=[("agentbricks models rollback", "Switch back")],
    )
