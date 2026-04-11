# Workflow Layer Spec

Implements the 4 modules under `src/poorcharlie/workflow/`. Pure orchestration logic — no LLM calls, no business analysis.

## 1. `context.py` — PipelineContext

Data bus that carries state across pipeline stages.

### State

| Field | Type | Description |
|---|---|---|
| `intake` | `CompanyIntake` | Immutable pipeline input |
| `_results` | `dict[str, BaseAgentOutput]` | agent_name -> validated output |
| `stopped` | `bool` | Whether pipeline was halted early |
| `stop_reason` | `str \| None` | Why it stopped |

### Methods

- `set_result(agent_name, output)` — store output, check `StopSignal`; if `stop_signal.should_stop` is `True`, set `stopped=True` and `stop_reason`
- `get_result(agent_name) -> BaseAgentOutput` — return stored output; raise `KeyError` if not found
- `is_stopped() -> bool` — return `self.stopped`
- `stop(reason)` — explicitly halt pipeline from outside (used by gates)
- `completed_agents() -> list[str]` — return names of agents that have run

## 2. `gates.py` — Gate Functions

Pure functions. Read from context, return `(proceed: bool, reason: str)`.

### `check_triage_gate(ctx)`
- Read `ctx.get_result("triage")` as `TriageOutput`
- If `decision == REJECT`: return `(False, "Triage rejected: {why}")`
- If `decision == WATCH`: return `(True, "Triage watch: {why}")` (proceed but note)
- If `decision == PASS`: return `(True, "")`

### `check_accounting_risk_gate(ctx)`
- Read `ctx.get_result("accounting_risk")` as `AccountingRiskOutput`
- If `risk_level == RED`: return `(False, "Accounting risk RED: {credibility_concern}")`
- Else: return `(True, "")`

### `check_financial_quality_gate(ctx)`
- Read `ctx.get_result("financial_quality")` as `FinancialQualityOutput`
- If `pass_minimum_standard == False`: return `(False, "Financial quality below minimum: {key_failures}")`
- Else: return `(True, "")`

## 3. `runner.py` — Agent Runner

### `async run_agent(agent, input_data, ctx) -> BaseAgentOutput`

1. Call `await agent.run(input_data)`
2. Validate: result must be a `BaseAgentOutput` subclass (already guaranteed by type system + pydantic)
3. Store: `ctx.set_result(agent.name, result)`
4. Return result

Runner does NOT decide whether to stop — that's the context's job (via StopSignal) or the gate's job.

## 4. `orchestrator.py` — Pipeline

### `async run_pipeline(intake) -> PipelineContext`

```
ctx = PipelineContext(intake)

# Stage 1: Triage
await run_agent(TriageAgent(), intake, ctx)
if ctx.is_stopped(): return ctx
proceed, reason = check_triage_gate(ctx)
if not proceed: ctx.stop(reason); return ctx

# Stage 2: Info Capture
await run_agent(InfoCaptureAgent(), intake, ctx)
if ctx.is_stopped(): return ctx

# Stage 3: Filing Structuring
await run_agent(FilingAgent(), <input from ctx>, ctx)
if ctx.is_stopped(): return ctx

# Stage 4: Accounting Risk
await run_agent(AccountingRiskAgent(), <input from ctx>, ctx)
if ctx.is_stopped(): return ctx
proceed, reason = check_accounting_risk_gate(ctx)
if not proceed: ctx.stop(reason); return ctx

# Stage 5: Financial Quality
await run_agent(FinancialQualityAgent(), <input from ctx>, ctx)
if ctx.is_stopped(): return ctx
proceed, reason = check_financial_quality_gate(ctx)
if not proceed: ctx.stop(reason); return ctx

# Stage 6-7: Net Cash + Valuation (sequential)
await run_agent(NetCashAgent(), <input from ctx>, ctx)
await run_agent(ValuationAgent(), <input from ctx>, ctx)
if ctx.is_stopped(): return ctx

# Stage 8: Mental Models (parallel)
await asyncio.gather(
    run_agent(MoatAgent(), ...),
    run_agent(CompoundingAgent(), ...),
    run_agent(PsychologyAgent(), ...),
    run_agent(SystemsAgent(), ...),
    run_agent(EcologyAgent(), ...),
)
if ctx.is_stopped(): return ctx

# Stage 9: Critic
await run_agent(CriticAgent(), <input from ctx>, ctx)
if ctx.is_stopped(): return ctx

# Stage 10: Committee
await run_agent(CommitteeAgent(), <input from ctx>, ctx)

return ctx
```

### Input routing note

Each agent receives `intake` as input for now. When agents are implemented, they will read what they need from `ctx` directly. The `input_data` parameter exists so that future agents can receive curated inputs rather than the full context.

## Design Decisions

- **No retry logic** — MVP keeps it simple. If an agent raises, let it propagate.
- **No timeout** — defer to caller.
- **StopSignal vs gates** — two ways to stop: an agent can self-stop via `StopSignal` in its output (checked by `set_result`), or the orchestrator explicitly stops via a gate check. Both funnel into `ctx.stop()`.
- **Mental models are fire-and-forget parallel** — if one fails, `gather` raises. No partial-failure handling in MVP.
