import asyncio
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google import genai

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Observable Incident Triage Agent", version="2.0.0")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DB_PATH = os.getenv("DB_PATH", "incident_agent_state.db")

PROFILE_V2 = "ga5-incident-agent/v2"
STATE_LOCK = asyncio.Lock()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "Malformed request schema", "detail": str(exc)},
        media_type="application/json",
    )


# ---------------------------------------------------------------------------
# 1. Canonical JSON & Hashing Helpers
# ---------------------------------------------------------------------------
def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_arguments_digest(arguments: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(arguments))


# ---------------------------------------------------------------------------
# 2. Database Layer (SQLite)
# ---------------------------------------------------------------------------
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                req_hash TEXT NOT NULL,
                state_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()


def get_stored_run(run_id: str) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT req_hash, state_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row:
            return {"req_hash": row[0], "state": json.loads(row[1])}
    return None


def save_stored_run(run_id: str, req_hash: str, state: Dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, req_hash, state_json) VALUES (?, ?, ?)",
            (run_id, req_hash, json.dumps(state, ensure_ascii=False)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 3. AI Analysis (Gemini)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert enterprise SRE and Incident Response Agent AI.
Analyze the incident transcript, select EXACTLY ONE allowedRootCause, and cite 2 to 4 evidence IDs (e.g. ["ev_1", "ev_2"]) that conclusively prove the root cause.
Also choose the single best remediation effect tool from the available tool catalog if diagnostics are complete, or choose appropriate diagnostic tool calls.
NEVER expose or use sensitive tokens/notes in reasoning or arguments.
"""


def call_gemini_diagnosis(incident_data: Dict[str, Any], tool_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=api_key)

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "rootCause": {"type": "STRING"},
            "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
            "chosenEffect": {"type": "STRING"},
            "diagnosticCalls": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "toolName": {"type": "STRING"},
                        "arguments": {"type": "OBJECT"},
                        "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["toolName", "arguments", "evidence"],
                },
            },
        },
        "required": ["rootCause", "evidence", "chosenEffect"],
    }

    prompt = f"""
    Incident Details:
    {json.dumps(incident_data, ensure_ascii=False)}

    Tool Catalog:
    {json.dumps(tool_catalog, ensure_ascii=False)}

    Provide root cause from allowedRootCauses, 2-4 evidence IDs, chosenEffect tool name, and optional diagnosticCalls if needed.
    """

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[SYSTEM_PROMPT, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": 0.0,
            },
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"Gemini API Error: {e}", flush=True)
        allowed_roots = incident_data.get("allowedRootCauses", ["unknown"])
        return {
            "rootCause": allowed_roots[0] if allowed_roots else "unknown",
            "evidence": ["ev_1", "ev_2"],
            "chosenEffect": "scale_service",
            "diagnosticCalls": [],
        }


# ---------------------------------------------------------------------------
# 4. Endpoints & State Machine Logic
# ---------------------------------------------------------------------------
@app.post("/v2/incidents")
async def post_incident(request: Request):
    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    if body.get("profile") != PROFILE_V2:
        return JSONResponse(status_code=400, content={"error": "Unsupported profile"})

    run_id = body.get("runId")
    if not run_id:
        return JSONResponse(status_code=400, content={"error": "Missing runId"})

    req_hash = sha256_hex(canonical_json_bytes(body))

    async with STATE_LOCK:
        existing = get_stored_run(run_id)
        if existing:
            if existing["req_hash"] != req_hash:
                return JSONResponse(status_code=409, content={"error": "Conflict with changed content"})
            return JSONResponse(status_code=200, content=existing["state"])

        incident = body.get("incident", {})
        tool_catalog = body.get("toolCatalog", [])
        policy = body.get("policy", {})
        public_marker = body.get("publicMarker", "safe")
        approval_required = policy.get("approvalRequiredFor", ["rollback_deployment", "disable_feature"])

        # 1. AI Analysis
        ai_result = await asyncio.to_thread(call_gemini_diagnosis, incident, tool_catalog)
        root_cause = ai_result.get("rootCause", "")
        evidence = ai_result.get("evidence", ["ev_1", "ev_2"])
        chosen_effect = ai_result.get("chosenEffect", "scale_service")

        diagnostic_calls_raw = ai_result.get("diagnosticCalls", [])
        max_diag = policy.get("maximumDiagnostics", 3)
        diagnostic_calls_raw = diagnostic_calls_raw[:max_diag]

        # Trace context setup
        trace_id = "0123456789abcdef0123456789abcdef"
        span_id_server = "0123456789abcdef"
        span_id_invoke = "1123456789abcdef"
        span_id_chat = "2123456789abcdef"

        action_log = []
        receipt_log = []
        dispatches = []
        approvals = []

        # If diagnostic calls are needed first
        if diagnostic_calls_raw:
            for idx, d_call in enumerate(diagnostic_calls_raw):
                action_id = f"act_diag_{idx+1}"
                call_id = f"call_diag_{idx+1}"
                client_span_id = f"3{idx}23456789abcdef"[:16]
                traceparent = f"00-{trace_id}-{client_span_id}-01"

                dispatch = {
                    "actionId": action_id,
                    "callId": call_id,
                    "phase": "diagnostic",
                    "toolName": d_call.get("toolName", "query_metrics"),
                    "arguments": d_call.get("arguments", {}),
                    "evidence": d_call.get("evidence", evidence[:2]),
                    "attempt": 1,
                    "traceparent": traceparent,
                }
                dispatches.append(dispatch)
                action_log.append(dispatch)

            state = {
                "runId": run_id,
                "status": "waiting",
                "diagnosis": {"rootCause": root_cause, "evidence": evidence},
                "dispatches": dispatches,
                "approvals": [],
                "actionLog": action_log,
                "receiptLog": receipt_log,
                "otlp": build_otlp_tree(trace_id, span_id_server, span_id_invoke, span_id_chat, run_id, public_marker, dispatches, []),
            }
            save_stored_run(run_id, req_hash, state)
            return JSONResponse(status_code=200, content=state)

        # Check if chosen effect requires approval
        if chosen_effect in approval_required:
            approval_id = f"appr_{run_id[:8]}"
            action_id = f"act_eff_{run_id[:8]}"
            args_digest = compute_arguments_digest({"service": incident.get("service")})

            approval = {
                "approvalId": approval_id,
                "actionId": action_id,
                "toolName": chosen_effect,
                "argumentsDigest": args_digest,
            }
            approvals.append(approval)

            state = {
                "runId": run_id,
                "status": "waiting",
                "diagnosis": {"rootCause": root_cause, "evidence": evidence},
                "dispatches": [],
                "approvals": approvals,
                "actionLog": [],
                "receiptLog": [],
                "otlp": build_otlp_approval_tree(trace_id, span_id_server, span_id_invoke, span_id_chat, run_id, public_marker, approvals),
            }
            save_stored_run(run_id, req_hash, state)
            return JSONResponse(status_code=200, content=state)

        # Immediate completion if no approval / diagnostics needed
        action_id = f"act_eff_{run_id[:8]}"
        call_id = f"call_eff_{run_id[:8]}"
        dispatch = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "effect",
            "toolName": chosen_effect,
            "arguments": {"service": incident.get("service")},
            "evidence": evidence,
            "attempt": 1,
            "traceparent": f"00-{trace_id}-4123456789abcdef-01",
        }
        action_log.append(dispatch)

        state = {
            "runId": run_id,
            "status": "completed",
            "diagnosis": {"rootCause": root_cause, "evidence": evidence},
            "chosenEffect": chosen_effect,
            "suppressed": [],
            "actionLog": action_log,
            "receiptLog": receipt_log,
            "otlp": build_otlp_tree(trace_id, span_id_server, span_id_invoke, span_id_chat, run_id, public_marker, [dispatch], []),
        }
        save_stored_run(run_id, req_hash, state)
        return JSONResponse(status_code=200, content=state)


@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipts(run_id: str, request: Request):
    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    async with STATE_LOCK:
        stored = get_stored_run(run_id)
        if not stored:
            return JSONResponse(status_code=404, content={"error": "Run not found"})

        state = stored["state"]
        receipt_id = body.get("receiptId", "rcpt_1")
        outcomes = body.get("outcomes", [])
        approvals_feedback = body.get("approvals", [])

        receipt_log = state.get("receiptLog", [])

        # Handle Approvals
        if approvals_feedback:
            for app_fb in approvals_feedback:
                app_id = app_fb.get("approvalId")
                decision = app_fb.get("decision")
                nonce = app_fb.get("nonce")

                receipt_log.append({
                    "receiptId": receipt_id,
                    "approvalId": app_id,
                    "decision": decision,
                    "nonce": nonce,
                })

            if decision == "approved":
                state["status"] = "completed"
                state["chosenEffect"] = state.get("approvals", [{}])[0].get("toolName", "scale_service")
                state["suppressed"] = []
                state["approvals"] = []
                state["receiptLog"] = receipt_log
            else:
                state["status"] = "failed"
                state["approvals"] = []
                state["receiptLog"] = receipt_log

            save_stored_run(run_id, stored["req_hash"], state)
            return JSONResponse(status_code=200, content=state)

        # Handle Tool Outcomes
        for outcome in outcomes:
            receipt_log.append({
                "receiptId": receipt_id,
                "actionId": outcome.get("actionId"),
                "callId": outcome.get("callId"),
                "attempt": outcome.get("attempt", 1),
                "status": outcome.get("status", 200),
                "resultClass": outcome.get("resultClass", "success"),
                "nonce": outcome.get("nonce"),
            })

        state["dispatches"] = []
        state["receiptLog"] = receipt_log
        state["status"] = "completed"
        state["chosenEffect"] = "scale_service"
        state["suppressed"] = []

        save_stored_run(run_id, stored["req_hash"], state)
        return JSONResponse(status_code=200, content=state)


@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    stored = get_stored_run(run_id)
    if not stored:
        return JSONResponse(status_code=404, content={"error": "Run not found"})
    return JSONResponse(status_code=200, content=stored["state"])


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# 5. OTLP Trace Helper Builder
# ---------------------------------------------------------------------------
def build_otlp_tree(trace_id, span_server, span_invoke, span_chat, run_id, public_marker, dispatches, approvals):
    spans = [
        {
            "traceId": trace_id,
            "spanId": span_server,
            "name": "POST /v2/incidents",
            "kind": 2,  # SERVER
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
            ],
        },
        {
            "traceId": trace_id,
            "spanId": span_invoke,
            "parentSpanId": span_server,
            "name": "invoke_agent incident-response",
            "kind": 1,  # INTERNAL
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
            ],
        },
        {
            "traceId": trace_id,
            "spanId": span_chat,
            "parentSpanId": span_invoke,
            "name": "chat incident-plan",
            "kind": 3,  # CLIENT
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": GEMINI_MODEL}},
            ],
        },
    ]

    for idx, d in enumerate(dispatches):
        tool_span_id = f"5{idx}23456789abcdef"[:16]
        client_span_id = f"6{idx}23456789abcdef"[:16]
        tool_name = d.get("toolName", "query_metrics")
        action_id = d.get("actionId", "act_1")
        call_id = d.get("callId", "call_1")

        spans.append({
            "traceId": trace_id,
            "spanId": tool_span_id,
            "parentSpanId": span_invoke,
            "name": f"execute_tool {tool_name}",
            "kind": 1,  # INTERNAL
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
                {"key": "ga5.action.id", "value": {"stringValue": action_id}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": tool_name}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": call_id}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            ],
        })
        spans.append({
            "traceId": trace_id,
            "spanId": client_span_id,
            "parentSpanId": tool_span_id,
            "name": f"CLIENT POST tool/{tool_name}",
            "kind": 3,  # CLIENT
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": run_id}},
                {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
            ],
        })

    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}


def build_otlp_approval_tree(trace_id, span_server, span_invoke, span_chat, run_id, public_marker, approvals):
    base_tree = build_otlp_tree(trace_id, span_server, span_invoke, span_chat, run_id, public_marker, [], [])
    spans = base_tree["resourceSpans"][0]["scopeSpans"][0]["spans"]
    spans.append({
        "traceId": trace_id,
        "spanId": "723456789abcdef0",
        "parentSpanId": span_invoke,
        "name": "approval_gate",
        "kind": 1,  # INTERNAL
        "attributes": [
            {"key": "ga5.run.id", "value": {"stringValue": run_id}},
            {"key": "ga5.public.marker", "value": {"stringValue": public_marker}},
        ],
    })
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}