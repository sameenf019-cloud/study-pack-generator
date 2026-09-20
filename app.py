"""
AI Study Pack Generator - multi-stage AI workflow
=================================================

Instead of one giant prompt, the pack is built by five cooperating stages that
share a single context object:

  1. PLANNING            learner profile + source  -> objectives, modules, schedule
  2. CONTENT GENERATION  one call per module        -> lesson, terms, example, flashcards
  3. ASSESSMENT          lessons + objectives       -> quiz mapped to objectives
  4. REVIEW              automatic checks + AI judge-> scores + targeted issues
  5. REFINEMENT          only the flagged parts     -> rewritten, then re-reviewed

Stack: Python, Streamlit (UI + deployment), Groq API (LLM), pypdf (PDF text).
"""

import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import groq
import streamlit as st
from groq import Groq
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_ATTEMPTS = 3            # per LLM call (retry with validation feedback)
MAX_SOURCE_CHARS = 14000    # only the planner sees the raw source
SCORE_KEYS = ["accuracy", "coverage", "clarity", "alignment", "personalization"]
ICONS = {"running": "⏳", "ok": "✅", "retried": "🔁", "degraded": "⚠️",
         "failed": "❌", "skipped": "⏭️"}

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
  Input [label="Learner profile\\n+ topic / notes / file"];
  Plan [label="1 Planner"]; Content [label="2 Content\\n(per module)"];
  Assess [label="3 Assessment"]; Review [label="4 Reviewer"]; Refine [label="5 Refiner"];
  Final [label="Study pack", fillcolor="#dcfce7"];
  Input -> Plan -> Content -> Assess -> Review;
  Review -> Refine [label="issues"]; Refine -> Review [label="re-check"];
  Review -> Final [label="pass"];
}
"""


class FatalError(Exception):
    """Unrecoverable problem (bad API key, nothing could be generated)."""


class StageError(Exception):
    """A single LLM step failed after all retries."""


# ----------------------------------------------------------------------------
# Shared workflow context (this is how stages pass information along)
# ----------------------------------------------------------------------------
@dataclass
class Ctx:
    topic: str
    source: str
    profile: dict
    settings: dict
    plan: dict = field(default_factory=dict)          # written by Planner
    modules: dict = field(default_factory=dict)       # module_id -> content
    module_memory: list = field(default_factory=list)  # running summaries -> later modules
    quiz: list = field(default_factory=list)          # written by Assessment
    reviews: list = field(default_factory=list)       # one entry per review round
    trace: list = field(default_factory=list)         # finished steps, for the UI
    warnings: list = field(default_factory=list)
    emit: Optional[Callable] = None

    def log(self, stage, status, message, seconds=None, attempts=None, progress=None):
        entry = {"stage": stage, "status": status, "message": message,
                 "seconds": round(seconds, 1) if seconds is not None else None,
                 "attempts": attempts, "progress": progress}
        if status != "running":
            self.trace.append(entry)
        if self.emit:
            self.emit(entry)


# ----------------------------------------------------------------------------
# LLM wrapper: retries, validation feedback loop, model fallback
# ----------------------------------------------------------------------------
def parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def safe_validate(validator, data):
    if validator is None:
        return []
    try:
        return validator(data)
    except Exception:  # a validator must never crash the pipeline
        return ["output did not match the expected structure"]


class LLM:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key, timeout=60.0, max_retries=0)
        self.models = [PRIMARY_MODEL, FALLBACK_MODEL]
        self.idx = 0
        self.calls = 0

    def json_call(self, system, user, validator=None, temperature=0.4, label="call"):
        """Return (data, attempts). Raises StageError / FatalError."""
        feedback, problem = "", "unknown error"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = user
            if feedback:  # send validation errors back so the model can self-correct
                prompt += (f"\n\nYour previous answer was rejected: {feedback}\n"
                           "Return the complete corrected JSON object.")
            # last attempt goes to the fallback model (separate rate limits)
            model = self.models[self.idx] if attempt < MAX_ATTEMPTS else self.models[-1]
            try:
                self.calls += 1
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                data = parse_json(resp.choices[0].message.content)
                errors = safe_validate(validator, data)
                if not errors:
                    return data, attempt
                feedback = "; ".join(errors[:6])
                problem = f"validation failed: {feedback}"
            except groq.AuthenticationError as exc:
                raise FatalError("The Groq API key was rejected. Check it and try again.") from exc
            except groq.RateLimitError as exc:
                feedback, problem = "", "rate limit reached"
                wait = 5 * attempt
                try:
                    wait = min(float(exc.response.headers.get("retry-after", wait)), 30)
                except Exception:
                    pass
                time.sleep(wait)
            except (groq.APIConnectionError, groq.APITimeoutError):
                feedback, problem = "", "network problem"
                time.sleep(attempt)
            except groq.APIStatusError as exc:
                feedback, problem = "", f"API error {exc.status_code}"
                if exc.status_code in (400, 404) and self.idx < len(self.models) - 1:
                    self.idx += 1          # model unavailable -> switch permanently
                elif exc.status_code >= 500:
                    time.sleep(attempt)
                else:
                    break
            except (json.JSONDecodeError, ValueError) as exc:
                feedback = f"the output was not a valid JSON object ({exc})"
                problem = feedback
        raise StageError(f"{label}: {problem}")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def profile_text(p: dict) -> str:
    return (f"Level: {p['level']}\nGoal: {p['goal'] or 'general understanding'}\n"
            f"Days available: {p['days']}\nMinutes per day: {p['minutes']}\n"
            f"Preferred style: {', '.join(p['styles']) or 'balanced'}\n"
            f"Weak areas / focus: {p['focus'] or 'none specified'}")


def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return ""


def active_modules(ctx):
    return [m for m in ctx.plan["modules"] if m["id"] in ctx.modules]


def active_objectives(ctx):
    ids = {i for m in active_modules(ctx) for i in m["objective_ids"]}
    return [o for o in ctx.plan["learning_objectives"] if o["id"] in ids]


def lessons_digest(ctx, per_module=1600, with_notes=False):
    parts = []
    for m in active_modules(ctx):
        c = ctx.modules[m["id"]]
        block = (f"### {m['id']} - {m['title']} (objectives: {', '.join(m['objective_ids'])})\n"
                 f"{c['lesson'][:per_module]}\n"
                 f"Terms: {', '.join(t['term'] for t in c['key_terms'])}")
        if with_notes and m["source_notes"]:
            block += f"\nSource notes: {m['source_notes'][:700]}"
        parts.append(block)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------
# STAGE 1: PLANNING
# ----------------------------------------------------------------------------
def plan_errors(d, max_modules):
    errs = []
    objs, mods, sched = d.get("learning_objectives"), d.get("modules"), d.get("schedule")
    if not isinstance(d.get("title"), str) or not d["title"].strip():
        errs.append("title is missing")
    if not isinstance(objs, list) or not objs:
        errs.append("learning_objectives must be a non-empty list")
    if not isinstance(mods, list) or not 1 <= len(mods) <= max_modules:
        errs.append(f"modules must be a list of 1-{max_modules} items")
    if not isinstance(sched, list) or not sched:
        errs.append("schedule must be a non-empty list")
    if errs:
        return errs

    obj_ids = set()
    for o in objs:
        if not isinstance(o, dict) or not o.get("id") or not o.get("text"):
            return ["every objective needs an id and text"]
        obj_ids.add(str(o["id"]))
    mod_ids, covered = set(), set()
    for m in mods:
        if not isinstance(m, dict) or not m.get("id") or not m.get("title"):
            return ["every module needs an id and title"]
        mod_ids.add(str(m["id"]))
        ids = m.get("objective_ids")
        if not isinstance(ids, list) or not ids:
            errs.append(f"module {m['id']} needs objective_ids")
            continue
        unknown = [i for i in ids if str(i) not in obj_ids]
        if unknown:
            errs.append(f"module {m['id']} uses unknown objective ids {unknown}")
        covered.update(str(i) for i in ids)
    if len(mod_ids) != len(mods):
        errs.append("module ids must be unique")
    if obj_ids - covered:
        errs.append(f"objectives not covered by any module: {sorted(obj_ids - covered)}")
    for day in sched:
        mids = day.get("module_ids") if isinstance(day, dict) else None
        if not isinstance(mids, list) or any(str(x) not in mod_ids for x in mids):
            errs.append("every schedule day needs module_ids that exist in modules")
            break
    return errs


def normalise_plan(plan):
    def strs(v):
        return [str(x) for x in v if x] if isinstance(v, list) else []
    return {
        "title": str(plan.get("title", "Study Pack")).strip(),
        "audience_summary": str(plan.get("audience_summary", "")).strip(),
        "learning_objectives": [{"id": str(o["id"]), "text": str(o["text"])}
                                for o in plan["learning_objectives"]],
        "modules": [{"id": str(m["id"]), "title": str(m["title"]),
                     "objective_ids": strs(m.get("objective_ids")),
                     "key_points": strs(m.get("key_points")),
                     "source_notes": str(m.get("source_notes", "")),
                     "depth": str(m.get("depth", "core"))} for m in plan["modules"]],
        "schedule": [{"day": str(d.get("day", f"Day {i}")), "focus": str(d.get("focus", "")),
                      "module_ids": strs(d.get("module_ids")), "tasks": strs(d.get("tasks")),
                      "minutes": d["minutes"] if isinstance(d.get("minutes"), (int, float)) else None}
                     for i, d in enumerate(plan["schedule"], 1)],
    }


def fallback_plan(ctx):
    """Deterministic plan used if the Planner LLM step fails completely."""
    topic = ctx.topic or "the provided material"
    titles = ["Foundations", "Core ideas", "Practice and application"][:min(3, ctx.settings["max_modules"])]
    n, src = len(titles), ctx.source
    chunk = max(1, len(src) // n) if src else 0
    modules = [{"id": f"M{i + 1}", "title": f"{t}: {topic}", "objective_ids": [f"O{i + 1}"],
                "key_points": [], "depth": "core",
                "source_notes": src[i * chunk:(i + 1) * chunk][:2500] if src else ""}
               for i, t in enumerate(titles)]
    objectives = [{"id": f"O{i + 1}", "text": f"Understand the {t.lower()} of {topic}"}
                  for i, t in enumerate(titles)]
    schedule = [{"day": f"Day {d + 1}", "focus": modules[d % n]["title"],
                 "module_ids": [modules[d % n]["id"]],
                 "tasks": ["Read the lesson", "Review the flashcards", "Attempt the quiz"],
                 "minutes": ctx.profile["minutes"]} for d in range(ctx.profile["days"])]
    return {"title": f"Study pack: {topic}", "audience_summary": "Generic fallback plan.",
            "learning_objectives": objectives, "modules": modules, "schedule": schedule}


def stage_plan(ctx, llm):
    t0 = time.time()
    ctx.log("Planning", "running", "Designing objectives, modules and schedule", progress=0.05)
    max_modules = ctx.settings["max_modules"]
    schema = {
        "title": "short pack title",
        "audience_summary": "1-2 sentences on how the plan is tailored to this learner",
        "learning_objectives": [{"id": "O1", "text": "measurable objective"}],
        "modules": [{"id": "M1", "title": "module title", "objective_ids": ["O1"],
                     "depth": "intro|core|advanced", "key_points": ["point"],
                     "source_notes": "facts this module must teach, taken from the source (<=120 words)"}],
        "schedule": [{"day": "Day 1", "module_ids": ["M1"], "focus": "focus of the day",
                      "tasks": ["concrete task"], "minutes": 45}],
    }
    system = ("You are the PLANNER stage of a multi-stage study-pack pipeline. You design "
              "personalised learning plans. Reply with ONE valid JSON object and nothing else.")
    user = f"""LEARNER PROFILE
{profile_text(ctx.profile)}

TOPIC: {ctx.topic or '(infer from the source)'}

SOURCE MATERIAL:
{ctx.source or '(none provided - rely on reliable general knowledge)'}

Design the plan:
- 2 to {max_modules} modules, ordered from foundations to advanced, sized for the learner's level.
- Give extra weight to the learner's weak areas / focus.
- Every objective must be covered by at least one module.
- Schedule: exactly {ctx.profile['days']} days, tasks must fit in about {ctx.profile['minutes']} minutes per day.
- Ids must look like O1, O2... for objectives and M1, M2... for modules.
- source_notes must contain the facts from the SOURCE that each module needs, so later stages can
  work without the full source.

Return JSON with this shape:
{json.dumps(schema, indent=2)}"""
    try:
        plan, attempts = llm.json_call(system, user, lambda d: plan_errors(d, max_modules),
                                       0.3, "planner")
        status = "ok" if attempts == 1 else "retried"
        note = f"{len(plan['modules'])} modules, {len(plan['learning_objectives'])} objectives"
    except StageError as exc:
        plan, attempts, status = fallback_plan(ctx), None, "degraded"
        note = "used a simple fallback plan"
        ctx.warnings.append(f"Planner failed ({exc}); a generic fallback plan was used.")
    ctx.plan = normalise_plan(plan)
    ctx.log("Planning", status, note, time.time() - t0, attempts, progress=0.15)


# ----------------------------------------------------------------------------
# STAGE 2: CONTENT GENERATION (one call per module, with context passing)
# ----------------------------------------------------------------------------
def cards_per_module(ctx):
    return max(2, round(ctx.settings["n_cards"] / max(1, len(ctx.plan["modules"]))))


def module_errors(d, n_cards):
    errs = []
    if not isinstance(d.get("lesson"), str) or len(d["lesson"]) < 600:
        errs.append("lesson must be a markdown string of roughly 250-450 words")
    terms = d.get("key_terms")
    if (not isinstance(terms, list) or not terms or
            not all(isinstance(t, dict) and t.get("term") and t.get("definition") for t in terms)):
        errs.append("key_terms must be a non-empty list of {term, definition}")
    cards = d.get("flashcards")
    if (not isinstance(cards, list) or len(cards) < n_cards or
            not all(isinstance(c, dict) and c.get("question") and c.get("answer") for c in cards)):
        errs.append(f"flashcards must be a list of at least {n_cards} {{question, answer}} items")
    if not isinstance(d.get("worked_example"), str) or not d["worked_example"].strip():
        errs.append("worked_example is required")
    if not isinstance(d.get("summary_line"), str) or not d["summary_line"].strip():
        errs.append("summary_line is required")
    return errs


def generate_module(ctx, llm, module, notes="", previous=None):
    n = cards_per_module(ctx)
    objectives = "; ".join(f"{o['id']}: {o['text']}" for o in ctx.plan["learning_objectives"]
                           if o["id"] in module["objective_ids"])
    prior = "\n".join(f"- {m['title']}: {m['summary_line']} (terms: {', '.join(m['terms'][:6])})"
                      for m in ctx.module_memory if m["title"] != module["title"])
    schema = {"lesson": "markdown, 250-450 words, headings + bullets",
              "key_terms": [{"term": "t", "definition": "d"}],
              "worked_example": "markdown worked example or scenario",
              "flashcards": [{"question": "q", "answer": "a"}],
              "summary_line": "one sentence summary of this module"}
    revision = ""
    if notes:
        revision = (f"\n\nTHIS IS A REVISION. Current version:\n{json.dumps(previous)[:5000]}\n"
                    f"Reviewer feedback you MUST fix:\n{notes}\n")
    system = ("You are the CONTENT stage of a multi-stage study-pack pipeline. You write accurate, "
              "engaging lessons tailored to the learner. Reply with ONE valid JSON object only.")
    user = f"""LEARNER PROFILE
{profile_text(ctx.profile)}

PACK: {ctx.plan['title']}
MODULE {module['id']}: {module['title']} (depth: {module['depth']})
Objectives to teach: {objectives}
Key points: {'; '.join(module['key_points']) or '(choose the most important)'}

SOURCE NOTES (ground the lesson in these, do not invent facts):
{module['source_notes'] or '(none - use reliable general knowledge)'}

ALREADY COVERED IN EARLIER MODULES (do not repeat; you may build on them):
{prior or '(this is the first module)'}
{revision}
Write the module. Provide exactly {n} flashcards. Match the learner's level and preferred style.
Return JSON with this shape:
{json.dumps(schema, indent=2)}"""
    data, attempts = llm.json_call(system, user, lambda d: module_errors(d, n), 0.5,
                                   f"module {module['id']}")
    content = {
        "lesson": data["lesson"].strip(),
        "key_terms": [{"term": str(t["term"]), "definition": str(t["definition"])}
                      for t in data["key_terms"]],
        "worked_example": data["worked_example"].strip(),
        "flashcards": [{"question": str(c["question"]), "answer": str(c["answer"])}
                       for c in data["flashcards"][:n]],
        "summary_line": data["summary_line"].strip(),
    }
    return content, attempts


def stage_content(ctx, llm):
    mods = ctx.plan["modules"]
    for i, m in enumerate(mods, 1):
        t0 = time.time()
        frac = 0.15 + 0.45 * (i - 1) / len(mods)
        ctx.log("Content generation", "running", f"Module {i}/{len(mods)}: {m['title']}", progress=frac)
        try:
            content, attempts = generate_module(ctx, llm, m)
            ctx.modules[m["id"]] = content
            # context passing: later modules see what earlier ones already covered
            ctx.module_memory.append({"title": m["title"], "summary_line": content["summary_line"],
                                      "terms": [t["term"] for t in content["key_terms"]]})
            ctx.log("Content generation", "ok" if attempts == 1 else "retried",
                    f"Module {i}: {m['title']}", time.time() - t0, attempts,
                    progress=0.15 + 0.45 * i / len(mods))
        except StageError as exc:
            ctx.warnings.append(f"Module '{m['title']}' was skipped ({exc}).")
            ctx.log("Content generation", "failed", f"Module {i}: {m['title']} - skipped",
                    time.time() - t0)
    if not ctx.modules:
        raise FatalError("No lesson content could be generated. Please try again in a minute.")


# ----------------------------------------------------------------------------
# STAGE 3: ASSESSMENT
# ----------------------------------------------------------------------------
def quiz_errors(d, n, obj_ids, mod_ids):
    qs = d.get("questions")
    if not isinstance(qs, list) or len(qs) < n:
        return [f"questions must be a list of at least {n} items"]
    errs = []
    for i, q in enumerate(qs[:n], 1):
        if not isinstance(q, dict):
            errs.append(f"q{i} is not an object")
            continue
        opts, idx = q.get("options"), q.get("answer_index")
        if not q.get("question"):
            errs.append(f"q{i}: missing question text")
        if not isinstance(opts, list) or len(opts) != 4 or len({str(o).strip().lower() for o in opts}) != 4:
            errs.append(f"q{i}: needs exactly 4 distinct options")
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx <= 3:
            errs.append(f"q{i}: answer_index must be an integer 0-3")
        if not q.get("explanation"):
            errs.append(f"q{i}: missing explanation")
        if str(q.get("objective_id")) not in obj_ids:
            errs.append(f"q{i}: objective_id must be one of {sorted(obj_ids)}")
        if str(q.get("module_id")) not in mod_ids:
            errs.append(f"q{i}: module_id must be one of {sorted(mod_ids)}")
    return errs


def generate_quiz(ctx, llm, notes="", previous=None):
    n = ctx.settings["n_quiz"]
    objs = active_objectives(ctx)
    obj_ids = {o["id"] for o in objs}
    mod_ids = {m["id"] for m in active_modules(ctx)}
    schema = {"questions": [{"module_id": "M1", "objective_id": "O1",
                             "difficulty": "easy|medium|hard", "question": "q",
                             "options": ["a", "b", "c", "d"], "answer_index": 0,
                             "explanation": "why the answer is right"}]}
    revision = ""
    if notes:
        revision = (f"\n\nTHIS IS A REVISION. Current quiz:\n{json.dumps(previous)[:5000]}\n"
                    f"Fix these problems, keep good questions unchanged:\n{notes}\n")
    system = ("You are the ASSESSMENT stage of a multi-stage study-pack pipeline. You write fair, "
              "unambiguous multiple-choice questions. Reply with ONE valid JSON object only.")
    user = f"""LEARNER PROFILE
{profile_text(ctx.profile)}

OBJECTIVES
{chr(10).join(f"{o['id']}: {o['text']}" for o in objs)}

LESSONS (write questions ONLY about what is taught here)
{lessons_digest(ctx)}
{revision}
Write exactly {n} multiple-choice questions with 4 plausible options and one correct answer.
Mix easy/medium/hard for a {ctx.profile['level']} learner and cover every objective at least once
when possible. Tag each question with the module_id and objective_id it tests.
Return JSON with this shape:
{json.dumps(schema, indent=2)}"""
    data, attempts = llm.json_call(system, user, lambda d: quiz_errors(d, n, obj_ids, mod_ids),
                                   0.4, "quiz")
    quiz = []
    for i, q in enumerate(data["questions"][:n], 1):
        opts = [str(o).strip() for o in q["options"]]
        correct = opts[q["answer_index"]]
        random.shuffle(opts)  # removes the "answer is always B" bias in code
        quiz.append({"id": f"Q{i}", "module_id": str(q["module_id"]),
                     "objective_id": str(q["objective_id"]),
                     "difficulty": str(q.get("difficulty", "medium")),
                     "question": str(q["question"]).strip(), "options": opts,
                     "answer_index": opts.index(correct),
                     "explanation": str(q["explanation"]).strip()})
    return quiz, attempts


def stage_assessment(ctx, llm):
    t0 = time.time()
    ctx.log("Assessment", "running", "Writing a quiz mapped to the objectives", progress=0.65)
    try:
        ctx.quiz, attempts = generate_quiz(ctx, llm)
        ctx.log("Assessment", "ok" if attempts == 1 else "retried",
                f"{len(ctx.quiz)} questions", time.time() - t0, attempts, progress=0.75)
    except StageError as exc:
        ctx.quiz = []
        ctx.warnings.append(f"The quiz could not be generated ({exc}). The rest of the pack is intact.")
        ctx.log("Assessment", "degraded", "quiz skipped", time.time() - t0, progress=0.75)


# ----------------------------------------------------------------------------
# STAGE 4: REVIEW (deterministic checks + AI judge)
# ----------------------------------------------------------------------------
def static_issues(ctx):
    issues = []
    if not ctx.quiz:
        return issues
    covered = {q["objective_id"] for q in ctx.quiz}
    objs = active_objectives(ctx)
    if len(ctx.quiz) >= len(objs):
        for o in objs:
            if o["id"] not in covered:
                issues.append({"target": "quiz", "severity": "medium", "source": "auto-check",
                               "problem": f"No quiz question covers objective {o['id']} ({o['text']}).",
                               "fix": f"Replace a redundant question with one that tests {o['id']}."})
    seen = {}
    for q in ctx.quiz:
        key = re.sub(r"\W+", " ", q["question"].lower()).strip()
        if key in seen:
            issues.append({"target": f"quiz:{q['id']}", "severity": "medium", "source": "auto-check",
                           "problem": f"Duplicates question {seen[key]}.",
                           "fix": "Rewrite it to test a different idea."})
        seen[key] = q["id"]
    return issues


def review_errors(d):
    errs = []
    s = d.get("scores")

    def bad(v):
        return isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5

    if not isinstance(s, dict) or any(bad(s.get(k)) for k in SCORE_KEYS):
        errs.append("scores must contain numbers 1-5 for: " + ", ".join(SCORE_KEYS))
    if not isinstance(d.get("issues"), list):
        errs.append("issues must be a list")
    return errs


def sanitize_issues(raw, allowed):
    out = []
    for i in raw if isinstance(raw, list) else []:
        if not isinstance(i, dict):
            continue
        target = str(i.get("target", "")).strip()
        problem = str(i.get("problem", "")).strip()
        if target not in allowed or not problem:
            continue
        sev = str(i.get("severity", "medium")).lower()
        out.append({"target": target, "severity": sev if sev in ("low", "medium", "high") else "medium",
                    "problem": problem, "fix": str(i.get("fix", "")).strip() or "Fix this problem.",
                    "source": "AI reviewer"})
    return out[:10]


def actionable(issues):
    return [i for i in issues if i["severity"] in ("high", "medium") and i["target"] != "plan"]


def stage_review(ctx, llm, round_no):
    t0 = time.time()
    ctx.log("Review", "running", f"Round {round_no + 1}: checking accuracy, coverage, alignment",
            progress=0.78 + 0.06 * round_no)
    allowed = {"plan", "quiz"} | {f"module:{m['id']}" for m in active_modules(ctx)} \
        | {f"quiz:{q['id']}" for q in ctx.quiz}
    issues, scores, strengths, llm_ok, attempts = static_issues(ctx), None, [], True, None
    quiz_text = json.dumps([{k: q[k] for k in ("id", "module_id", "objective_id", "question",
                                               "options", "answer_index")} for q in ctx.quiz])[:6000]
    schema = {"scores": {k: "1-5" for k in SCORE_KEYS}, "verdict": "pass|revise",
              "strengths": ["short strength"],
              "issues": [{"target": "module:M1 | quiz:Q3 | quiz | plan", "severity": "low|medium|high",
                          "problem": "what is wrong", "fix": "precise instruction to fix it"}]}
    system = ("You are the REVIEW stage of a multi-stage study-pack pipeline: a strict but fair "
              "educational editor. Reply with ONE valid JSON object only.")
    user = f"""LEARNER PROFILE
{profile_text(ctx.profile)}

OBJECTIVES
{chr(10).join(f"{o['id']}: {o['text']}" for o in ctx.plan['learning_objectives'])}

LESSONS AND THEIR SOURCE NOTES
{lessons_digest(ctx, per_module=1300, with_notes=True)}

QUIZ (answer_index is the correct option)
{quiz_text}

Judge the pack. Only flag REAL problems: factual errors, claims not supported by the source notes,
wrong answer keys, ambiguous questions, missing objectives, or mismatch with the learner's level,
goal or focus. At most 8 issues; return an empty list if the pack is good.
Allowed targets: {sorted(allowed)}
Return JSON with this shape:
{json.dumps(schema, indent=2)}"""
    try:
        data, attempts = llm.json_call(system, user, review_errors, 0.2, "reviewer")
        issues += sanitize_issues(data["issues"], allowed)
        scores = {k: float(data["scores"][k]) for k in SCORE_KEYS}
        strengths = [s for s in data.get("strengths", []) if isinstance(s, str)][:4]
    except StageError:
        llm_ok = False
        ctx.warnings.append(f"The AI reviewer was unavailable in round {round_no + 1}; "
                            "only automatic checks were applied.")
    review = {"round": round_no + 1, "scores": scores, "issues": issues, "strengths": strengths,
              "llm_ok": llm_ok, "passed": not actionable(issues)}
    ctx.reviews.append(review)
    status = "degraded" if not llm_ok else "ok" if attempts == 1 else "retried"
    verdict = "passed" if review["passed"] else f"{len(actionable(issues))} issue(s) to fix"
    ctx.log("Review", status, f"Round {round_no + 1}: {verdict}", time.time() - t0, attempts,
            progress=0.84 + 0.06 * round_no)
    return review


# ----------------------------------------------------------------------------
# STAGE 5: REFINEMENT (targeted rewrites only)
# ----------------------------------------------------------------------------
def stage_refine(ctx, llm, issues, round_no):
    t0 = time.time()
    ctx.log("Refinement", "running", f"Round {round_no}: fixing {len(issues)} issue(s)",
            progress=0.86 + 0.05 * (round_no - 1))
    by_target = defaultdict(list)
    for i in issues:
        by_target[i["target"]].append(i)

    fixed, failed, revised = [], [], []
    for target, items in by_target.items():
        if not target.startswith("module:"):
            continue
        mid = target.split(":", 1)[1]
        module = next((m for m in ctx.plan["modules"] if m["id"] == mid), None)
        if module is None or mid not in ctx.modules:
            continue
        notes = "\n".join(f"- {i['problem']} Fix: {i['fix']}" for i in items)
        try:
            ctx.modules[mid], _ = generate_module(ctx, llm, module, notes, ctx.modules[mid])
            fixed.append(module["title"])
            revised.append(module["title"])
        except StageError as exc:
            failed.append(module["title"])
            ctx.warnings.append(f"Could not refine '{module['title']}'; kept the previous version ({exc}).")

    quiz_items = [i for t, its in by_target.items() if t == "quiz" or t.startswith("quiz:") for i in its]
    if ctx.quiz and (quiz_items or revised):
        notes = "\n".join(f"- [{i['target']}] {i['problem']} Fix: {i['fix']}" for i in quiz_items)
        if revised:  # context passing: tell the quiz writer which lessons changed
            notes += ("\n- These lessons were rewritten; make sure related questions match the new "
                      "text: " + ", ".join(revised))
        try:
            ctx.quiz, _ = generate_quiz(ctx, llm, notes.strip(), ctx.quiz)
            fixed.append("quiz")
        except StageError as exc:
            failed.append("quiz")
            ctx.warnings.append(f"Could not refine the quiz; kept the previous version ({exc}).")

    status = "failed" if failed and not fixed else "degraded" if failed else "ok"
    ctx.log("Refinement", status, f"Round {round_no}: revised {', '.join(fixed) or 'nothing'}",
            time.time() - t0, progress=0.9 + 0.03 * (round_no - 1))


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def run_workflow(ctx, llm):
    stage_plan(ctx, llm)
    stage_content(ctx, llm)
    stage_assessment(ctx, llm)
    rounds = ctx.settings["refine_rounds"]
    for r in range(rounds + 1):
        review = stage_review(ctx, llm, r)
        if review["passed"] or r == rounds:
            break
        stage_refine(ctx, llm, actionable(review["issues"]), r + 1)
    ctx.log("Workflow", "ok", f"Finished with {llm.calls} LLM call(s)", progress=1.0)
    return ctx


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
def to_markdown(ctx) -> str:
    p = ctx.plan
    out = [f"# {p['title']}", "", p["audience_summary"], "", "## Learning objectives"]
    out += [f"- **{o['id']}** {o['text']}" for o in p["learning_objectives"]]
    out += ["", "## Study schedule"]
    for d in p["schedule"]:
        out.append(f"### {d['day']} - {d['focus']}" + (f" ({d['minutes']} min)" if d["minutes"] else ""))
        out += [f"- {t}" for t in d["tasks"]] + [""]
    out.append("## Lessons")
    for m in active_modules(ctx):
        c = ctx.modules[m["id"]]
        out += [f"### {m['title']}", c["lesson"], "", "**Key terms**"]
        out += [f"- **{t['term']}**: {t['definition']}" for t in c["key_terms"]]
        out += ["", "**Worked example**", c["worked_example"], ""]
    out.append("## Flashcards")
    n = 0
    for m in active_modules(ctx):
        for c in ctx.modules[m["id"]]["flashcards"]:
            n += 1
            out += [f"**Q{n}. {c['question']}**", f"A: {c['answer']}", ""]
    if ctx.quiz:
        out.append("## Quiz")
        for i, q in enumerate(ctx.quiz, 1):
            out.append(f"**{i}. {q['question']}**")
            out += [f"   {chr(65 + j)}. {o}" for j, o in enumerate(q["options"])]
            out += [f"   Answer: {chr(65 + q['answer_index'])}. {q['explanation']}", ""]
    return "\n".join(out)


def to_json(ctx) -> str:
    return json.dumps({"plan": ctx.plan, "modules": ctx.modules, "quiz": ctx.quiz,
                       "reviews": ctx.reviews, "warnings": ctx.warnings, "trace": ctx.trace},
                      indent=2)


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def configured_key():
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


def render_quiz(quiz):
    st.caption("Pick an answer for instant feedback.")
    for q in quiz:
        st.markdown(f"**{q['id']}. {q['question']}**  \n"
                    f"<small>{q['difficulty']} | tests {q['objective_id']}</small>",
                    unsafe_allow_html=True)
        choice = st.radio("Choose one", q["options"], index=None, key=f"quiz_{q['id']}",
                          label_visibility="collapsed")
        if choice is not None:
            right = q["options"][q["answer_index"]]
            if choice == right:
                st.success("Correct! " + q["explanation"])
            else:
                st.error(f"Not quite. Correct answer: {right}. {q['explanation']}")
        st.divider()


def render_review(ctx):
    for r in ctx.reviews:
        label = f"Round {r['round']} - {'passed ✅' if r['passed'] else 'issues found ⚠️'}"
        with st.expander(label, expanded=r is ctx.reviews[-1]):
            if r["scores"]:
                cols = st.columns(len(SCORE_KEYS))
                for col, k in zip(cols, SCORE_KEYS):
                    col.metric(k.title(), f"{r['scores'][k]:.0f}/5")
            else:
                st.info("AI scoring unavailable this round; automatic checks only.")
            for s in r["strengths"]:
                st.markdown(f"- 👍 {s}")
            if r["issues"]:
                st.dataframe([{"target": i["target"], "severity": i["severity"], "found by": i["source"],
                               "problem": i["problem"], "fix": i["fix"]} for i in r["issues"]],
                             hide_index=True)
            else:
                st.write("No issues found.")


def render_result(ctx):
    plan = ctx.plan
    st.header(plan["title"])
    if plan["audience_summary"]:
        st.caption(plan["audience_summary"])
    for w in ctx.warnings:
        st.warning(w)

    last = ctx.reviews[-1] if ctx.reviews else None
    n_cards = sum(len(c["flashcards"]) for c in ctx.modules.values())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modules", len(ctx.modules))
    m2.metric("Flashcards", n_cards)
    m3.metric("Quiz questions", len(ctx.quiz))
    m4.metric("Final review", "Passed" if last and last["passed"] else "Open issues" if last else "n/a")

    tabs = st.tabs(["🗺️ Plan & schedule", "📖 Lessons", "🃏 Flashcards", "❓ Quiz",
                    "🔍 Review", "🧭 Workflow trace"])
    with tabs[0]:
        for o in plan["learning_objectives"]:
            st.markdown(f"- **{o['id']}** {o['text']}")
        st.subheader("Schedule")
        for d in plan["schedule"]:
            mins = f" - {d['minutes']} min" if d["minutes"] else ""
            with st.expander(f"{d['day']}: {d['focus']}{mins}"):
                for t in d["tasks"]:
                    st.checkbox(t, key=f"task_{d['day']}_{t[:40]}")
    with tabs[1]:
        for m in active_modules(ctx):
            c = ctx.modules[m["id"]]
            with st.expander(f"{m['id']} - {m['title']}", expanded=False):
                st.markdown(c["lesson"])
                st.markdown("**Key terms**")
                for t in c["key_terms"]:
                    st.markdown(f"- **{t['term']}**: {t['definition']}")
                st.markdown("**Worked example**")
                st.markdown(c["worked_example"])
    with tabs[2]:
        st.caption("Click a card to reveal the answer.")
        n = 0
        for m in active_modules(ctx):
            for c in ctx.modules[m["id"]]["flashcards"]:
                n += 1
                with st.expander(f"Card {n} - {c['question']}"):
                    st.write(c["answer"])
    with tabs[3]:
        if ctx.quiz:
            render_quiz(ctx.quiz)
        else:
            st.info("No quiz was generated for this pack.")
    with tabs[4]:
        render_review(ctx)
    with tabs[5]:
        st.dataframe([{k: v for k, v in e.items() if k != "progress"} for e in ctx.trace],
                     hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download pack (Markdown)", to_markdown(ctx), "study_pack.md", "text/markdown")
    d2.download_button("⬇️ Download workflow data (JSON)", to_json(ctx), "study_pack.json",
                       "application/json")


def main():
    st.set_page_config(page_title="AI Study Pack Workflow", page_icon="📚", layout="wide")
    st.title("📚 AI Study Pack Generator")
    st.caption("A multi-stage AI workflow: Plan → Generate → Assess → Review → Refine")

    with st.expander("How the workflow works"):
        st.graphviz_chart(WORKFLOW_DOT)
        st.markdown(
            "- **Planner** turns your profile and source into objectives, modules and a schedule.\n"
            "- **Content** writes one module at a time, seeing what earlier modules covered.\n"
            "- **Assessment** writes a quiz mapped to the objectives, from the lessons only.\n"
            "- **Reviewer** runs automatic checks plus an AI editor that scores the pack.\n"
            "- **Refiner** rewrites only the flagged parts, then the pack is re-reviewed.")

    with st.sidebar:
        st.header("⚙️ Settings")
        if configured_key():
            st.success("API key loaded ✅")
        else:
            st.text_input("Groq API key", type="password", key="manual_key",
                          help="Free key at console.groq.com. On Streamlit Cloud use Secrets.")
        max_modules = st.slider("Max modules", 2, 6, 4)
        n_cards = st.slider("Total flashcards", 6, 30, 12)
        n_quiz = st.slider("Quiz questions", 3, 15, 6)
        refine_rounds = st.slider("Max refinement rounds", 0, 2, 1,
                                  help="0 = review only, no automatic rewriting")

    left, right = st.columns(2)
    with left:
        st.subheader("1. Your material")
        topic = st.text_input("Topic", placeholder="e.g. Operating system scheduling")
        notes = st.text_area("Paste notes (optional)", height=150)
        uploaded = st.file_uploader("Or upload PDF / TXT / MD", type=["pdf", "txt", "md"])
    with right:
        st.subheader("2. About you")
        level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"], index=1)
        goal = st.text_input("Goal", placeholder="e.g. Pass my midterm, prepare for interviews")
        c1, c2 = st.columns(2)
        days = c1.number_input("Days available", 1, 14, 5)
        minutes = c2.number_input("Minutes per day", 15, 240, 45, step=15)
        styles = st.multiselect("Learning style", ["Analogies & examples", "Step-by-step",
                                                   "Concise", "Practice-heavy"])
        focus = st.text_input("Weak areas / focus (optional)")

    if st.button("✨ Generate study pack", type="primary"):
        api_key = configured_key() or st.session_state.get("manual_key", "")
        source = (notes.strip() + "\n\n" + extract_text(uploaded).strip()).strip()
        if not api_key:
            st.error("Add your Groq API key in the sidebar or in Streamlit Secrets.")
            return
        if not topic.strip() and not source:
            st.error("Enter a topic, paste notes, or upload a file.")
            return
        if len(source) > MAX_SOURCE_CHARS:
            st.info(f"Long source: only the first {MAX_SOURCE_CHARS:,} characters are used.")
            source = source[:MAX_SOURCE_CHARS]

        ctx = Ctx(topic=topic.strip(), source=source,
                  profile={"level": level, "goal": goal.strip(), "days": int(days),
                           "minutes": int(minutes), "styles": styles, "focus": focus.strip()},
                  settings={"max_modules": max_modules, "n_cards": n_cards, "n_quiz": n_quiz,
                            "refine_rounds": refine_rounds})
        for k in [k for k in st.session_state if str(k).startswith(("quiz_", "task_"))]:
            del st.session_state[k]
        st.session_state.pop("ctx", None)

        with st.status("Running the multi-stage workflow...", expanded=True) as status:
            bar = st.progress(0.0)

            def on_event(e):
                st.write(f"{ICONS[e['status']]} **{e['stage']}** - {e['message']}")
                if e.get("progress") is not None:
                    bar.progress(min(float(e["progress"]), 1.0))

            ctx.emit = on_event
            try:
                run_workflow(ctx, LLM(api_key))
                ctx.emit = None
                st.session_state["ctx"] = ctx
                status.update(label="Study pack ready", state="complete", expanded=False)
            except FatalError as exc:
                status.update(label="Workflow stopped", state="error")
                st.error(str(exc))
            except Exception as exc:  # last-resort boundary so the app never shows a stack trace
                status.update(label="Workflow stopped", state="error")
                st.error(f"Unexpected error: {exc}")

    if "ctx" in st.session_state:
        st.divider()
        render_result(st.session_state["ctx"])


if __name__ == "__main__":
    main()
