"""Run the research parts (C1 stance, C2 Trust Score, C3 EEA) on one analysis.

Each part runs only when configured, and none can fail the request: errors become notes.
  C1  STANCE_MODEL set and evidence found  -> evidence items get 'stance'; evidence_score = s_ev
  C2  TRUST_MODEL_PATH set and real models -> calibrated Trust Score with breakdown and decision
      (never on simulated scores: a calibrated number on random inputs would look trustworthy)
  C3  s_ev available and SHAP tokens exist -> Explanation-Evidence Agreement
  FC  a Google fact check of the same claim with a true/false rating -> out['fact_check']: its verdict decides the
      score (app/services/factcheck_verdict.py)
  JDG JUDGE_ENABLED and no fact-check verdict -> out['judge']: Gemini reads the evidence and says supported / refuted /
      not_enough_evidence (app/services/judge.py). Only used conservatively, see main.py
  OOD otherwise, DOMAIN_MODEL_PATH set and the claim is unlike the model's training claims -> out['out_of_domain']:
      the result is "Needs review" instead of a confident number (app/services/domain.py)
"""
from __future__ import annotations

import asyncio

from app.config import get_settings
from app.services.domain import out_of_domain
from app.services.evidence import with_wikipedia_intros
from app.services.explain_audit import eea
from app.services.factcheck_verdict import fact_check_verdict
from app.services.judge import judge
from app.services.model_adapter import ModelOutput
from app.services.stance import classify_stance, evidence_score
from app.services.trust import load_trust_model, trust_score


JUDGE_DEADLINE = 30   # seconds for the whole judging step


async def _judge(claim: str, evidence: list[dict], model: str, variant: str) -> dict | None:
    """The judge reads Wikipedia's opening paragraphs (measured: far more decisive than search fragments)."""
    async def run():
        return await judge(claim, await with_wikipedia_intros(evidence), timeout=15, retries=1, model=model, variant=variant)
    try:
        return await asyncio.wait_for(run(), timeout=JUDGE_DEADLINE)   # never let Gemini delay an analysis for long
    except asyncio.TimeoutError:
        return None


def assess(claim: str, evidence: list[dict], model: ModelOutput) -> tuple[list[dict], dict]:
    """Returns (evidence, with stance when C1 ran; {'evidence_score', 'trust', 'explanation_audit', 'notes'})."""
    settings = get_settings()
    out: dict = {'evidence_score': None, 'trust': None, 'explanation_audit': None, 'notes': [],
                 'fact_check': None, 'out_of_domain': False, 'judge': None}
    if model.mode == 'model-sinhala':   # the NLI and Trust models were trained on English inputs
        if settings.stance_model or settings.trust_model_path:
            out['notes'].append('Evidence stance and calibrated trust score are English-only; skipped for Sinhala text.')
        return evidence, out
    if settings.stance_model and evidence:
        try:
            evidence = classify_stance(claim, evidence)
            out['evidence_score'] = evidence_score(evidence)
        except Exception as exc:  # stance is optional; never fail the analysis because of it
            out['notes'].append(f'Evidence stance unavailable: {type(exc).__name__}')
    if settings.trust_model_path and model.mode != 'simulated':
        has_image = model.image_text_alignment is not None
        signals = {'text': model.text_probability_real,
                   'fused': model.fused_probability_real if has_image else None,
                   'evidence': out['evidence_score']}
        try:
            out['trust'] = trust_score(signals, load_trust_model())
        except Exception as exc:
            out['notes'].append(f'Calibrated trust score unavailable: {type(exc).__name__}')
    if out['evidence_score'] is not None and model.explanation_tokens:
        out['explanation_audit'] = eea(model.explanation_tokens, evidence)
    out['fact_check'] = fact_check_verdict(claim, evidence)
    if out['fact_check']:
        fc = out['fact_check']
        out['notes'].append(f"Fact-checked by {fc['publisher'] or 'a fact-checker'}: rated \"{fc['rating']}\". "
                            'The score follows that fact check, not the model.')
    elif model.mode == 'model':
        if getattr(settings, 'judge_enabled', False) and evidence:
            verdict = asyncio.run(_judge(claim, evidence, getattr(settings, 'judge_model', ''), getattr(settings, 'judge_variant', 'strict')))
            if verdict:
                out['judge'] = verdict
                out['notes'].append(f"Gemini evidence judge: {verdict['verdict'].replace('_', ' ')}. {verdict['reason']}")
            else:
                out['notes'].append('The Gemini evidence judge gave no answer (quota or timeout), so this claim was not '
                                    'checked against the evidence. Try again in a minute.')
        unlike, similarity = out_of_domain(claim)
        if unlike:
            out['out_of_domain'] = True
            out['notes'].append(f'This claim is unlike the political statements the model was trained on '
                                f'(similarity {similarity}), and no matching fact check was found. '
                                'The model score is not reliable here, so it is not used.')
    active = settings.stance_model or settings.trust_model_path or getattr(settings, 'domain_model_path', '') or getattr(settings, 'judge_enabled', False)
    if active and model.mode == 'model' and not out['fact_check'] and not out['out_of_domain'] and (out['judge'] or {}).get('verdict') not in ('supported', 'refuted'):
        out['notes'].append('No fact-checker or evidence review backs this score. It comes mainly from the text model, which '
                            'learned US political claims and cannot check facts, so treat it as a prompt to verify.')
    return evidence, out
