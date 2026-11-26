"""Tests must not depend on a developer's local backend/.env (which may select real models).

Environment variables override .env in pydantic-settings, so pin the model settings before the app reads them.
Tests that need other settings monkeypatch get_settings themselves.
"""
import os

for name, value in {'MODEL_MODE': 'simulation', 'TEXT_MODEL_PATH': '', 'MULTIMODAL_MODEL_PATH': '', 'SINHALA_MODEL_PATH': '',
                    'STANCE_MODEL': '', 'TRUST_MODEL_PATH': '', 'EMBEDDING_MODEL': '', 'DOMAIN_MODEL_PATH': '', 'JUDGE_ENABLED': 'false', 'JUDGE_MODEL': '', 'GEMINI_API_KEYS': '', 'WIKIPEDIA_INTRO': 'false'}.items():
    os.environ[name] = value
