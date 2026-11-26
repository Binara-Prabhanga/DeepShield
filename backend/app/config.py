from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    mongodb_uri: str = 'mongodb://localhost:27017'
    mongodb_db: str = 'deepshield'
    jwt_secret: str = 'DEV_ONLY_CHANGE_ME_BEFORE_DEPLOYMENT'
    frontend_origins: str = 'http://localhost:5173,http://127.0.0.1:5173'
    model_mode: str = 'simulation'
    tavily_api_key: str = ''
    google_factcheck_api_key: str = ''
    gemini_api_key: str = ''
    gemini_api_keys: str = ''  # extra Gemini keys, comma-separated; the judge moves to the next one when a key hits its quota
    gemini_model: str = 'gemini-3.8-flash'
    text_model_path: str = ''
    multimodal_model_path: str = ''
    sinhala_model_path: str = ''  # directory with model.joblib + config.json (research/eval_sinhala.py --export)
    stance_model: str = ''  # NLI model for evidence stance (C1); empty = disabled
    trust_model_path: str = ''  # directory with trust_config.json (C2); empty = disabled
    judge_enabled: bool = False  # Gemini evidence judge (app/services/judge.py); sends claim + snippets to Google
    judge_variant: str = 'plausibility'  # judge prompt: 'strict' or 'knowledge' (see app/services/judge.py)
    judge_model: str = ''  # Gemini model for the judge; empty = GEMINI_MODEL
    wikipedia_intro: bool = False  # use each Wikipedia page's opening sentences as the evidence text (not the search fragment)
    domain_model_path: str = ''  # directory with domain.joblib (research/fit_domain.py); empty = out-of-domain check disabled
    embedding_model: str = ''  # sentence-embedding model for evidence ranking (e.g. sentence-transformers/all-MiniLM-L6-v2); empty = TF-IDF

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(',') if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
