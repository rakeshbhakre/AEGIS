"""Runtime configuration and repository data paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    seed: int = 7
    data_dir: Path = PACKAGE_ROOT / "data"
    host: str = "127.0.0.1"
    port: int = 8890
    openrouter_api_key: str | None = None
    omniroute_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            seed=int(os.environ.get("AEGIS_SEED", "7")),
            data_dir=Path(os.environ.get("AEGIS_DATA_DIR", str(PACKAGE_ROOT / "data"))).expanduser().resolve(),
            host=os.environ.get("AEGIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("AEGIS_PORT", "8890")),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
            omniroute_api_key=os.environ.get("OMNIROUTE_API_KEY") or None,
        )


settings = Settings.from_env()
