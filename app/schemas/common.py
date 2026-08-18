from __future__ import annotations

from pydantic import BaseModel, model_validator


class PlayerLookupRequest(BaseModel):
    player_id: int | str | None = None
    player_name: str | None = None

    @model_validator(mode="after")
    def require_player_identifier(self):
        if self.player_id is None and self.player_name is None:
            raise ValueError("player_id or player_name is required")
        return self
