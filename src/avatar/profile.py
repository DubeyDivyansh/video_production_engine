"""
Avatar Profile and Media Asset Data Models.
Provides serialized metadata structures for persistent avatars on Google Drive.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime, timezone


@dataclass
class MediaAsset:
    file_path: str
    media_type: str  # 'image', 'audio', 'video'
    quality_score: float
    timestamp: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    original_filename: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaAsset":
        return cls(**data)


@dataclass
class AvatarProfile:
    avatar_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    optimal_audio_path: Optional[str] = None
    optimal_audio_score: float = -1.0
    optimal_video_path: Optional[str] = None
    optimal_video_score: float = -1.0
    optimal_image_path: Optional[str] = None
    optimal_image_score: float = -1.0
    optimal_body_pose_path: Optional[str] = None
    optimal_body_pose_score: float = -1.0
    embedding_paths: Dict[str, str] = field(default_factory=dict)
    raw_assets: List[MediaAsset] = field(default_factory=list)
    custom_configs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["raw_assets"] = [a.to_dict() if isinstance(a, MediaAsset) else a for a in self.raw_assets]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvatarProfile":
        assets_data = data.get("raw_assets", [])
        assets = [MediaAsset.from_dict(a) if isinstance(a, dict) else a for a in assets_data]
        data_copy = dict(data)
        data_copy["raw_assets"] = assets
        return cls(**data_copy)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AvatarProfile":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Avatar profile not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
