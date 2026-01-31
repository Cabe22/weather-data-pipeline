"""
Model Registry for tracking model versions and metadata.
Stores version history in a JSON file alongside the models directory.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Track model versions, metadata, and performance history."""

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = registry_path
        self._registry: Dict = self._load()

    def _load(self) -> Dict:
        """Load registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read registry file: {e}")
        return {"versions": []}

    def _save(self) -> None:
        """Persist registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path) or '.', exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)

    def _next_version(self) -> str:
        """Generate the next sequential version string (v1, v2, ...)."""
        versions = self._registry.get("versions", [])
        if not versions:
            return "v1"
        last = max(int(v["version"].lstrip("v")) for v in versions)
        return f"v{last + 1}"

    def _file_hash(self, filepath: str) -> Optional[str]:
        """Compute SHA-256 of a model artifact file."""
        if not os.path.exists(filepath):
            return None
        sha = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha.update(chunk)
        return sha.hexdigest()

    def register(self, model_type: str, artifact_path: str,
                 metadata: Dict) -> str:
        """Register a new model version.

        Args:
            model_type: e.g. 'temperature' or 'rain'
            artifact_path: path to the saved .pkl file
            metadata: training metadata dict from WeatherPredictor

        Returns:
            The version string assigned to this model.
        """
        version = self._next_version()
        entry = {
            "version": version,
            "model_type": model_type,
            "artifact_path": artifact_path,
            "artifact_hash": self._file_hash(artifact_path),
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
        }
        self._registry["versions"].append(entry)
        self._save()
        logger.info(f"Registered {model_type} model as {version}")
        return version

    def list_versions(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all registered versions, optionally filtered by model type."""
        versions = self._registry.get("versions", [])
        if model_type:
            versions = [v for v in versions if v["model_type"] == model_type]
        return versions

    def get_version(self, version: str) -> Optional[Dict]:
        """Get a specific version entry by version string."""
        for v in self._registry.get("versions", []):
            if v["version"] == version:
                return v
        return None

    def get_latest(self, model_type: str) -> Optional[Dict]:
        """Get the most recently registered version for a model type."""
        versions = self.list_versions(model_type)
        return versions[-1] if versions else None

    def compare(self, version_a: str, version_b: str) -> Optional[Dict]:
        """Compare metrics between two registered versions.

        Returns a dict with each metric showing both values and the delta,
        or None if either version is not found.
        """
        a = self.get_version(version_a)
        b = self.get_version(version_b)
        if not a or not b:
            return None

        metrics_a = a.get("metadata", {}).get("metrics", {})
        metrics_b = b.get("metadata", {}).get("metrics", {})
        all_keys = set(metrics_a) | set(metrics_b)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "model_type_a": a["model_type"],
            "model_type_b": b["model_type"],
            "metrics": {},
        }
        for key in sorted(all_keys):
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            delta = None
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
            comparison["metrics"][key] = {
                version_a: val_a,
                version_b: val_b,
                "delta": delta,
            }
        return comparison
