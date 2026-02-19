"""Tests for dimension scoring: auto-derivation, prompt generation, validation, fallback."""
import re
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from taster.core.config import Config
from taster.core.profiles import TasteProfile, CategoryDefinition
from taster.classification.prompt_builder import PromptBuilder
from taster.classification.classifier import MediaClassifier


# ── Dimension auto-derivation ────────────────────────────────────────


class TestDimensionAutoDerivation:

    def _make_profile(self, priorities=None, dimensions=None):
        return TasteProfile(
            name="test", description="test", media_types=["image"],
            categories=[CategoryDefinition("Good", "good"), CategoryDefinition("Bad", "bad")],
            top_priorities=priorities or [],
            dimensions=dimensions or [],
        )

    def test_explicit_dimensions_used(self):
        dims = [{"name": "comp", "description": "Composition quality"}]
        profile = self._make_profile(
            priorities=["Some priority"],
            dimensions=dims,
        )
        pb = PromptBuilder(Config(), profile=profile)
        result = pb._get_dimensions()
        assert result == dims

    def test_auto_derive_from_priorities(self):
        profile = self._make_profile(priorities=[
            "Parent-child interaction quality",
            "Baby's expression",
        ])
        pb = PromptBuilder(Config(), profile=profile)
        dims = pb._get_dimensions()
        assert len(dims) == 2
        assert dims[0]["name"] == "parent_child_interaction_quality"
        assert dims[1]["name"] == "baby_expression"  # possessive 's stripped
        assert dims[0]["description"] == "Parent-child interaction quality"

    def test_sanitize_special_chars(self):
        profile = self._make_profile(priorities=["Baby's face (clearly visible!)"])
        pb = PromptBuilder(Config(), profile=profile)
        dims = pb._get_dimensions()
        name = dims[0]["name"]
        assert re.match(r"^[a-z0-9_]+$", name)
        assert len(name) <= 40
        assert "baby_face_clearly_visible" == name  # possessive stripped, parens removed

    def test_max_7_dimensions(self):
        profile = self._make_profile(priorities=[f"Priority {i}" for i in range(10)])
        pb = PromptBuilder(Config(), profile=profile)
        dims = pb._get_dimensions()
        assert len(dims) == 7

    def test_no_priorities_no_dimensions(self):
        profile = self._make_profile(priorities=[])
        pb = PromptBuilder(Config(), profile=profile)
        dims = pb._get_dimensions()
        assert dims == []

    def test_no_profile_no_dimensions(self, tmp_path):
        config = Config()
        # Point paths away from CWD so no legacy JSON is loaded
        config.paths.taste_preferences_generated = tmp_path / "none.json"
        config.paths.taste_preferences = tmp_path / "none2.json"
        pb = PromptBuilder(config)
        dims = pb._get_dimensions()
        assert dims == []


# ── Dimension prompt sections ────────────────────────────────────────


class TestDimensionPromptSections:

    def _make_builder(self, priorities):
        profile = TasteProfile(
            name="test", description="test", media_types=["image"],
            categories=[CategoryDefinition("Share", "share"), CategoryDefinition("Storage", "store")],
            top_priorities=priorities,
        )
        return PromptBuilder(Config(), profile=profile)

    def test_dimensions_section_in_singleton(self):
        pb = self._make_builder(["Composition", "Expression"])
        prompt = pb.build_singleton_prompt(media_type="image")
        assert "DIMENSION SCORING" in prompt
        assert "composition" in prompt.lower()
        assert '"dimensions"' in prompt

    def test_dimensions_section_in_burst(self):
        pb = self._make_builder(["Sharpness", "Emotion"])
        prompt = pb.build_burst_prompt(3)
        assert "DIMENSION SCORING" in prompt
        assert '"dimensions"' in prompt

    def test_dimensions_section_in_video(self):
        pb = self._make_builder(["Audio clarity"])
        prompt = pb.build_video_prompt()
        assert "DIMENSION SCORING" in prompt

    def test_dimensions_section_in_document(self):
        pb = self._make_builder(["Content relevance"])
        prompt = pb.build_singleton_prompt(media_type="document")
        assert "DIMENSION SCORING" in prompt

    def test_dimensions_section_in_audio(self):
        pb = self._make_builder(["Audio quality"])
        prompt = pb.build_singleton_prompt(media_type="audio")
        assert "DIMENSION SCORING" in prompt

    def test_no_dimensions_no_section(self):
        pb = self._make_builder([])
        prompt = pb.build_singleton_prompt(media_type="image")
        assert "DIMENSION SCORING" not in prompt
        assert '"dimensions"' not in prompt

    def test_document_group_has_dimensions(self):
        pb = self._make_builder(["Quality", "Relevance"])
        prompt = pb.build_group_prompt(3, media_type="document")
        assert "DIMENSION SCORING" in prompt
        assert '"dimensions"' in prompt


# ── Dimension validation in classifier ───────────────────────────────


class TestDimensionValidation:

    def _make_classifier(self, priorities=None, dimensions=None):
        profile = TasteProfile(
            name="test", description="test", media_types=["image"],
            categories=[CategoryDefinition("Good", "good")],
            default_category="Good",
            top_priorities=priorities or [],
            dimensions=dimensions or [],
        )
        config = Config()
        mock_client = MagicMock()
        mock_pb = MagicMock()
        return MediaClassifier(config, mock_client, mock_pb, profile=profile)

    def test_validate_dimensions_valid(self):
        clf = self._make_classifier(dimensions=[
            {"name": "comp", "description": "Composition"},
            {"name": "expr", "description": "Expression"},
        ])
        response = {
            "classification": "Good", "score": 4,
            "dimensions": {"comp": 3, "expr": 5},
        }
        result = clf._validate_dimensions(response)
        assert result["dimensions"] == {"comp": 3, "expr": 5}

    def test_validate_dimensions_clamps(self):
        clf = self._make_classifier(dimensions=[
            {"name": "a", "description": "A"},
        ])
        response = {
            "classification": "Good", "score": 4,
            "dimensions": {"a": 7},
        }
        result = clf._validate_dimensions(response)
        assert result["dimensions"]["a"] == 5

    def test_validate_dimensions_fills_missing(self):
        clf = self._make_classifier(dimensions=[
            {"name": "a", "description": "A"},
            {"name": "b", "description": "B"},
        ])
        response = {
            "classification": "Good", "score": 4,
            "dimensions": {"a": 3},
        }
        result = clf._validate_dimensions(response)
        assert result["dimensions"]["a"] == 3
        assert result["dimensions"]["b"] is None

    def test_validate_dimensions_no_dict(self):
        clf = self._make_classifier(dimensions=[
            {"name": "a", "description": "A"},
        ])
        response = {"classification": "Good", "score": 4}
        result = clf._validate_dimensions(response)
        assert result["dimensions"]["a"] is None

    def test_no_dimensions_expected(self):
        clf = self._make_classifier(priorities=[])
        response = {"classification": "Good", "score": 4}
        result = clf._validate_dimensions(response)
        assert "dimensions" not in result

    def test_fallback_includes_dimensions(self):
        clf = self._make_classifier(dimensions=[
            {"name": "x", "description": "X"},
        ])
        fallback = clf._create_fallback_response("test error")
        assert fallback["dimensions"] == {"x": None}

    def test_fallback_no_dimensions_without_profile(self):
        clf = self._make_classifier(priorities=[])
        fallback = clf._create_fallback_response("test error")
        assert "dimensions" not in fallback

    def test_singleton_validation_includes_dimensions(self):
        clf = self._make_classifier(dimensions=[
            {"name": "q", "description": "Quality"},
        ])
        response = {
            "classification": "Good", "score": 4, "reasoning": "ok",
            "dimensions": {"q": 4},
        }
        result = clf._validate_singleton_response(response)
        assert result["dimensions"] == {"q": 4}

    def test_audio_fallback_includes_dimensions(self):
        clf = self._make_classifier(dimensions=[
            {"name": "clarity", "description": "Audio clarity"},
        ])
        fallback = clf._create_fallback_response("err", is_audio=True)
        assert fallback["audio_quality"] == "unknown"
        assert fallback["content_summary"] == ""
        assert fallback["dimensions"] == {"clarity": None}
