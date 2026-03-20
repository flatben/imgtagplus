from pathlib import Path

import pytest

from imgtagplus.vlm import (
    _florence_pretrained_kwargs,
    _resolve_florence_revision,
    _validate_florence_transformers_version,
)


def test_resolve_florence_revision_uses_variant_specific_pins() -> None:
    assert _resolve_florence_revision("microsoft/Florence-2-base") == "5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
    assert _resolve_florence_revision("microsoft/Florence-2-large") == "21a599d414c4d928c9032694c424fb94458e3594"


def test_resolve_florence_revision_leaves_unknown_models_unpinned() -> None:
    assert _resolve_florence_revision("microsoft/Florence-2-large-ft") is None


def test_florence_pretrained_kwargs_include_revision_when_available() -> None:
    kwargs = _florence_pretrained_kwargs("microsoft/Florence-2-large", Path("/tmp/cache"))

    assert kwargs["trust_remote_code"] is True
    assert kwargs["cache_dir"] == "/tmp/cache"
    assert kwargs["revision"] == "21a599d414c4d928c9032694c424fb94458e3594"


def test_florence_pretrained_kwargs_skip_revision_for_unknown_models() -> None:
    kwargs = _florence_pretrained_kwargs("custom/florence-experiment", Path("/tmp/cache"))

    assert "trust_remote_code" not in kwargs
    assert kwargs["cache_dir"] == "/tmp/cache"
    assert "revision" not in kwargs


def test_validate_florence_transformers_version_rejects_5x() -> None:
    with pytest.raises(RuntimeError, match="transformers>=4.44.2,<5.0.0"):
        _validate_florence_transformers_version("5.3.0")


def test_validate_florence_transformers_version_allows_supported_range() -> None:
    _validate_florence_transformers_version("4.44.2")
