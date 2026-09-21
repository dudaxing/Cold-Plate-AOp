"""The variant machinery itself: the patches must bite, and must fail loudly.

Cheap guards. Without them a typo could turn a substitution into a no-op and
test_supg_variants.py would quietly compare upstream against upstream.
"""

import pytest

from validation import variants


def test_every_substitution_changes_the_source():
    source = variants._upstream_source()
    for old, new in (variants.TAU1, variants.SUPG_BRINKMAN, variants.SUPG_CONVECTION):
        assert old in source, f"upstream no longer contains:\n{old}"
        assert new not in source, f"upstream already contains the fix:\n{new}"
        assert source.replace(old, new) != source


def test_variants_differ_from_each_other():
    sources = {}
    for name in ("v0", "v1", "v2", "v3"):
        variants.build(name)  # generates the cached file
        sources[name] = (variants.CACHE / f"fe_fluid_{name}.py").read_text("utf-8")

    assert len({*sources.values()}) == 4, "variants are not all distinct"
    assert sources["v0"] == variants._upstream_source(), "v0 must be untouched upstream"


def test_unknown_variant_is_rejected():
    with pytest.raises(KeyError):
        variants.build("v9")


def test_stale_patch_raises_source_changed(monkeypatch):
    monkeypatch.setattr(variants, "_upstream_source", lambda: "def nothing(): pass\n")
    with pytest.raises(variants.SourceChanged):
        variants.build("v3")
