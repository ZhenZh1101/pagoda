"""Shard build bookkeeping: rebuilding stage 2 alone starts from where stage 1 left the pools."""

import json

import pagoda.data.build as B


def test_restore_cursors_after(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "TOK_DIR", tmp_path)
    (tmp_path / "stage1").mkdir()
    (tmp_path / "stage1" / "manifest.json").write_text(json.dumps({"sources": {
        "fineweb": {"pool": "fineweb_edu", "cursor_end": 50},
        "chat": {"pool": "smoltalk_chat", "cursor_end": 7},
    }}))
    cfg = {}
    B.save_cursors(cfg, {"fineweb_edu": 50, "smoltalk_chat": 90, "mc": 30})  # as left by a stage-2 build
    assert B.restore_cursors_after(cfg, "stage1") == {"fineweb_edu": 50, "smoltalk_chat": 7}
    assert B.load_cursors(cfg) == {"fineweb_edu": 50, "smoltalk_chat": 7}  # stage-2-only pools restart at 0
