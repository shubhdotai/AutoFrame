"""Check integrity and source selection without requiring a public mirror."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('download_models', Path(__file__).resolve().parents[1] / 'scripts/download_models.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_verified_fetch_and_failed_hash_cleanup(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'checkpoint bytes')
    target = tmp_path / 'models/model'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    download.fetch(source.as_uri(), target, digest, 'model')
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError, match='checksum mismatch'):
        download.fetch(source.as_uri(), target, '0' * 64, 'model')
    assert target.read_bytes() == source.read_bytes()
    assert list(target.parent.iterdir()) == [target]


def test_existing_mismatched_file_is_preserved(tmp_path):
    target = tmp_path / 'model'
    target.write_bytes(b'existing')
    asset = {'files': [{'path': 'model', 'sha256': '0' * 64, 'size': 8}]}
    with pytest.raises(SystemExit, match='checksum mismatch'):
        download.install(asset, 'asd', tmp_path, 'shubhdotai/autoclip', 'main', 'hf')
    assert target.read_bytes() == b'existing'


def test_mirror_url_and_upstream_selection(tmp_path, monkeypatch):
    urls = []
    monkeypatch.setattr(download, 'fetch', lambda url, *args: urls.append(url))
    asset = {'files':[{'path':'pretrain_AVA.model','size':1,'sha256':'x'}],
             'origin':{'pretrain_AVA.model':'https://example.org/upstream.model'}}
    download.install(asset, 'asd', tmp_path, 'shubhdotai/autoclip', 'revision', 'hf')
    download.install(asset, 'asd', tmp_path, 'shubhdotai/autoclip', 'revision', 'upstream')
    assert urls == ['https://huggingface.co/shubhdotai/autoclip/resolve/revision/pretrain_AVA.model',
                    'https://example.org/upstream.model']
