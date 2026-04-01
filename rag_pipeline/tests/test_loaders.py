import pytest
import json
import tempfile
from pathlib import Path

from src.loaders import LoaderFactory, Document
from src.loaders.text_loader import TextLoader
from src.loaders.json_loader import JSONLoader


@pytest.fixture
def tmp_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Ini adalah dokumen uji coba.\nBaris kedua di sini.", encoding="utf-8")
    return str(f)


@pytest.fixture
def tmp_md(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Judul\n\nIni paragraf pertama.\n\n## Sub-judul\n\nIsi sub-judul.", encoding="utf-8")
    return str(f)


@pytest.fixture
def tmp_json(tmp_path):
    data = [{"id": 1, "title": "Artikel A", "content": "Isi artikel pertama."},
            {"id": 2, "title": "Artikel B", "content": "Isi artikel kedua."}]
    f = tmp_path / "test.json"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(f)


@pytest.fixture
def tmp_jsonl(tmp_path):
    lines = [
        json.dumps({"id": 1, "text": "Baris pertama"}, ensure_ascii=False),
        json.dumps({"id": 2, "text": "Baris kedua"}, ensure_ascii=False),
    ]
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(lines), encoding="utf-8")
    return str(f)


# === TextLoader ===

def test_text_loader_txt(tmp_txt):
    docs = TextLoader().load(tmp_txt)
    assert len(docs) == 1
    assert "dokumen uji coba" in docs[0].content
    assert docs[0].metadata["type"] == "txt"


def test_text_loader_md(tmp_md):
    docs = TextLoader().load(tmp_md)
    assert len(docs) == 1
    assert "Judul" in docs[0].content


def test_text_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        TextLoader().load("/tidak/ada/file.txt")


# === JSONLoader ===

def test_json_loader_list(tmp_json):
    docs = JSONLoader().load(tmp_json)
    assert len(docs) == 2
    assert "Artikel A" in docs[0].content


def test_json_loader_text_key(tmp_json):
    docs = JSONLoader(text_key="content").load(tmp_json)
    assert docs[0].content == "Isi artikel pertama."


def test_jsonl_loader(tmp_jsonl):
    docs = JSONLoader().load(tmp_jsonl)
    assert len(docs) == 2
    assert docs[0].metadata["line"] == 1


# === LoaderFactory ===

def test_factory_auto_detect_txt(tmp_txt):
    docs = LoaderFactory.load(tmp_txt)
    assert len(docs) > 0


def test_factory_auto_detect_json(tmp_json):
    docs = LoaderFactory.load(tmp_json)
    assert len(docs) == 2


def test_factory_unsupported_extension(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("test")
    with pytest.raises(ValueError, match="Tidak ada loader"):
        LoaderFactory.load(str(f))


def test_factory_load_many(tmp_txt, tmp_json):
    docs = LoaderFactory.load_many([tmp_txt, tmp_json])
    assert len(docs) == 3  # 1 txt + 2 json records


def test_document_has_doc_id(tmp_txt):
    docs = LoaderFactory.load(tmp_txt)
    assert docs[0].doc_id is not None
    assert len(docs[0].doc_id) == 12
