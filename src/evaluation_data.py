"""Load a training corpus and its frozen five-fold evaluation inputs."""

import json

from prepare_data import ROOT, protocol_path, read_protocol, read_training


def load_evaluation_data(protocol_id, assets, root=ROOT):
    directory = protocol_path(root, protocol_id)
    descriptor = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if descriptor.get("input_kind") == "open_pool":
        from prepare_open_pool import load_open_pool_protocol
        source = root / descriptor["source_relative_path"]
        data, manifest, folds = load_open_pool_protocol(directory, source)
    else:
        source = assets["official_train"]
        data, train_sha256 = read_training(source)
        manifest, folds = read_protocol(directory, data, train_sha256)
    return source, data, manifest, folds
