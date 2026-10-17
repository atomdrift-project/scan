"""Generate the tiny, constant-benign ONNX tree used by cli_smoke (stdlib only)."""

from pathlib import Path
import struct


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    return bytes(result) + bytes([value])


def number(tag, value):
    return varint(tag << 3) + varint(value)


def field(tag, value):
    if isinstance(value, str):
        value = value.encode()
    return varint((tag << 3) | 2) + varint(len(value)) + value


def integers(name, values):
    return field(5, field(1, name) + number(20, 7) + field(8, b"".join(map(varint, values))))


def floats(name, values):
    return field(5, field(1, name) + number(20, 6) + field(7, struct.pack(f"<{len(values)}f", *values)))


def tensor(name, element, dimensions):
    shape = b"".join(field(1, number(1, dim)) for dim in dimensions)
    return field(1, name) + field(2, field(1, number(1, element) + field(2, shape)))


node = field(1, "input") + field(2, "label") + field(2, "probabilities")
node += field(4, "TreeEnsembleClassifier") + field(7, "ai.onnx.ml")
node += integers("classlabels_int64s", [0, 1])
for name in ("nodes_featureids", "nodes_nodeids", "nodes_treeids", "nodes_truenodeids", "nodes_falsenodeids"):
    node += integers(name, [0])
node += floats("nodes_values", [0])
node += field(5, field(1, "nodes_modes") + number(20, 8) + field(9, "LEAF"))
for name in ("class_nodeids", "class_treeids"):
    node += integers(name, [0, 0])
node += integers("class_ids", [0, 1]) + floats("class_weights", [1, 0])
graph = field(1, node) + field(2, "cli-smoke") + field(11, tensor("input", 1, [1, 1]))
graph += field(12, tensor("label", 7, [1])) + field(12, tensor("probabilities", 1, [1, 2]))
model = number(1, 8) + field(2, "cli-smoke") + field(7, graph)
model += field(8, number(2, 13)) + field(8, field(1, "ai.onnx.ml") + number(2, 3))
Path(__file__).with_name("cli-model.onnx").write_bytes(model)
