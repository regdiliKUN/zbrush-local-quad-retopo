"""Polypaint and mask (ZBrush #MRGB block) for LocalQuadRetopo.

ZBrush writes one MMRRGGBB value (8 hex digits; MM = mask, then red, green,
blue) per vertex, in vertex order, up to 64 values per "#MRGB " line, in a
block right after the vertex lines. ZBrush reads the same block back on
import, so a result OBJ that carries it keeps the SubTool's polypaint and mask.

The worker never edits colours. Vertices it keeps keep their own value; a
vertex it creates takes the value of the nearest original vertex.
"""
import numpy as np

HEADER = (b"# The following MRGB block contains ZBrush Vertex Color (Polypaint) and masking output "
          b"as 4 hexadecimal values per vertex. The vertex color format is MMRRGGBB with up to 64 "
          b"entries per MRGB line.")
FOOTER = b"# End of MRGB block "
PER_LINE = 64
_HEX = np.zeros(256, dtype=bool)
_HEX[np.frombuffer(b"0123456789abcdefABCDEF", dtype=np.uint8)] = True


def is_block_line(line):
    """True for the #MRGB lines and the two comment lines ZBrush puts around them."""
    if isinstance(line, str):
        line = line.encode("ascii", "replace")
    line = line.lstrip()
    return (line.startswith(b"#MRGB") or line.startswith(b"# The following MRGB block")
            or line.startswith(b"# End of MRGB block"))


def parse(lines, vertex_count):
    """Values of the #MRGB block as an array of 8-byte strings, or None if there is none.

    Raises ValueError when the block is present but does not give exactly one
    valid value per vertex; the caller must not guess colours then.
    """
    chunks = []
    for line in lines:
        if isinstance(line, str):
            line = line.encode("ascii", "replace")
        line = line.strip()
        if line.startswith(b"#MRGB"):
            chunks.append(line[5:].strip())
    if not chunks:
        return None
    data = b"".join(chunks)
    count = len(data) // 8
    if len(data) % 8 or count != vertex_count:
        raise ValueError(f"The polypaint block lists {count} vertices but the mesh has {vertex_count}; "
                         "the colours cannot be matched to the vertices")
    if not _HEX[np.frombuffer(data, dtype=np.uint8)].all():
        raise ValueError("The polypaint block contains values that are not hexadecimal")
    return np.frombuffer(data, dtype="S8").copy()


def block(values):
    """The #MRGB block (header, data lines, footer) for the values, as bytes lines."""
    values = np.asarray(values, dtype="S8")
    out = [HEADER]
    for start in range(0, len(values), PER_LINE):
        out.append(b"#MRGB " + values[start:start + PER_LINE].tobytes())
    out.append(FOOTER)
    return out


def nearest(source_points, source_values, points):
    """Value of the nearest source point for each point."""
    from scipy.spatial import cKDTree

    if not len(points):
        return np.zeros(0, dtype="S8")
    _, index = cKDTree(np.asarray(source_points, dtype=np.float64)).query(
        np.asarray(points, dtype=np.float64))
    return np.asarray(source_values, dtype="S8")[index]


def compare(expected, actual):
    """(same, different) counts of two equal-length value arrays."""
    same = int(np.count_nonzero(np.asarray(expected) == np.asarray(actual)))
    return same, len(expected) - same


def roundtrip_note(expected, actual, mapping):
    """Report line on the polypaint after ZBrush's import and re-export.

    expected: values of the written result (or None); actual: values of the
    re-export (or None); mapping: for each re-exported vertex its index in
    the written result. Informational only: geometry is checked elsewhere.
    """
    if expected is None:
        return None
    if actual is None:
        return ("[WARN] Polypaint: the result was written with polypaint, but ZBrush's re-export "
                "has none. Check that Colorize is on in Tool > Polypaint.")
    same, different = compare(np.asarray(expected)[np.asarray(mapping)], actual)
    if different == 0:
        return f"Polypaint and mask kept on all {same} vertices after the ZBrush import."
    return (f"[WARN] Polypaint: {different} of {same + different} vertices came back from ZBrush "
            "with a different colour or mask value.")
