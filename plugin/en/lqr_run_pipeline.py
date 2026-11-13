"""Worker for LocalQuadRetopo's ZBrush button. Publish the status last."""
import argparse
from collections import Counter, defaultdict
import os
import re
from pathlib import Path
from types import SimpleNamespace

import lqr_validate as validation

BEFORE = Path("lqr_before.obj")
AFTER = Path("lqr_after.obj")
SELECTION = Path("lqr_selection.obj")
RECOVERED = Path("lqr_recovered.obj")
ROUNDTRIP = Path("lqr_roundtrip.obj")
NATIVE_REPORT = Path("lqr_native_report.txt")
REPORT = Path("lqr_report.txt")
PREFIX = "lqr_"


def artifact(name):
    return Path(PREFIX + name)


def use_prefix(prefix):
    global PREFIX, BEFORE, AFTER, SELECTION, RECOVERED, ROUNDTRIP, NATIVE_REPORT, REPORT
    PREFIX = prefix
    BEFORE = artifact("before.obj")
    AFTER = artifact("after.obj")
    SELECTION = artifact("selection.obj")
    RECOVERED = artifact("recovered.obj")
    ROUNDTRIP = artifact("roundtrip.obj")
    NATIVE_REPORT = artifact("native_report.txt")
    REPORT = artifact("report.txt")


def publish(path, text):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(text, encoding="ascii", errors="replace")
    os.replace(pending, path)
    if path == REPORT and PREFIX != "lqr_":
        latest = Path("lqr_report.txt")
        latest_pending = artifact("latest.pending")
        try:
            latest_pending.write_text(text, encoding="ascii", errors="replace")
            os.replace(latest_pending, latest)
        except OSError:
            pass


def write_failure(native_report, reason):
    details = native_report.splitlines()[1:]
    publish(REPORT, "LQR_STATUS:FAIL\nLocalQuadRetopo 1.4.3\n"
            + "[FAIL] " + reason + "\n" + "\n".join(details) + "\n")


def exact_signature(vertices, faces):
    # Match cyclic vertex order, ignoring only the index of the first corner.
    return Counter(min(points[i:] + points[:i] for i in range(len(points)))
                   for f in faces for points in [tuple(vertices[n] for n in f)])


def verify_roundtrip(vertices, faces, imported_vertices, imported_faces):
    from scipy.spatial import cKDTree

    if len(imported_vertices) != len(vertices):
        raise ValueError("ZBrush round trip altered the vertex count")
    distance, source_index = cKDTree(vertices).query(imported_vertices)
    tolerance = validation.bbox_scale(vertices) * 1e-6
    if distance.max() > tolerance or len(set(source_index)) != len(vertices):
        raise ValueError("ZBrush round trip shifted or merged vertices")
    ids = list(range(len(vertices)))
    if exact_signature(ids, faces) != exact_signature(source_index, imported_faces):
        raise ValueError("ZBrush round trip altered winding or face layout")
    return source_index


def recovery_scope(report):
    patch = re.search(r"^Selected patch: [0-9]+ source faces -> ([0-9]+) quads$", report, re.M)
    outside = re.search(r"^Outside transition faces: ([01])$", report, re.M)
    if not patch or not outside:
        raise ValueError("Recovery report is missing its validated face scope")
    return int(patch[1]), int(outside[1])


def remaining_faces(vertices, faces, preserved):
    remaining = preserved.copy()
    active = set()
    for i, face in enumerate(faces):
        points = tuple(vertices[n] for n in face)
        key = min(points[j:] + points[:j] for j in range(len(points)))
        if remaining[key]:
            remaining[key] -= 1
        else:
            active.add(i)
    return active, any(remaining.values())


def open_edges(vertices, faces):
    edges = Counter(tuple(sorted((a, b))) for face in faces
                    for a, b in zip(face, face[1:] + face[:1]))
    return Counter(tuple(sorted((vertices[a], vertices[b])))
                   for (a, b), count in edges.items() if count == 1)


def seam_band(vertices, faces, active):
    edges = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edges[tuple(sorted((a, b)))].append(i)
    current = set()
    adjacent = defaultdict(set)
    for owners in edges.values():
        if len(owners) != 2:
            continue
        a, b = owners
        if (a in active) != (b in active):
            current.add(a if a in active else b)
        elif a in active:
            adjacent[a].add(b)
            adjacent[b].add(a)
    band = set()
    for _ in range(8):
        band.update(current)
        current = {next_face for face in current for next_face in adjacent[face]} - band
        if not current:
            break
    ratios = []
    for i in band:
        face = faces[i]
        if len(face) != 4:
            continue
        longest = max(validation.vlen(validation.vsub(vertices[a], vertices[b]))
                      for a, b in zip(face, face[1:] + face[:1]))
        ratios.append(validation.polygon_area(vertices, face) / longest ** 2 if longest else 0.0)
    return len(ratios), sum(r < 0.23 for r in ratios), sum(r < 0.02 for r in ratios)


def verified_selection():
    if not SELECTION.exists():
        return None
    bv, bf, _ = validation.parse_obj(BEFORE)
    sv, sf, _ = validation.parse_obj(SELECTION)
    if not 0 < len(sf) <= len(bf):
        return None
    before = exact_signature(bv, bf)
    if exact_signature(sv, sf) - before:
        return None
    return before - exact_signature(sv, sf)


def recommend():
    """Mode 3: suggest a face target for the masked area. AFTER holds the mask export."""
    from lqr_recover import identify, read_obj

    vertices, faces, _ = read_obj(BEFORE)
    mask_vertices, mask_faces, _ = read_obj(AFTER)
    selected = identify(vertices, faces, mask_vertices, mask_faces, True)
    if not selected:
        raise ValueError("The mask does not cover any face of this mesh")

    def quads(face):
        # A triangle counts as half a quad, so triangle soup is not over-rated.
        return 0.5 if len(face) == 3 else 1.0

    inside_area = sum(validation.polygon_area(vertices, faces[i]) for i in selected)
    outside = [f for i, f in enumerate(faces) if i not in selected]
    outside_area = sum(validation.polygon_area(vertices, f) for f in outside)
    if outside_area > 0:
        density = sum(map(quads, outside)) / outside_area
    else:
        density = sum(quads(faces[i]) for i in selected) / inside_area
    count = int(min(5000, max(100, round(inside_area * density / 10.0) * 10)))
    # R1: the masked surface passes through itself heavily. ZBrush crashed
    # inside ZRemesher on such a patch, so the plugin then skips ZRemesher.
    from lqr_recover import source_intersections
    # The one crash seen had 7 crossing pairs in 289 masked faces; five other
    # masks with up to 44 pairs in 2501 faces went through ZRemesher safely.
    # The line is drawn between the two: more than 2% of the masked faces.
    risk = 1 if source_intersections(vertices, faces, selected) * 50 > len(selected) else 0
    publish(REPORT, f"LQR_STATUS:RECOMMEND {count:05d} R{risk}\n"
            f"Masked faces: {len(selected)}; masked area share: "
            f"{inside_area / (inside_area + outside_area):.3f}; unmasked faces: {len(outside)}\n")


def run(args):
    if args.mode == 3:
        recommend()
        return
    if args.phase == 2:
        native = NATIVE_REPORT.read_text(encoding="ascii")
        try:
            from lqr_recover import check_candidate, read_obj

            cv, cf, _ = read_obj(RECOVERED)
            rv, rf, _ = read_obj(ROUNDTRIP)
            vertex_map = verify_roundtrip(cv, cf, rv, rf)
            recovered_report = artifact("recovered_report.txt").read_text(encoding="ascii")
            count, transition = recovery_scope(recovered_report)
            stage = artifact("roundtrip_report.txt")
            # ZBrush stores float32 and may start a face at another corner, so
            # the re-export is never compared by coordinates. The vertex map
            # from verify_roundtrip identifies the patch faces exactly; they
            # are then checked in the layout ZBrush actually wrote.
            outside_count = len(cf) - count
            outside = exact_signature(list(range(len(cv))), cf[:outside_count])
            active, missing = remaining_faces(vertex_map, rf, outside)
            if missing or len(active) != count:
                raise ValueError("Re-export changed the validated patch membership")
            inherited = folds = 0
            if SELECTION.is_file():
                from lqr_recover import identify, source_folds, source_intersections
                bv, bf, _ = read_obj(BEFORE)
                sv, sf, _ = read_obj(SELECTION)
                chosen = identify(bv, bf, sv, sf, True)
                inherited = source_intersections(bv, bf, chosen)
                folds = source_folds(bv, bf, chosen)
            report = check_candidate(rv, rf, active, {"intersect": True, "strict_seam": True,
                                                      "inherited_intersections": inherited,
                                                      "inherited_folds": folds})
            publish(stage, "LQR_STATUS:PASS\nExact round-trip patch and shared boundary verified.\n"
                    + str(report["info"]) + "\n")
            # Keep ZRemesher's own output for later inspection.
            os.replace(AFTER, artifact("native.obj"))
            os.replace(ROUNDTRIP, AFTER)
            publish(REPORT, "LQR_STATUS:PASS\nLocalQuadRetopo 1.4.3 - recovered and checked after ZBrush re-export\n"
                    + "\n".join(recovered_report.splitlines()[1:]) + "\n")
        except (OSError, ValueError, ImportError, IndexError) as exc:
            write_failure(native, f"Recovery could not be applied: {exc}")
        return

    report_path = NATIVE_REPORT if args.phase == 1 else artifact("recheck_report.txt")
    result = validation.main(["--before", str(BEFORE), "--after", str(AFTER),
                              "--strict-seam", "--report", str(report_path),
                              "--max-changed-ratio", str(args.ratio)])
    native = report_path.read_text(encoding="ascii") if report_path.exists() else ""
    preserved = verified_selection() if args.phase == 1 else None
    if args.phase == 1:
        bv, bf, _ = validation.parse_obj(BEFORE)
        av, af, _ = validation.parse_obj(AFTER)
        if preserved is not None:
            source_active, _ = remaining_faces(bv, bf, preserved)
            active, missing = remaining_faces(av, af, preserved)
        else:
            tolerance = validation.bbox_scale(bv) * 1e-6
            source_active = validation.active_face_set(av, af, bv, bf, tolerance)
            active = validation.active_face_set(bv, bf, av, af, tolerance)
            missing = False
        # If the native path changed faces beyond the explicit mask, use the
        # observed native boundary for diagnostics only, never for recovery.
        if missing:
            tolerance = validation.bbox_scale(bv) * 1e-6
            native_scope = validation.active_face_set(bv, bf, av, af, tolerance)
        else:
            native_scope = active
        old_total, old_thin, old_extreme = seam_band(bv, bf, source_active)
        new_total, new_thin, new_extreme = seam_band(av, af, native_scope)
        seam_summary = (f"Source seam band: {old_thin}/{old_total} narrow quads; "
                        f"{old_extreme} extremely narrow. Native seam band: "
                        f"{new_thin}/{new_total} narrow quads; {new_extreme} extremely narrow.")
        native += "[INFO] " + seam_summary + "\n"
        issue = ""
        if exact_signature(bv, bf) == exact_signature(av, af):
            issue = "ZRemesher returned unchanged geometry"
        elif open_edges(bv, bf) != open_edges(av, af):
            issue = "ZRemesher changed the original open boundary or created an unwelded seam"
        elif missing:
            issue = "ZRemesher changed faces outside the original selection"
        elif any(len(af[i]) != 4 for i in active):
            issue = "The selected area still contains non-quad faces"
        elif (new_extreme > old_extreme or (new_thin >= 12 and
                new_thin / max(new_total, 1) > old_thin / max(old_total, 1) + 0.10)):
            issue = "Native result created a band of thin or folded quads"
        if issue:
            result = 1
            native = native.replace("LQR_STATUS:PASS", "LQR_STATUS:FAIL", 1)
            native = native.replace("RESULT: PASS", "RESULT: FAIL")
            native += f"[FAIL] {issue}\n"
        publish(NATIVE_REPORT, native)
    if result == 0 or args.phase == 0:
        publish(REPORT, native or "LQR_STATUS:FAIL\nValidation did not produce a report.\n")
        return
    try:
        from lqr_recover import recover
        from recovery_geometry import extreme_faces

        # Mode 0: honeycomb layout with the given face target.
        # Mode 1: regular layout (ZRemesher's block plus a transition band);
        #         a target below 0.0995 means automatic.
        # Mode 2: honeycomb layout, target from the surrounding density.
        regular = args.mode == 1
        automatic = args.mode == 2 or (regular and args.target_k < 0.0995)
        target = None if automatic else max(1, round(1000 * args.target_k))
        config = SimpleNamespace(before=BEFORE, selection=SELECTION if preserved is not None else None,
                                 scope_from=None if preserved is not None else AFTER, target_faces=target,
                                 max_changed_ratio=args.ratio, native=AFTER,
                                 style="grid" if regular and preserved is not None else "honeycomb")
        obj, detail = recover(config)
        publish(RECOVERED, obj)
        cv, cf, _ = validation.parse_obj(RECOVERED)
        patch_count, transition = recovery_scope(detail)
        if preserved is not None and sum((preserved - exact_signature(cv, cf)).values()) != transition:
            raise ValueError("Recovery modified more faces outside the selection than the approved transition")
        recovered_scope = set(range(len(cf) - patch_count, len(cf)))
        count, thin, extreme = seam_band(cv, cf, recovered_scope)
        detail += (seam_summary + "\n"
                   f"Recovered seam band: {thin}/{count} narrow quads; {extreme} extremely narrow.\n")
        source_extreme = len(extreme_faces(bv, [bf[i] for i in source_active]))
        output_extreme = len(extreme_faces(cv, [cf[i] for i in recovered_scope]))
        if output_extreme > source_extreme:
            raise ValueError("Recovery increased the number of extremely narrow faces")
        publish(artifact("recovered_report.txt"), detail)
        publish(REPORT, "LQR_STATUS:IMPORT\nLocalQuadRetopo 1.4.3\n"
                "Recovered patch passed offline validation. Pending ZBrush import check.\n")
    except ImportError:
        write_failure(native, "Recovery requires NumPy and SciPy in the configured Python interpreter")
    except (ValueError, OSError, IndexError) as exc:
        write_failure(native, f"Automatic recovery rejected this candidate: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", type=int, choices=(0, 1, 2))
    parser.add_argument("ratio", type=float)
    parser.add_argument("mode", type=float)
    parser.add_argument("target_k", type=float)
    parser.add_argument("prefix", nargs="?", default="lqr_")
    args = parser.parse_args()
    if not re.fullmatch(r"lqr[-+0-9.eE_]*_", args.prefix):
        parser.error("Invalid job filename prefix")
    use_prefix(args.prefix)
    if not 0 < args.ratio <= 1 or not 0.09 <= args.target_k <= 100.01 or not -0.01 <= args.mode <= 3.01:
        parser.error("Invalid local ratio, density mode or face target")
    args.mode = round(args.mode)
    try:
        if not BEFORE.is_file() or not AFTER.is_file():
            raise ValueError("A current before and after OBJ export is required")
        run(args)
    except (OSError, ValueError, IndexError) as exc:
        write_failure("", str(exc))


if __name__ == "__main__":
    main()

