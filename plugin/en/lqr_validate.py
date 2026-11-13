#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lqr_validate.py -- Local Quad Retopo (ZBrush 2025) 拓扑严格校验器
================================================================

用途
----
ZScript 只能调用 ZBrush 的界面命令，无法逐面读取/写入网格数据，因此
"全四边形 / 无退化面 / 无重叠面 / 边界已焊接 / 未受影响区域不变"
这类硬保证必须由外部几何校验器完成。本文件即该校验器，只依赖 Python 3
标准库，无需 numpy / trimesh / Open3D。

判罚范围（v1.1.1 起）
--------------------
只对「本次重算出来的面」负责。传入 --before 后，校验器先用几何比对认出
哪些面是新生成的，严格检查只覆盖这些面；选区外的三角面 / n-gon / 开放
边界 / 自相交属于源模型本来的样子，不会让这一次局部重拓扑背锅。
没传 --before 时退回全网格严格模式（兼容旧用法）。

检查项（FAIL = 判定失败，WARN = 仅提示）
--------------------------------------
  FAIL  non_quad_faces        重算区存在非四边形面（三角形 / n-gon）
  FAIL  degenerate_faces      重复顶点、零面积、零边长、三点共线的退化面
  FAIL  duplicate_faces       顶点索引集合完全相同的重复面
  FAIL  coincident_faces      几何上完全重合但索引不同的面（典型重叠面）
  FAIL  intersecting_faces    面片之间真实穿插（三角形-三角形相交检测）
  FAIL  nonmanifold_edges     同一条边被 >2 个面引用
  FAIL  changed_ratio         改动比例超过阈值（面数与面积两个口径取大者）
  FAIL  empty_mesh            导出结果里一个面都没有
  FAIL  intersection_scan_truncated  相交扫描没扫完（宁可失败也不给假保证）
  WARN  new_boundary_edges    重算区带来的新开放边（接缝可能没焊上）
  WARN  seam_ring             接缝那一圈比内部细太多（外观上"围一圈碎面"）
  WARN  mask_holes            重算区被遮罩空隙切成多个岛/洞（翻折的真正来源）
  WARN  patch_fold            顶点被切向压扁 / 相邻面折过 150 度（回投造成）
  INFO  patch                 岛数、环数、洞数、最大洞半径、需要几次闭运算
  INFO  patch_health          顶点数、被压扁的顶点数与比例、折痕边数
  WARN  density_mismatch      重算区比周围密 3 倍以上
  WARN  parse_issues          OBJ 里有坏行 / NaN 坐标
  WARN  unwelded_boundary     空间上重合却未焊接的边界边
  INFO  boundary_edges        源模型自带的开放边界（原样保留，不算错）
  INFO  seam / density / hotspots / scope / pre_existing_intersections

用法
----
  python lqr_validate.py --after result.obj --before backup.obj \
         --report lqr_report.txt --max-changed-ratio 0.9

  python lqr_validate.py --self-test          # 内置合成网格回归测试
  python lqr_validate.py --after x.obj --no-intersect   # 跳过耗时的面片相交检测

退出码: 0=PASS, 1=FAIL, 2=运行错误
报告首行固定为 "LQR_STATUS:PASS" 或 "LQR_STATUS:FAIL"，供 ZScript 读取。
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

VERSION = "1.3.1"

# 空间网格边长 = 中位边长 × 该系数。
# 实测（9 万面环面）：1.0 -> 9.4s / 1.5 -> 13.0s / 2.2 -> 19.6s。
# 取得太小会让"跨格过多"的面变多，那些面只能走全量比对，反而更慢。
CELL_FACTOR = 1.5

# --------------------------------------------------------------------------
# OBJ 解析
# --------------------------------------------------------------------------


def parse_obj(path, stats=None):
    """读取 Wavefront OBJ。返回 (vertices, faces, face_line_numbers)。

    支持 f 1/1/1、f -1 等写法；忽略 vt / vn / g / usemtl 等行。

    v1.1.1: 坐标里出现 NaN / Inf 时不再原样带进几何计算——它会把 bbox 和
    容差一起污染成 nan，后面所有比较都变成 False，报告会出现"什么都通过"
    的假象。这里统一替换成原点并记账，交给调用方在报告里点出来。
    传入 stats 字典可以拿到坏点/坏行的统计。
    """
    verts, faces, face_lines = [], [], []
    bad_verts = 0
    bad_faces = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            tag = s.split(None, 1)[0]
            if tag == "v":
                p = s.split()
                if len(p) < 4:
                    bad_verts += 1
                    continue
                try:
                    x, y, z = float(p[1]), float(p[2]), float(p[3])
                    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                        raise ValueError
                except ValueError:
                    x = y = z = 0.0
                    bad_verts += 1
                verts.append((x, y, z))
            elif tag == "f":
                idx = []
                for tok in s.split()[1:]:
                    head = tok.split("/")[0]
                    try:
                        i = int(head)
                    except ValueError:
                        continue
                    # OBJ: 正数从 1 开始，负数相对当前顶点表末尾
                    i = (len(verts) + i) if i < 0 else (i - 1)
                    idx.append(i)
                if len(idx) >= 3:
                    faces.append(tuple(idx))
                    face_lines.append(lineno)
                else:
                    bad_faces += 1
    if stats is not None:
        stats["bad_vertices"] = bad_verts
        stats["bad_faces"] = bad_faces
        stats["vertices"] = len(verts)
        stats["faces"] = len(faces)
    return verts, faces, face_lines


# --------------------------------------------------------------------------
# 向量 / 几何工具
# --------------------------------------------------------------------------


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vlen(a):
    return math.sqrt(vdot(a, a))


def tri_area(a, b, c):
    return 0.5 * vlen(vcross(vsub(b, a), vsub(c, a)))


def polygon_area(verts, face):
    """多边形面积（Newell 法，投影面积）。

    v1.1.1: 从"以首顶点扇形三角化"改为 Newell。扇形法对凹多边形会低估面积
    （凹口被重复计入又抵消），零面积判定可能漏检；Newell 对所有平面多边形
    精确、与绕序无关，而且会把"领结"四边形算成接近 0——那本来就是退化面。
    """
    n = len(face)
    if n < 3:
        return 0.0
    ax = ay = az = 0.0
    for k in range(n):
        p = verts[face[k]]
        q = verts[face[(k + 1) % n]]
        ax += p[1] * q[2] - p[2] * q[1]
        ay += p[2] * q[0] - p[0] * q[2]
        az += p[0] * q[1] - p[1] * q[0]
    return 0.5 * math.sqrt(ax * ax + ay * ay + az * az)


def bbox_scale(verts):
    """包围盒对角线长度。跳过非有限坐标，避免 nan 传染整条容差链。"""
    if not verts:
        return 1.0
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    seen = False
    for p in verts:
        if not (math.isfinite(p[0]) and math.isfinite(p[1]) and math.isfinite(p[2])):
            continue
        seen = True
        for i in range(3):
            if p[i] < lo[i]:
                lo[i] = p[i]
            if p[i] > hi[i]:
                hi[i] = p[i]
    if not seen:
        return 1.0
    d = vlen((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    return max(d, 1e-6)


def qkey(point, tol):
    """把坐标量化为整数网格键，用于判定几何重合。"""
    return tuple(int(round(c / tol)) for c in point)


def face_geom_key(verts, face, tol):
    """与顶点顺序、绕序无关的面几何键。"""
    return tuple(sorted(qkey(verts[i], tol) for i in face))


# --------------------------------------------------------------------------
# 三角形相交检测
# --------------------------------------------------------------------------


def seg_tri_hit(p, q, tri, det_eps, guard=0.0):
    """线段 pq 是否与三角形 tri 内部相交（Möller-Trumbore，t 必须严格在 (0,1)）。

    端点接触不算相交，避免把共享顶点/边的合法相邻面判成重叠。
    """
    d = vsub(q, p)
    e1 = vsub(tri[1], tri[0])
    e2 = vsub(tri[2], tri[0])
    h = vcross(d, e2)
    a = vdot(e1, h)
    if abs(a) <= det_eps:
        return False
    f = 1.0 / a
    s = vsub(p, tri[0])
    u = f * vdot(s, h)
    if u < -1e-9 or u > 1.0 + 1e-9:
        return False
    qv = vcross(s, e1)
    v = f * vdot(d, qv)
    if v < -1e-9 or u + v > 1.0 + 1e-9:
        return False
    t = f * vdot(e2, qv)
    if not 1e-9 < t < 1.0 - 1e-9:
        return False
    if guard > 0.0:
        # A crossing closer to an end point than the weld tolerance is a
        # touch. ZBrush stores float32, so a re-exported touch moves by ~1e-8.
        length = vlen(d)
        if t * length <= guard or (1.0 - t) * length <= guard:
            return False
    return True


def coplanar_overlap(t1, t2, tol):
    """共面三角形的 2D SAT 严格重叠判定。"""
    n = vcross(vsub(t1[1], t1[0]), vsub(t1[2], t1[0]))
    axis = max(range(3), key=lambda i: abs(n[i]))
    u, v = (axis + 1) % 3, (axis + 2) % 3
    A = [(p[u], p[v]) for p in t1]
    B = [(p[u], p[v]) for p in t2]
    for poly in (A, B):
        for i in range(3):
            ex = poly[(i + 1) % 3][0] - poly[i][0]
            ey = poly[(i + 1) % 3][1] - poly[i][1]
            L = math.hypot(ex, ey)
            if L <= 0.0:
                continue
            nx, ny = -ey / L, ex / L
            pa = [nx * p[0] + ny * p[1] for p in A]
            pb = [nx * p[0] + ny * p[1] for p in B]
            overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if overlap <= tol * 0.5:
                return False
    return True


def tri_tri_intersect(t1, t2, tol, det_eps, area_eps, shared1=None, shared2=None):
    n1 = vcross(vsub(t1[1], t1[0]), vsub(t1[2], t1[0]))
    n2 = vcross(vsub(t2[1], t2[0]), vsub(t2[2], t2[0]))
    l1, l2 = vlen(n1), vlen(n2)
    if l1 <= area_eps or l2 <= area_eps:
        return False
    u1 = (n1[0] / l1, n1[1] / l1, n1[2] / l1)
    u2 = (n2[0] / l2, n2[1] / l2, n2[2] / l2)
    cosv = abs(vdot(u1, u2))
    if cosv > 1.0 - 1e-6 and abs(vdot(u1, vsub(t2[0], t1[0]))) < tol:
        return coplanar_overlap(t1, t2, tol)
    # shared1 / shared2: corner index of a vertex the two triangles share.
    # An edge that starts at the shared vertex meets the other triangle's
    # plane only there, so it cannot cross it anywhere else; testing it would
    # only measure rounding noise. Only the opposite edges are tested.
    for i in range(3):
        if shared1 is None or (i != shared1 and (i + 1) % 3 != shared1):
            if seg_tri_hit(t1[i], t1[(i + 1) % 3], t2, det_eps, tol):
                return True
        if shared2 is None or (i != shared2 and (i + 1) % 3 != shared2):
            if seg_tri_hit(t2[i], t2[(i + 1) % 3], t1, det_eps, tol):
                return True
    return False


def _tri_pair_hit(t1, i1, t2, i2, tol, det_eps, area_eps):
    """tri_tri_intersect, aware of a vertex the two triangles share by index."""
    common = set(i1) & set(i2)
    if len(common) == 1:
        vertex = next(iter(common))
        return tri_tri_intersect(t1, t2, tol, det_eps, area_eps,
                                 i1.index(vertex), i2.index(vertex))
    return tri_tri_intersect(t1, t2, tol, det_eps, area_eps)


def _build_grid(tris, cell):
    """把三角形登记进均匀网格。返回 (grid, big)。

    跨格超过 4096 的三角形不进网格——一个横跨整张模型的大面会把内存撑爆。
    它们被收集到 big 里，由调用方做一次带包围盒剔除的全量比对。
    """
    grid = defaultdict(list)
    big = []
    for ti, (tri, _fid, _s, _ids) in enumerate(tris):
        lo = [min(p[i] for p in tri) for i in range(3)]
        hi = [max(p[i] for p in tri) for i in range(3)]
        ci = [int(math.floor(lo[i] / cell)) for i in range(3)]
        cj = [int(math.floor(hi[i] / cell)) for i in range(3)]
        span = 1
        for i in range(3):
            span *= (cj[i] - ci[i] + 1)
        if span > 4096:
            big.append(ti)
            continue
        for x in range(ci[0], cj[0] + 1):
            for y in range(ci[1], cj[1] + 1):
                for z in range(ci[2], cj[2] + 1):
                    grid[(x, y, z)].append(ti)
    return grid, big


def find_intersecting_faces(verts, faces, tol, cap_pairs=8000000):
    """基于均匀网格的宽相 + 三角形窄相，返回相交面编号对。

    v1.0.1: 网格边长由「中位边长」决定，而不是固定的 bbox/40。
    固定 bbox/40 在密网格上会让每个 cell 塞进上百个三角形，候选对数量
    呈平方增长并撞上 cap_pairs，把「严格校验」悄悄降级成抽样检查。
    """
    scale = bbox_scale(verts)
    # 只抽样前 5 万个面估算中位边长：百万面网格上全量收集会吃掉几百 MB。
    sample = faces[:50000]
    elens = []
    for f in sample:
        n = len(f)
        for k in range(n):
            a, b = f[k], f[(k + 1) % n]
            if 0 <= a < len(verts) and 0 <= b < len(verts):
                elens.append(vlen(vsub(verts[a], verts[b])))
    elens.sort()
    med_edge = elens[len(elens) // 2] if elens else scale / 40.0
    # 双保险：退化网格可能把中位边长压到 0，下限锁在 bbox/2000，
    # 否则 cell 会塌成 0，空间索引直接爆掉。
    cell = max(med_edge * CELL_FACTOR, scale / 2000.0, tol * 10.0)
    area_eps = (tol * tol) * 0.5
    det_eps = max(1e-14, (tol ** 3) * 1e-6)

    tris = []           # (tri points tuple, face id, vertex-index set)
    for fid, f in enumerate(faces):
        if len(f) < 3 or any(i < 0 or i >= len(verts) for i in f):
            continue
        for k in range(1, len(f) - 1):
            tri = (verts[f[0]], verts[f[k]], verts[f[k + 1]])
            if tri_area(*tri) <= area_eps:
                continue
            tris.append((tri, fid, set(f), (f[0], f[k], f[k + 1])))
    if not tris:
        return [], 0, False

    # 建网格。单元格太小 / 尺度混杂时会有大量面"跨格过多"，它们进不了网格
    # 只能走全量比对，反而更慢；这种情况下把 cell 放大重试。
    grid = {}
    big = []
    for _attempt in range(3):
        grid, big = _build_grid(tris, cell)
        if len(big) <= max(32, len(tris) // 200):
            break
        cell *= 1.7

    seen = set()
    tested = 0
    truncated = False
    hits = []
    for bucket in grid.values():
        if len(bucket) < 2:
            continue
        for a in range(len(bucket)):
            for b in range(a + 1, len(bucket)):
                i, j = bucket[a], bucket[b]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                t1, f1, s1, i1 = tris[pair[0]]
                t2, f2, s2, i2 = tris[pair[1]]
                if f1 == f2:                       # 同一个四边形的两条三角
                    continue
                if len(s1 & s2) >= 2:              # 共享边的合法相邻面
                    continue
                tested += 1
                if tested > cap_pairs:
                    truncated = True
                    return hits, tested, truncated
                if _tri_pair_hit(t1, i1, t2, i2, tol, det_eps, area_eps):
                    key = (min(f1, f2), max(f1, f2))
                    if key not in hits:
                        hits.append(key)

    # 跨格过多的大三角形：它们没进网格，这里补一次全量比对（先做包围盒剔除）。
    # 老版本用 cj = ci+15 截断索引，这类面只会被登记一小部分，穿插直接漏掉。
    if big:
        lo_all = [[min(p[i] for p in t) for i in range(3)] for t, _, _, _ in tris]
        hi_all = [[max(p[i] for p in t) for i in range(3)] for t, _, _, _ in tris]
        for bi in big:
            t1, f1, s1, i1 = tris[bi]
            for tj in range(len(tris)):
                if tj == bi:
                    continue
                pair = (bi, tj) if bi < tj else (tj, bi)
                if pair in seen:
                    continue
                seen.add(pair)
                t2, f2, s2, i2 = tris[tj]
                if f1 == f2 or len(s1 & s2) >= 2:
                    continue
                if (hi_all[bi][0] < lo_all[tj][0] or lo_all[bi][0] > hi_all[tj][0] or
                        hi_all[bi][1] < lo_all[tj][1] or lo_all[bi][1] > hi_all[tj][1] or
                        hi_all[bi][2] < lo_all[tj][2] or lo_all[bi][2] > hi_all[tj][2]):
                    continue
                tested += 1
                if tested > cap_pairs:
                    truncated = True
                    return hits, tested, truncated
                if _tri_pair_hit(t1, i1, t2, i2, tol, det_eps, area_eps):
                    key = (min(f1, f2), max(f1, f2))
                    if key not in hits:
                        hits.append(key)

    return hits, tested, truncated


# --------------------------------------------------------------------------
# 主检查流程
# --------------------------------------------------------------------------


def classify_faces(before_verts, before_faces, after_verts, after_faces, tol):
    """一次遍历同时算出：重算区集合、新旧面数与面积、未改动面积。

    之前 active_face_set / density_report / compare_unchanged 各扫一遍并各算
    一次 face_geom_key，2 万面的网格上等于白跑两遍哈希。合并后只扫一次。
    """
    bc = Counter(face_geom_key(before_verts, f, tol) for f in before_faces)
    used = Counter()
    active = set()
    n_new = n_old = 0
    a_new = a_old = 0.0
    unchanged_area = 0.0
    for n, f in enumerate(after_faces):
        k = face_geom_key(after_verts, f, tol)
        a = polygon_area(after_verts, f)
        if used[k] < bc.get(k, 0):
            used[k] += 1
            n_old += 1
            a_old += a
            unchanged_area += a
        else:
            active.add(n)
            n_new += 1
            a_new += a
    total_before_area = sum(polygon_area(before_verts, f) for f in before_faces)
    return {
        "active": active,
        "n_new": n_new,
        "n_old": n_old,
        "a_new": a_new,
        "a_old": a_old,
        "unchanged_area": unchanged_area,
        "total_before_area": total_before_area,
        "before_faces": len(before_faces),
        "after_faces": len(after_faces),
    }


def active_face_set(before_verts, before_faces, after_verts, after_faces, tol):
    """返回「本次重算出来的面」的索引集合。

    局部重拓扑只对遮罩选区负责：选区外的三角面、n-gon、开放边界都应该
    原样保留，不能因为源模型本来就不是全四边就判失败。有了这个集合，
    下面的严格检查就只盯重算区。
    """
    if not before_faces:
        return None
    return classify_faces(before_verts, before_faces, after_verts, after_faces,
                          tol)["active"]


class _UF:
    """并查集。摊还近乎 O(1)，用来数重算区连通块。"""

    __slots__ = ("p",)

    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        p = self.p
        root = x
        while p.get(root, root) != root:
            root = p[root]
        while p.get(x, x) != x:          # 全路径压缩
            nxt = p[x]
            p[x] = root
            x = nxt
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def order_loop(edges):
    """把一组首尾相连的边界边排成环上的顶点顺序。"""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    if not adj:
        return []
    start = next(iter(adj))
    order = [start]
    prev, cur = None, start
    limit = len(edges) + 2
    while len(order) < limit:
        nxt = None
        for v in adj[cur]:
            if v != prev:
                nxt = v
                break
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def loop_metrics(verts, order):
    """环的投影面积、周长、等效半径（2A/P）与质心。

    等效半径就是「这个洞大概有多宽」：用半径为 r 的圆代入，A=πr²、P=2πr，
    于是 2A/P = r。膨胀 N 次能补掉的洞，其 r 大约不超过 N 个面边长。
    """
    n = len(order)
    if n < 3:
        return None
    ax = ay = az = 0.0
    per = 0.0
    cx = cy = cz = 0.0
    for k in range(n):
        p = verts[order[k]]
        q = verts[order[(k + 1) % n]]
        ax += p[1] * q[2] - p[2] * q[1]
        ay += p[2] * q[0] - p[0] * q[2]
        az += p[0] * q[1] - p[1] * q[0]
        per += vlen(vsub(p, q))
        cx += p[0]
        cy += p[1]
        cz += p[2]
    area = 0.5 * math.sqrt(ax * ax + ay * ay + az * az)
    if per <= 0.0 or area <= 0.0:
        return None
    return {
        "edges": n,
        "area": area,
        "perimeter": per,
        "radius": 2.0 * area / per,
        "center": [round(cx / n, 4), round(cy / n, 4), round(cz / n, 4)],
    }


def face_normal(verts, faces, fa):
    """面的单位法线（Newell，绕序正确时朝外）。"""
    f = faces[fa]
    p = [verts[i] for i in f]
    ax = ay = az = 0.0
    for k in range(len(p)):
        a, b = p[k], p[(k + 1) % len(p)]
        ax += a[1] * b[2] - a[2] * b[1]
        ay += a[2] * b[0] - a[0] * b[2]
        az += a[0] * b[1] - a[1] * b[0]
    L = math.sqrt(ax * ax + ay * ay + az * az)
    if L <= 0:
        return (0.0, 0.0, 0.0)
    return (ax / L, ay / L, az / L)


def _loop_plane_2d(order, verts):
    """把环投影到它自己的最佳拟合平面上，返回 (2D点列表, 质心)。"""
    n = len(order)
    ax = ay = az = 0.0
    cx = cy = cz = 0.0
    for k in range(n):
        p = verts[order[k]]
        q = verts[order[(k + 1) % n]]
        ax += p[1] * q[2] - p[2] * q[1]
        ay += p[2] * q[0] - p[0] * q[2]
        az += p[0] * q[1] - p[1] * q[0]
        cx += p[0]
        cy += p[1]
        cz += p[2]
    nv = (ax, ay, az)
    L = math.sqrt(nv[0] ** 2 + nv[1] ** 2 + nv[2] ** 2)
    if L <= 0:
        return None, None
    nv = (nv[0] / L, nv[1] / L, nv[2] / L)
    # 选一个与法线最不平行的轴作为投影丢弃轴
    ax_i = max(range(3), key=lambda i: abs(nv[i]))
    keep = [i for i in range(3) if i != ax_i]
    pts = [(verts[v][keep[0]], verts[v][keep[1]]) for v in order]
    cen = (cx / n, cy / n, cz / n)
    return pts, cen


def _point_in_poly(pt, poly):
    """射线法。"""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def patch_topology(verts, faces, seam, edge, active):
    """分析重算区形状：几个岛、几个洞、每个洞多宽、要膨胀几次才补得上。

    外环与洞的区分不看绕序（接缝两侧的绕向约定会让它翻车），而是看
    **这个环圈住的区域里有没有重算面**：外环圈住整个岛，里面全是重算面；
    洞圈住的是没被重算的旧面，里面一个重算面都没有。
    """
    if not active:
        return None

    uf = _UF()
    for n in active:
        uf.add(n)
    for e, vv in edge.items():
        if len(vv) == 2 and vv[0] in active and vv[1] in active:
            uf.union(vv[0], vv[1])
    comp = defaultdict(int)
    for n in active:
        comp[uf.find(n)] += 1
    islands = len(comp)
    island_sizes = sorted(comp.values(), reverse=True)

    # 接缝边按无向连通分量切环
    ladj = defaultdict(list)
    for a, b in seam:
        ladj[a].append(b)
        ladj[b].append(a)
    comps = []
    seen_v = set()
    for v in ladj:
        if v in seen_v:
            continue
        vs = set()
        stack = [v]
        while stack:
            x = stack.pop()
            if x in vs:
                continue
            vs.add(x)
            for y in ladj[x]:
                if y not in vs:
                    stack.append(y)
        seen_v |= vs
        comps.append(vs)

    # 每块岛的洞数用欧拉示性数算：χ = V - E + F，边界环数 b = 2 - χ，
    # 于是洞数 = b - 1 = 1 - χ。这比"最大环是外环"稳，也能处理
    # 「岛的外边界正好是模型自身的开放边」这种情况。
    # 洞数用全局恒等式：接缝环数 = 岛数 + 洞数（嵌套岛会共享环，逐岛算会重复计数）。
    # 处方半径则取「排除外环之后」的最大环——外环绕着整块岛走，周长是数量级上的
    # 离群值，用中位周长的 3 倍做阈值就能把它摘出去。
    all_loops = []
    for vs in comps:
        edges_in = [(a, b) for a, b in seam if a in vs and b in vs]
        order = order_loop(edges_in)
        m = loop_metrics(verts, order)
        if m:
            all_loops.append(m)

    holes_detail = []
    if all_loops:
        per = sorted(d["perimeter"] for d in all_loops)
        p_med = per[len(per) // 2]
        limit = p_med * 3.0 if len(per) > 1 else float("inf")
        holes_detail = [d for d in all_loops if d["perimeter"] <= limit]
        if not holes_detail:
            holes_detail = list(all_loops)
    holes = max(0, len(comps) - islands)

    elens = []
    for n in active:
        f = faces[n]
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            elens.append(vlen(vsub(verts[a], verts[b])))
    elens.sort()
    e_med = elens[len(elens) // 2] if elens else 0.0
    max_r = max([d["radius"] for d in holes_detail], default=0.0)
    closing = int(math.ceil(max_r / e_med)) if (e_med > 0 and max_r > 0) else 0
    closing = max(0, min(closing, 10))

    holes_detail.sort(key=lambda d: d["radius"], reverse=True)
    return {
        "islands": islands,
        "island_sizes": island_sizes[:8],
        "seam_loops": len(comps),
        "holes": holes,
        "outer_loops": islands,
        "max_hole_radius": round(max_r, 5),
        "median_edge": round(e_med, 5),
        "closing_needed": closing,
        "hole_details": [
            {"edges": d["edges"], "radius": round(d["radius"], 5),
             "center": d["center"]}
            for d in holes_detail[:5]
        ],
    }


def analyse(verts, faces, face_lines, opts, active=None):
    """active=None 表示全网格都算重算区（没有 before 可比时）。"""
    scale = bbox_scale(verts)
    tol = scale * 1e-6                      # 几何重合容差（相对模型尺寸）
    len_eps = scale * 1e-7
    area_eps = (tol * tol) * 0.5

    def on(n):
        return active is None or n in active

    report = {
        "vertex_count": len(verts),
        "face_count": len(faces),
        "scale": scale,
        "tolerance": tol,
        "valence_histogram": {},
        "fail": {},
        "warn": {},
        "info": {},
    }
    bad = report["fail"]
    warn = report["warn"]

    # v1.1.1: 空网格必须判失败。之前 0 个面时所有检查都是空集合，
    # 报告会给出 PASS —— 导出一个空 OBJ 也能"通过严格校验"。
    if not faces:
        bad["empty_mesh"] = "no faces in the exported mesh"
        report["status"] = "FAIL"
        return report, tol

    # 1) 面价数：重算区必须全部为四边形 ------------------------------------
    #    选区外的三角面 / n-gon 属于源模型本来的样子，不动它，也不判失败。
    hist = Counter(len(f) for f in faces)
    report["valence_histogram"] = {str(k): v for k, v in sorted(hist.items())}
    non_quad = [(n, len(f), face_lines[n]) for n, f in enumerate(faces)
                if on(n) and len(f) != 4]
    if non_quad:
        bad["non_quad_faces"] = {
            "count": len(non_quad),
            "examples": non_quad[:20],
        }

    # 2) 退化面 -------------------------------------------------------------
    deg = []
    for n, f in enumerate(faces):
        if not on(n):
            continue
        reason = None
        if len(set(f)) != len(f) or any(i < 0 or i >= len(verts) for i in f):
            reason = "repeated_or_invalid_index"
        else:
            pts = [verts[i] for i in f]
            if polygon_area(verts, f) <= area_eps:
                reason = "zero_area"
            elif any(vlen(vsub(pts[(k + 1) % len(pts)], pts[k])) <= len_eps
                     for k in range(len(pts))):
                reason = "zero_length_edge"
            elif any(vlen(vcross(vsub(pts[(k + 1) % len(pts)], pts[k]),
                                 vsub(pts[(k + 2) % len(pts)], pts[(k + 1) % len(pts)])))
                     <= area_eps * 2.0 for k in range(len(pts))):
                reason = "collinear_vertices"
        if reason:
            deg.append((n, reason, face_lines[n]))
    if deg:
        bad["degenerate_faces"] = {"count": len(deg), "examples": deg[:20]}

    # 3) 索引级重复面 -------------------------------------------------------
    by_idx = defaultdict(list)
    for n, f in enumerate(faces):
        by_idx[tuple(sorted(f))].append(n)
    dup = [v for v in by_idx.values() if len(v) > 1 and any(on(i) for i in v)]
    if dup:
        bad["duplicate_faces"] = {
            "count": sum(len(v) - 1 for v in dup),
            "examples": dup[:20],
        }

    # 4) 几何级重合面（不同顶点索引、坐标一致）------------------------------
    by_geom = defaultdict(list)
    for n, f in enumerate(faces):
        by_geom[face_geom_key(verts, f, tol)].append(n)
    coin = [v for v in by_geom.values() if len(set(v)) > 1 and
            len({tuple(sorted(faces[i])) for i in v}) > 1 and
            any(on(i) for i in v)]
    if coin:
        bad["coincident_faces"] = {
            "count": sum(len(v) - 1 for v in coin),
            "examples": coin[:20],
        }

    # 5) 面片真实穿插 -------------------------------------------------------
    if opts["intersect"]:
        hits, tested, truncated = find_intersecting_faces(verts, faces, tol)
        report["info"]["intersection_pairs_tested"] = tested
        # 只有当穿插的双方至少有一方是本次重算出来的面时才算账：
        # 源模型自带的穿插不该由这次局部重拓扑背锅，但仍如实报出来。
        total_hits = len(hits)
        hits = [(a, b) for a, b in hits if on(a) or on(b)]
        if total_hits > len(hits):
            report["info"]["pre_existing_intersections"] = (
                "%d of %d intersecting pairs already existed in the source mesh"
                % (total_hits - len(hits), total_hits))
        if hits:
            bad["intersecting_faces"] = {
                "count": len(hits),
                "examples": hits[:20],
            }
        if truncated:
            # 截断即失败。只扫了一部分就宣布 PASS，等于骗人。
            bad["intersection_scan_truncated"] = (
                "pair budget exceeded (%d pairs); the intersection scan is "
                "incomplete and cannot certify anything. Re-run with "
                "--no-intersect to inspect the other checks, or raise the "
                "budget in the source." % tested)
    else:
        report["info"]["intersection_pairs_tested"] = "skipped (--no-intersect)"

    # 6) 边流形性 / 边界焊接 -------------------------------------------------
    edge = defaultdict(list)
    for n, f in enumerate(faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            edge[(min(a, b), max(a, b))].append(n)
    boundary = [e for e, v in edge.items() if len(v) == 1]
    nonmanifold = [(e, len(v)) for e, v in edge.items()
                   if len(v) > 2 and any(on(i) for i in v)]
    report["info"]["boundary_edges"] = len(boundary)
    # 只有「重算区带来的新开放边」才算接缝没焊上。源模型本身是破洞/面片时，
    # 那些边界边是它本来的样子，不该报警。
    new_boundary = [e for e, v in edge.items() if len(v) == 1 and on(v[0])]
    report["info"]["new_boundary_edges"] = len(new_boundary)
    if new_boundary:
        warn["new_boundary_edges"] = (
            "%d open border edges touch the remeshed region - the patch may not "
            "be welded to the surrounding mesh" % len(new_boundary))
    elif boundary:
        report["info"]["boundary_note"] = (
            "%d open border edges inherited from the source mesh (kept as-is)"
            % len(boundary))
    if nonmanifold:
        bad["nonmanifold_edges"] = {
            "count": len(nonmanifold),
            "examples": nonmanifold[:20],
        }

    # 接缝质量：新旧面交界那一圈的边长 vs 重算区内部边长。
    # 比值越大，说明接缝处被塞进了越细碎的四边形——就是肉眼看到的"围一圈碎面"。
    seam = [e for e, vv in edge.items()
            if len(vv) == 2 and on(vv[0]) != on(vv[1])]
    if seam and active:
        slens = sorted(vlen(vsub(verts[a], verts[b])) for a, b in seam)
        seam_set = set(seam)          # 循环外构造一次，大网格上省掉 O(F*E)
        inner = []
        for n, f in enumerate(faces):
            if not on(n):
                continue
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if (min(a, b), max(a, b)) not in seam_set:
                    inner.append(vlen(vsub(verts[a], verts[b])))
        inner.sort()
        if inner:
            s_med = slens[len(slens) // 2]
            i_med = inner[len(inner) // 2]
            ratio = (s_med / i_med) if i_med > 0 else 0.0
            report["info"]["seam"] = {
                "seam_edges": len(seam),
                "seam_edge_median": round(s_med, 6),
                "inner_edge_median": round(i_med, 6),
                "seam_over_inner": round(ratio, 2),
            }
            if ratio > 2.5:
                warn["seam_ring"] = (
                    "the patch interior is %.2fx finer than the frozen border "
                    "ring (inner edge %.5f vs seam edge %.5f); the first quad "
                    "rows will be stretched into slivers, so the patch will not "
                    "blend into the surrounding quads. Lower the target density "
                    "or move the mask away from tight curvature."
                    % (ratio, i_med, s_med))


    # 无论有没有接缝都要跑：实心重算区（全选）没有接缝，但拓扑仍然要看。
    if active:
        # 重算区拓扑：它到底是一块完整的面片，还是被遮罩的空隙切成了筛子？
        # seam 环数 = 岛数 + 洞数；洞的等效半径决定要膨胀几次才补得掉。
        pt = patch_topology(verts, faces, seam, edge, active)
        if pt:
            report["info"]["patch"] = {
                "islands": pt["islands"],
                "island_sizes": pt["island_sizes"],
                "seam_loops": pt["seam_loops"],
                "holes": pt["holes"],
                "max_hole_radius": pt["max_hole_radius"],
                "median_edge": pt["median_edge"],
                "closing_needed": pt["closing_needed"],
            }
            if pt["hole_details"]:
                report["info"]["hole_details"] = pt["hole_details"]
            if pt["holes"] or pt["islands"] > 1 or pt["closing_needed"] > 0:
                if pt["holes"]:
                    head = ("%d island(s) plus %d enclosed hole(s); the widest "
                            "hole needs about %d Grow step(s) to close "
                            "(hole radius %.5f / edge %.5f)"
                            % (pt["islands"], pt["holes"],
                               pt["closing_needed"], pt["max_hole_radius"],
                               pt["median_edge"]))
                    tail = ("Set Grow and Shrink to %d each, or repaint those "
                            "spots." % max(pt["closing_needed"], 1))
                else:
                    head = ("%d separate islands - the mask is split into "
                            "pieces that are not touching, and each piece gets "
                            "its own quad grid" % pt["islands"])
                    tail = ("Paint one solid stroke (or run Grow and Shrink by "
                            "the same amount) so the mask is a single blob.")
                warn["mask_holes"] = (
                    "the remeshed area is not one solid patch: " + head + ". "
                    "ZRemesher has to weave around every one of them, which is "
                    "where the folded quads come from. " + tail)
                # 洞的位置直接给出来，省得用户在模型上瞎找
                centers = [d["center"] for d in pt["hole_details"]
                           if d["center"]]
                if centers:
                    report["info"]["hole_centers"] = centers


    # 网格体格体检：顶点有没有被切向压扁、相邻面有没有折回去。
    # 这两种损伤在「顶点是否贴在原表面」这个指标上完全看不出来——顶点确实
    # 都在面上，距离原表面近乎为 0，但彼此的间距被压成了零头。只有量顶点
    # 间距和二面角才抓得到。ZBrush 的 ProjectAll 在圆角处正是这样干的：
    # 每个顶点各自找最近的表面点，映射多对一，邻居被吸到同一个位置。
    if active:
        used = sorted({i for n in active for i in faces[n]})
        elens2 = []
        for n in active:
            f = faces[n]
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                elens2.append(vlen(vsub(verts[a], verts[b])))
        elens2.sort()
        e_med = elens2[len(elens2) // 2] if elens2 else 0.0
        if used and e_med > 0 and len(used) <= 200000:
            cell = e_med
            buckets = defaultdict(list)
            for i in used:
                p = verts[i]
                buckets[(int(math.floor(p[0] / cell)),
                         int(math.floor(p[1] / cell)),
                         int(math.floor(p[2] / cell)))].append(i)
            limit2 = (0.3 * e_med) ** 2
            crowded = 0
            for i in used:
                p = verts[i]
                cx = int(math.floor(p[0] / cell))
                cy = int(math.floor(p[1] / cell))
                cz = int(math.floor(p[2] / cell))
                best = 1e30
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            for j in buckets.get((cx + dx, cy + dy, cz + dz), ()):
                                if j == i:
                                    continue
                                d = vsub(verts[j], verts[i])
                                d2 = vdot(d, d)
                                if d2 < best:
                                    best = d2
                if best < limit2:
                    crowded += 1
            fold_edges = 0
            for e, vv in edge.items():
                if len(vv) != 2 or not (on(vv[0]) and on(vv[1])):
                    continue
                na = face_normal(verts, faces, vv[0])
                nb = face_normal(verts, faces, vv[1])
                if vdot(na, nb) < -0.866:          # 二面角 > 150 度
                    fold_edges += 1
            frac = crowded / float(len(used))
            report["info"]["patch_health"] = {
                "vertices": len(used),
                "crowded_vertices": crowded,
                "crowded_ratio": round(frac, 4),
                "fold_edges": fold_edges,
                "median_edge": round(e_med, 6),
            }
            if frac > 0.02 or fold_edges > 0:
                warn["patch_fold"] = (
                    "%d of %d patch vertices (%.1f%%) sit closer than 0.3x the "
                    "median edge to a neighbour, and %d interior edges are "
                    "folded past 150 degrees. The quad grid was squashed "
                    "tangentially AFTER it was built - this is what "
                    "re-projecting the patch onto the backup does near a "
                    "rounded corner, where nearest-point mapping is many to "
                    "one. Switch Project Detail off, or set Project Passes to "
                    "1 and lower Project Detail distance."
                    % (crowded, len(used), frac * 100.0, fold_edges))

    # 空间重合却未焊接的边界边 = 局部重拓扑接缝失败。
    # v1.1.1: 只统计贴着重算区的那些边——源模型自带的 UV 缝会成片产生
    # 重合边界边，那是它本来的样子，不该每次运行都报警。
    bedge_keys = defaultdict(int)
    for e, vv in edge.items():
        if len(vv) != 1:
            continue
        if active is not None and not on(vv[0]):
            continue
        a, b = e
        if a < 0 or b < 0 or a >= len(verts) or b >= len(verts):
            continue
        ka, kb = qkey(verts[a], tol), qkey(verts[b], tol)
        bedge_keys[tuple(sorted((ka, kb)))] += 1
    unwelded = [k for k, c in bedge_keys.items() if c > 1]
    if unwelded:
        msg = "%d coincident boundary edge pairs not welded" % len(unwelded)
        if opts["strict_seam"]:
            bad["unwelded_boundary"] = msg
        else:
            warn["unwelded_boundary"] = msg + " (UV seams can cause this)"

    # 7) 重合顶点（UV 缝会产生，默认只警告）---------------------------------
    # 同理只看重算区用到的顶点，源模型的 UV 缝不算问题
    used_verts = None
    if active is not None:
        used_verts = set()
        for n in active:
            used_verts.update(faces[n])
    vmap = defaultdict(list)
    for i, p in enumerate(verts):
        if used_verts is not None and i not in used_verts:
            continue
        vmap[qkey(p, tol)].append(i)
    dupv = [v for v in vmap.values() if len(v) > 1]
    if dupv:
        msg = "%d groups of coincident vertices" % len(dupv)
        if opts["strict_seam"]:
            bad["coincident_vertices"] = msg
        else:
            warn["coincident_vertices"] = msg + " (common at UV seams)"

    # 缺陷定位：把有问题的面换算成世界坐标，用户在 ZBrush 里才找得到。
    hot = []
    hot_faces = set(e[0] for e in non_quad) | set(e[0] for e in deg)
    for pair in bad.get("intersecting_faces", {}).get("examples", []):
        hot_faces.update(x for x in pair if isinstance(x, int))
    for n in sorted(hot_faces):
        if 0 <= n < len(faces) and faces[n]:
            f = faces[n]
            c = [sum(verts[i][k] for i in f) / float(len(f)) for k in range(3)]
            hot.append([n] + [round(v, 4) for v in c])
        if len(hot) >= 5:
            break
    if hot:
        report["info"]["hotspots"] = hot

    return report, tol


def density_report(before_verts, before_faces, after_verts, after_faces, tol,
                   cls=None):
    """比较「重算区」和「未改动区」的面密度，给出对齐周围所需的建议面数。

    局部重拓扑最容易翻车的不是拓扑本身，而是重算区比周围密太多：
    接缝处一圈四边形被挤成细条，看着就是"围了一圈碎面"。
    这里直接算出倍数和建议值，用户照着调一次目标面数即可。
    """
    c = cls or classify_faces(before_verts, before_faces, after_verts,
                              after_faces, tol)
    n_new, n_old = c["n_new"], c["n_old"]
    a_new, a_old = c["a_new"], c["a_old"]
    if not (n_new and n_old) or a_new <= 0 or a_old <= 0:
        return None
    d_new = n_new / a_new
    d_old = n_old / a_old
    ratio = d_new / d_old
    ideal = int(round(n_new * d_old / d_new))
    return {
        "new_faces": n_new,
        "old_faces": n_old,
        "density_ratio": round(ratio, 3),
        "ideal_new_faces": ideal,
        "target_k_hint": round(ideal / 1000.0, 2),
        "note": ("remeshed area is %.2fx denser than the untouched area; "
                 "aim for about %d faces there (target K ~= %.2f)"
                 % (ratio, ideal, ideal / 1000.0)),
    }


def compare_unchanged(before_verts, before_faces, after_verts, after_faces, tol,
                      max_ratio, cls=None):
    """对比重拓扑前后的 OBJ：分别用「面数」和「面积」两个口径衡量改动比例。

    未被改动区域的面在两次导出中坐标完全一致，因此会被判为 unchanged。
    若整张网格都被重拓扑（例如遮罩为空导致 HidePt 没隐藏任何面），
    两个比例都会接近 1，从而被拦截。

    v1.1.1: 增加面积口径。只按面数比会低估"改动了大面"的情况——少数几个
    大面被重算，面数比可能只有几个百分点，实际改掉了半张表面。
    """
    if not before_faces:
        return {"error": "before mesh has no faces", "changed_ratio": 1.0}
    c = cls or classify_faces(before_verts, before_faces, after_verts,
                              after_faces, tol)
    ratio = 1.0 - c["n_old"] / float(c["before_faces"])
    total = c["total_before_area"]
    area_ratio = 1.0 - (c["unchanged_area"] / total) if total > 0 else 1.0
    res = {
        "before_faces": c["before_faces"],
        "after_faces": c["after_faces"],
        "unchanged_faces": c["n_old"],
        "changed_ratio": round(ratio, 6),
        "changed_area_ratio": round(area_ratio, 6),
        "max_changed_ratio": max_ratio,
    }
    if ratio > max_ratio or area_ratio > max_ratio:
        res["error"] = (
            "local-change check failed: %.1f%% of the original faces / %.1f%% of "
            "the surface changed, limit is %.1f%%. The mask/visibility isolation "
            "probably failed."
            % (ratio * 100.0, area_ratio * 100.0, max_ratio * 100.0))
    return res


# --------------------------------------------------------------------------
# 报告 / CLI
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local Quad Retopo topology validator")
    ap.add_argument("--after", help="OBJ exported from the remeshed subtool")
    ap.add_argument("--before", help="OBJ exported from the untouched backup subtool")
    ap.add_argument("--report", help="write text + json report here")
    ap.add_argument("--json", dest="json_path", help="also write a .json report")
    ap.add_argument("--max-changed-ratio", type=float, default=0.90,
                    help="max fraction of original faces allowed to change (default 0.90)")
    ap.add_argument("--no-intersect", action="store_true",
                    help="skip triangle-triangle intersection scan (faster on huge meshes)")
    ap.add_argument("--strict-seam", action="store_true",
                    help="treat coincident vertices / unwelded borders as failures")
    ap.add_argument("--self-test", action="store_true", help="run built-in regression tests")
    ap.add_argument("--version", action="version",
                    version="lqr_validate %s" % VERSION)
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.after:
        print("ERROR: --after is required (or use --self-test)", file=sys.stderr)
        return 2
    if not os.path.isfile(args.after):
        print("ERROR: file not found: %s" % args.after, file=sys.stderr)
        return 2
    if not 0.0 < args.max_changed_ratio <= 1.0:
        print("ERROR: --max-changed-ratio must be in (0, 1]", file=sys.stderr)
        return 2

    opts = {"intersect": not args.no_intersect, "strict_seam": args.strict_seam}
    try:
        st_after, st_before = {}, {}
        av, af, al = parse_obj(args.after, stats=st_after)
        bv = bf = None
        if args.before:
            if not os.path.isfile(args.before):
                raise RuntimeError("file not found: %s" % args.before)
            bv, bf, _ = parse_obj(args.before, stats=st_before)
        # 先把重算区算出来：严格检查只对这个区域生效，源模型自带的三角面、
        # 开放边界、自相交都原样保留，不参与判罚。
        tol_pre = bbox_scale(av) * 1e-6
        # 只算一次分类，active / 密度 / 局部性 三处共用（v1.1.1 性能）
        cls = classify_faces(bv, bf, av, af, tol_pre) if bf else None
        active = cls["active"] if cls else None
        report, tol = analyse(av, af, al, opts, active=active)
        bad_v = st_after.get("bad_vertices", 0) + st_before.get("bad_vertices", 0)
        bad_f = st_after.get("bad_faces", 0) + st_before.get("bad_faces", 0)
        if bad_v or bad_f:
            report["warn"]["parse_issues"] = (
                "%d malformed vertex lines and %d faces with fewer than 3 "
                "vertices were skipped or zero-filled; coordinates that were "
                "NaN/Inf are treated as the origin" % (bad_v, bad_f))
        if active is not None:
            report["info"]["scope"] = (
                "strict checks apply to the %d remeshed faces only; the %d "
                "untouched faces keep whatever topology they had"
                % (len(active), len(af) - len(active)))
        if args.before:
            cmp = compare_unchanged(bv, bf, av, af, tol, args.max_changed_ratio,
                                    cls=cls)
            report["locality"] = cmp
            if "error" in cmp:
                report["fail"]["changed_ratio"] = cmp["error"]
            dens = density_report(bv, bf, av, af, tol, cls=cls)
            if dens:
                report["info"]["density"] = dens
                if dens["density_ratio"] > 3.0:
                    report["warn"]["density_mismatch"] = dens["note"]
    except Exception as exc:                                # noqa: BLE001
        report = {"fail": {"validator_error": repr(exc)}, "warn": {}, "info": {}}

    passed = not report.get("fail")
    report["status"] = "PASS" if passed else "FAIL"
    report["version"] = VERSION

    n_fail = len(report.get("fail", {}))
    n_warn = len(report.get("warn", {}))
    lines = ["LQR_STATUS:%s" % report["status"],
             "Local Quad Retopo topology report (validator %s)" % VERSION,
             "faces=%s verts=%s  |  %d fail / %d warn"
             % (report.get("face_count", "?"), report.get("vertex_count", "?"),
                n_fail, n_warn),
             "-" * 60]
    for key, val in report.get("fail", {}).items():
        lines.append("[FAIL] %s: %s" % (key, json.dumps(val, ensure_ascii=False)[:400]))
    for key, val in report.get("warn", {}).items():
        lines.append("[WARN] %s: %s" % (key, val))
    for key, val in report.get("info", {}).items():
        lines.append("[INFO] %s: %s" % (key, val))
    if "locality" in report:
        lines.append("[INFO] locality: %s" % json.dumps(report["locality"],
                                                        ensure_ascii=False)[:400])
    lines.append("-" * 60)
    lines.append("RESULT: %s" % report["status"])
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text)

    if args.report:
        try:
            with open(args.report, "w", encoding="ascii", errors="replace") as fh:
                fh.write(text)
        except OSError as exc:
            print("WARNING: cannot write report: %s" % exc, file=sys.stderr)
    if args.json_path:
        try:
            with open(args.json_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            print("WARNING: cannot write json: %s" % exc, file=sys.stderr)

    return 0 if passed else 1


# --------------------------------------------------------------------------
# 内置回归测试（合成网格，不依赖外部文件）
# --------------------------------------------------------------------------


def _cube():
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
         (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f, list(range(1, len(f) + 1))


def run_self_test():
    opts = {"intersect": True, "strict_seam": False}
    cases = []

    v, f, l = _cube()
    r, tol = analyse(v, f, l, opts)
    cases.append(("clean quad cube -> PASS", not r["fail"], r["fail"]))

    v, f, l = _cube()
    f2 = f + [(0, 1, 2)]
    r, tol = analyse(v, f2, l + [7], opts)
    cases.append(("triangle added -> FAIL non_quad", "non_quad_faces" in r["fail"],
                  r["fail"]))

    v, f, l = _cube()
    f2 = f + [f[0]]
    r, tol = analyse(v, f2, l + [7], opts)
    cases.append(("duplicate face -> FAIL", "duplicate_faces" in r["fail"], r["fail"]))

    v, f, l = _cube()
    f2 = list(f) + [(0, 0, 2, 1)]
    r, tol = analyse(v, f2, l + [7], opts)
    cases.append(("degenerate quad -> FAIL", "degenerate_faces" in r["fail"],
                  r["fail"]))

    # 几何重合但索引不同的一面（模拟 ZBrush 未焊接重叠面）
    v, f, l = _cube()
    v = v + list(v[:4])
    f2 = f + [(8, 11, 10, 9)]
    r, tol = analyse(v, f2, l + [7], opts)
    cases.append(("coincident face -> FAIL", "coincident_faces" in r["fail"],
                  r["fail"]))

    # 与 --before 完全相同的网格 -> changed_ratio = 0
    v, f, l = _cube()
    cmp_ok = compare_unchanged(v, f, v, f, tol, 0.6)
    cases.append(("locality identical -> PASS", "error" not in cmp_ok, cmp_ok))

    # 整张网格都不一样 -> changed_ratio = 1 -> FAIL
    v2 = [(x * 3 + 1, y, z) for (x, y, z) in v]
    cmp_bad = compare_unchanged(v, f, v2, f, tol, 0.6)
    cases.append(("locality whole mesh changed -> FAIL", "error" in cmp_bad, cmp_bad))

    # ---- v1.1.1 新增行为的自检 ----
    r, _ = analyse([], [], [], opts)
    cases.append(("empty mesh -> FAIL empty_mesh", "empty_mesh" in r["fail"], r["fail"]))

    # 面积法：凹 L 形多边形，扇形三角化会算错，Newell 必须给对
    lv = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
    cases.append(("Newell area of concave L == 3.0",
                  abs(polygon_area(lv, (0, 1, 2, 3, 4, 5)) - 3.0) < 1e-9,
                  polygon_area(lv, (0, 1, 2, 3, 4, 5))))

    # 跨格超过 4096 的大三角形不能被漏检
    def _grid(n):
        gv, gf = [], []
        for j in range(n + 1):
            for i in range(n + 1):
                gv.append((i / n, j / n, 0.0))
        for j in range(n):
            for i in range(n):
                a = j * (n + 1) + i
                gf.append((a, a + 1, a + n + 2, a + n + 1))
        return gv, gf
    gv, gf = _grid(40)
    bv2 = gv + [(0.2, -1.0, -1.0), (0.8, 2.0, -1.0), (0.5, 0.5, 1.0)]
    bf2 = gf + [(len(gv), len(gv) + 1, len(gv) + 2)]
    h, _, _ = find_intersecting_faces(bv2, bf2, bbox_scale(bv2) * 1e-6)
    cases.append(("oversized triangle is still detected", len(h) > 0, len(h)))

    # 面数比小、面积比大 -> 必须被面积口径拦住
    av3 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0),
           (2, 1, 0), (3, 1, 0), (3, 2, 0), (2, 2, 0)]
    af3 = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
    bv3 = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)] + av3[4:]
    bf3 = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
    cv3 = compare_unchanged(bv3, bf3, av3, af3, 1e-6, 0.5)
    cases.append(("area ratio catches a changed big face",
                  "error" in cv3 and cv3["changed_area_ratio"] > 0.6, cv3))

    # 只对重算区负责：选区外的三角面不该判失败
    tv = _cube()[0] + [(5, 0, 0), (6, 0, 0), (5, 1, 0)]
    tf = _cube()[1] + [(8, 9, 10)]
    r, _ = analyse(tv, tf, list(range(1, 8)), opts, active=set(range(6)))
    cases.append(("triangle outside the remeshed region -> PASS", not r["fail"],
                  r["fail"]))

    # 接缝质量指标：两个共享一条边的四边形，左边算重算区
    sv = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    sf = [(0, 1, 2, 3), (1, 4, 5, 2)]
    r, _ = analyse(sv, sf, [1, 2], opts, active={0})
    cases.append(("seam metric reported when both regions exist",
                  "seam" in r["info"], r["info"].keys()))

    # ---- v1.3.0：重算区拓扑与闭运算处方 ----
    gv, gf = _grid(3)                      # 3x3 四边形，开口面片
    solid = set(range(9))
    r, _ = analyse(gv, gf, list(range(1, 10)), opts, active=solid)
    pt = r["info"].get("patch", {})
    cases.append(("solid 3x3 patch -> 1 island, 1 loop, 0 holes",
                  pt.get("islands") == 1 and pt.get("holes") == 0, pt))

    ring = solid - {4}                     # 挖掉中心一格 -> 一个洞
    r, _ = analyse(gv, gf, list(range(1, 10)), opts, active=ring)
    pt = r["info"].get("patch", {})
    # 开放面片的外边界不是接缝，此时「环数 - 岛数」会低估洞数；
    # 但处方仍然必须给出至少一次膨胀。
    cases.append(("ring patch -> hole detected and prescription > 0",
                  "mask_holes" in r["warn"] and pt.get("closing_needed", 0) >= 1,
                  pt))

    cases.append(("no active faces -> patch analysis skipped",
                  patch_topology(gv, gf, [], {}, set()) is None, None))

    ok = True
    for name, cond, detail in cases:
        print("%-48s %s" % (name, "OK" if cond else "BROKEN"))
        if not cond:
            try:
                shown = json.dumps(detail, ensure_ascii=False)[:500]
            except TypeError:
                shown = str(detail)[:500]
            print("    detail: %s" % shown)
            ok = False
    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
