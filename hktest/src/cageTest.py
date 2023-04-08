import sys

import numpy as np
from functools import partial

from chimerax.core.models import Surface
from chimerax.markers.cmd import markers_from_mesh


def show_hk_lattice(session, h, k, H, K, edge_radius, color=(255, 0, 255, 255), replace=True):

    name = f"Icosahedron({h}, {k}, {H}, {K})"

    v, t, e = list(map(np.array, hk_facet(h, k, H, K, 1)))
    v2, t2, e2 = list(map(np.array, hk_facet(h, k, H, K, 2)))
    v3, t3, e3 = list(map(np.array, hk_facet(h, k, H, K, 3)))

    sm = Surface(name + " (1)", session)
    sm.set_geometry(v, None, t)
    sm.edge_mask = e
    sm.color = color
    sm.display_style = sm.Mesh
    sm2 = Surface(name + " (2)", session)
    sm2.set_geometry(v2, None, t2)
    sm2.edge_mask = e2
    sm2.color = color
    sm2.display_style = sm.Mesh
    sm3 = Surface(name + " (3)", session)
    sm3.set_geometry(v3, None, t3)
    sm3.edge_mask = e3
    sm3.color = color
    sm3.display_style = sm.Mesh
    mset = _cage_markers(session, name) if replace else None
    model = markers_from_mesh(
        session,
        [sm, sm2, sm3],
        color=color,
        edge_radius=edge_radius,
        markers=mset
    )
    model.name = name
    if mset:
        mset._prev_markers.delete()

    model.hkcage = True

    return model


def hk_facet(h, k, H=1, K=1, t=1, R=1):
    r = R * np.sqrt(3.0) / 2.0
    b = np.matrix(((2.0 * r, 1.0 * r), (0.0 * R, 1.5 * R)))

    hv = (h * b[:, 0])
    kv = (k * rotmat(60).dot(b[:, 0]))
    HV = (H * rotmat(60).dot(b[:, 0]))
    KV = (K * rotmat(120).dot(b[:, 0]))
    origin = np.array((0, 0))

    Ct = hv + kv
    CT = rotmat(60).T.dot(Ct)
    CTT = rotmat(-120).T.dot(Ct)
    Ct = Ct.A1
    CT = CT.A1
    CTT = CTT.A1
    CQ = (HV + KV).A1

    corners = (
        np.array([0, R]),
        np.array([r, R / 2]),
        np.array([r, -R / 2]),
        np.array([0, -R]),
        np.array([-r, -R / 2]),
        np.array([-r, R / 2]),
    )

    if t == 1:
        xy_vertexes = (origin, Ct, CT)
        hk_vertexes = {(0, 0), (h + k, -h), (h, k)}
        j_range = range(-h, k + 1)
        i_range = range(h + k + 1)
    elif t == 2:
        xy_vertexes = (origin, CQ, Ct)
        hk_vertexes = {(0, 0), (-K, H + K), (h, k)}
        j_range = range(-h, max(k, H + K) + 1)
        i_range = range(-K, h + 1)
    elif t == 3:
        xy_vertexes = (origin, CTT, CQ)
        hk_vertexes = {(0, 0), (-(h + k), h), (-K, H + K)}
        j_range = range(max(h, H + K) + 1)
        i_range = range(-(h + k), 0 + 1)
    else:
        raise ValueError("t not in {1, 2, 3}")

    varray = []
    tarray = []
    earray = []
    nth = 0

    for j in j_range:  # row
        for i in i_range:  # col
            print(f"coor {(i, j)}")
            # calculate hexagonal lattice position
            p = (b @ np.array([i, j])).A1
            # update hex corners
            # ## calculate corner positions
            p_corners = [ele + p for ele in corners]
            # ## keep corners that are inside the facet
            xy_corners = [
                ele for ele in p_corners
                if in_triangle(ele, *xy_vertexes)
            ]
            # add new corners based on facet edge intersection
            cuts = []
            for c1, c2 in iter_ring(p_corners):
                for v1, v2 in iter_ring(xy_vertexes):
                    ip = np.array(intersection(*c1, *c2, *v1, *v2))
                    if ip.size > 0 and (not any(zeroish(np.linalg.norm(ip - ele)) for ele in (c1, c2))):
                        cuts.append(MeshPoint(ip, info=True))

            cuts = [ele for ele in cuts if not any(
                zeroish(np.linalg.norm(c1 - ele)) for c1 in xy_corners)]
            xy_corners.extend(cuts)
            # triangulate
            # ## proceed if hex has a corner in the facet
            if len(xy_corners) > 1:
                if (i, j) in hk_vertexes:
                    xy_corners.append(p)
                p = np.mean(xy_corners, axis=0)
                xy_corners.sort(key=partial(sort_ccw, p))
                varray.append(p)
                varray.extend(xy_corners)
                for idx, ele in enumerate(iter_ring(range(nth + 1, nth + len(xy_corners) + 1))):
                    c1 = xy_corners[idx],
                    c2 = xy_corners[(idx+1) % len(xy_corners)]
                    mask = 2
                    tarray.append((nth, *ele))
                    earray.append(mask)
                nth += len(xy_corners) + 1

    varray = [(*ele, 0) for ele in varray]

    assert len(varray) == max(*tarray[-1]) + 1

    return varray, tarray, earray


class MeshPoint(np.ndarray):

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.info = getattr(obj, 'info', None)


def sort_ccw(p, c):
    return np.arctan2(p[1] - c[1], p[0] - c[0])


def on_segment(a, b, c):
    return np.isclose(np.linalg.norm(a-c) + np.linalg.norm(b-c), np.linalg.norm(a - b))


def triangle_area(p, q, r):
    # Weisstein, Eric W. "Triangle Area." From MathWorld--A Wolfram Web Resource.
    # https://mathworld.wolfram.com/TriangleArea.html
    return 0.5 * np.abs(np.cross(p - q, p - r))


def in_triangle(p, q1, q2, q3):
    return np.isclose(
        triangle_area(q1, q2, q3),
        np.sum(triangle_area(p, *ele) for ele in iter_ring((q1, q2, q3)))
    )


def zeroish(value):
    return np.isclose(value, 0)


def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # adapted from http://paulbourke.net/geometry/pointlineplane/edge_intersection.py
    d = ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

    if zeroish(d):
        return ()

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
    if ua < 0 or ua > 1:
        return ()

    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d
    if ub < 0 or ub > 1:
        return ()

    return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))


def iter_pairs(container):
    """ABC -> AB BC"""
    yield from zip(container, container[1:])


def iter_ring(container):
    """ABC -> AB BC CA"""
    yield from iter_pairs(container)
    yield container[-1], container[0]


def rotmat(deg):
    th = np.deg2rad(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def _cage_markers(session, name):
    from chimerax.markers import MarkerSet
    mlist = [m for m in session.models.list(type=MarkerSet)
             if hasattr(m, 'hkcage')]
    if mlist:
        mset = mlist[0]
        mset._prev_markers = mset.atoms
        mset.name = name
        mset.hkcage = True
        return mset
    return None


def _cage_surface(session, name, replace):
    # Make new surface model or find an existing one.
    sm = None
    from chimerax.core.models import Surface
    if replace:
        mlist = [m for m in session.models.list(type=Surface)
                 if hasattr(m, 'hkcage')]
        if mlist:
            sm = mlist[0]
            sm.name = name
    if sm is None:
        sm = Surface(name, session)
    sm.hkcage = True
    return sm
