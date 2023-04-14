from chimerax.geometry import Place
import sys

import numpy as np
from functools import partial

from chimerax.core.models import Surface
from chimerax.markers.cmd import markers_from_mesh


def show_hk_lattice(session, h, k, H, K, symmetry, edge_radius, color, replace):
    name = f"Icosahedron({', '.join(map(str, (h, k, H, K, symmetry)))})"

    facets = [
        list(map(np.array, hk_facet(h, k, H, K, i))) for i in range(1, 4)
    ]

    if symmetry == 5:
        fp, ft = icosahedron_geometry_5(h, k, H, K)
    elif symmetry == 3:
        fp, ft = icosahedron_geometry_3(h, k, H, K)
    elif symmetry == 2:
        fp, ft = icosahedron_geometry_2(h, k, H, K)
    else:
        raise ValueError(f"symmetry {symmetry} not equal to 5, 3, or 2")

    surfaces = []
    for idx, ele in enumerate(ft, start=1):
        tri, mode = ele

        ic, iv, it, ie = facets[mode]

        A = np.array([(*e, 1) for e in ic])
        B = np.array([fp[i] for i in tri])

        R, c, t = kabsch_umeyama(B, A)
        v = np.array([t + c * R @ e for e in iv])

        sm = Surface(f"facet-{idx}", session)
        sm.set_geometry(v, None, it)
        sm.edge_mask = ie
        sm.display_style = sm.Mesh
        surfaces.append(sm)

    mset = _cage_markers(session, name) if replace else None
    model = markers_from_mesh(
        session,
        surfaces,
        color=color,
        edge_radius=edge_radius,
        markers=mset
    )
    model.name = name
    if mset:
        mset._prev_markers.delete()

    model.hkcage = True

    return model


def kabsch_umeyama(A, B):
    # https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/

    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


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
                        cuts.append(ip)

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

    varray = [(*ele, 1) for ele in varray]

    assert len(varray) == max(*tarray[-1]) + 1

    return xy_vertexes, varray, tarray, earray


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


def rot2(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_theta, sin_theta],
            [sin_theta, cos_theta]
        ]
    )


def rot3_x(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, cos_theta, -sin_theta, 0],
            [0, sin_theta, cos_theta, 0]
        ]
    )


def rot3_y(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_theta, 0, sin_theta, 0],
            [0, 1, 0, 0],
            [-sin_theta, 0, cos_theta, 0],
        ]
    )


def rot3_z(theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0]
        ]
    )


def rot_rodrigues(v, k, t):
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + np.dot(k, np.dot(k, v)) * (1 - np.cos(t))


def bisection(f, a, b, tol, iter):
    for _ in range(iter):
        c = (a + b) / 2
        f_of_c = f(c)
        if f_of_c == 0 or (b - a) / 2 < tol:
            return c
        if np.sign(f_of_c) == np.sign(f(a)):
            a = c
        else:
            b = c


def brackets(f, a, b, iter):
    frac = b / iter
    prev = np.sign(f(a))
    for i in range(iter):
        x = a + i * frac
        curr = np.sign(f(x))
        if prev != curr:
            yield a + (i - 1) * frac, x
            prev = curr


def circle_cylinder_intersection(e, uxe, center, r_cir, r_cyl, iter=10000, tol=10**-15):
    ux, uy, uz = e
    vx, vy, vz = uxe
    cx, cy, cz = center
    def fy(t): return r_cir * uy * np.cos(t) + r_cir * vy * np.sin(t) + cy
    def fx(t): return r_cir * ux * np.cos(t) + r_cir * vx * np.sin(t) + cx
    def fz(t): return r_cir * uz * np.cos(t) + r_cir * vz * np.sin(t) + cz
    def f(t): return r_cyl * r_cyl - (np.power(fx(t), 2) + np.power(fz(t), 2))
    yield from (
        np.array((fx(t), fy(t), fz(t)))
        for t in (
            bisection(f, a, b, tol=tol, iter=iter)
            for a, b in brackets(f, 0, 2 * np.pi, iter=iter)
        )
        if t
    )


def icosahedron_geometry_5(h, k, H, K):
    """Calculate vertex coordinates and connectivity of icosahedron with 5-fold symmetry.
    @see: https://www.geogebra.org/3d/ebt6eb7f
    """
    from numpy import arccos, arctan2, array, dot, pi, radians, sqrt, vstack
    from numpy.linalg import norm

    b = np.array((1, 0))
    Ct = h * b + k * dot(rot2(np.radians(60)), b)
    Cq = H * np.dot(rot2(np.radians(60)), b) + K * \
        np.dot(rot2(np.radians(120)), b)

    phi = (1 + np.sqrt(5)) / 2.0
    b = 0.5
    a = phi * b
    ivarray = np.array((
        (0, b, -a),     # A
        (-b, a, 0),     # B
        (b, a, 0),      # C
        (a, 0, -b),     # D
        (0, -b, -a),    # E
        (-a, 0, -b),    # F
    ))
    theta = np.pi / 2 - np.arctan2((1 / (2 * phi)), 0.5)
    ivarray = Place(matrix=rot3_x(theta)).transform_points(ivarray)

    s = np.linalg.norm(Cq) / np.linalg.norm(Ct)
    alpha = np.arccos(
        np.dot(Ct, Cq) / (np.linalg.norm(Ct) * np.linalg.norm(Cq)))
    v1 = ivarray[1]  # B
    tv = v1 + Place(matrix=rot3_z(-alpha)
                    ).transform_vector(s * np.array([1, 0, 0]))

    r1 = np.linalg.norm(v1 - np.array((0, v1[1], 0)))
    r2 = v1[1] - tv[1]
    v6z = np.sqrt(r1 * r1 - tv[0] * tv[0])
    v6 = np.array((tv[0], v1[1] - np.sqrt(-(v1[2] * v1[2]) + 2 *
                                          v1[2] * v6z + r2 * r2 - v6z * v6z), v6z))  # G

    placer = Place(matrix=rot3_y(np.radians(72)))
    v7 = placer.transform_points(np.array([v6]))  # H
    v8 = placer.transform_points(v7)           # I
    v9 = placer.transform_points(v8)           # J
    vA = placer.transform_points(v9)           # K

    ivarray = np.vstack((ivarray, v6, v7[0], v8[0], v9[0], vA[0]))
    ivarray -= np.array((0, (v1[1] + v6[1]) / 2, 0))
    ivarray = np.vstack((ivarray, -ivarray[0]))   # L <- -A

    # TODO: replace with numbers
    from string import ascii_uppercase
    itarray = (
        "ABC", "ACD", "ADE", "AEF", "AFB",  # cap Δ
        "LGH", "LHI", "LIJ", "LJK", "LKG",  # cap ∇
        "BGC", "CHD", "DIE", "EJF", "FKB",  # mid ∇
        "HCG", "GBK", "KFJ", "JEI", "IDH",  # mid Δ
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)
    itarray = (
        *((ele, 0) for ele in itarray[:10]),
        *((ele, 1) for ele in itarray[10:])
    )

    return ivarray, itarray


def icosahedron_geometry_3(h, k, H, K):
    """Calculate vertex coordinates and connectivity of icosahedron with 3-fold symmetry.
    @see: https://www.geogebra.org/3d/z2pxwfn4
    """
    from operator import itemgetter

    from chimerax.geometry import Place
    from numpy import (abs, arccos, array, cross, dot, mean, radians, sqrt,
                       vstack)
    from numpy.linalg import norm

    b = array((1, 0))
    Ct = h * b + k * dot(rot2(radians(60)), b)
    Cq = H * dot(rot2(radians(60)), b) + K * dot(rot2(radians(120)), b)

    phi = (1 + sqrt(5)) / 2.0
    b = 0.5
    a = phi * b
    ivarray = array((
        (0, b, -a),  # A
        (-b, a, 0),  # B
        (b, a, 0),   # C
        (0, b, a),   # D
        (a, 0, -b),  # E
        (-a, 0, -b)  # F
    ))
    y_axis = array((0, 1, 0))
    centroid = mean(ivarray[1:4], axis=0)  # @ BCD
    theta = arccos(dot(y_axis, centroid) / (norm(y_axis) * norm(centroid)))
    ivarray = Place(matrix=rot3_x(theta)).transform_points(ivarray)

    u = ivarray[2] - ivarray[3]  # C - D
    u /= norm(u)
    v = array((u[1], -u[0], 0))
    w = cross(u, v)
    k = w / norm(w)
    alpha = arccos(dot(Ct, Cq) / (norm(Ct) * norm(Cq)))
    side = norm(Cq) / norm(Ct)
    # ivarray[3] -> D
    tip = ivarray[3] + side * rot_rodrigues(u, k, alpha)
    center = ivarray[3] + dot(tip - ivarray[3], u) / dot(u, u) * u
    r_cir = norm(tip - center)
    r_cyl = norm(array([0, ivarray[3][1], 0]) - ivarray[3])
    e = tip - center
    e /= norm(e)
    uxe = cross(u, e)
    uxe /= norm(uxe)

    v6 = min(circle_cylinder_intersection(
        e, uxe, center, r_cir, r_cyl), key=itemgetter(1))  # G
    placer = Place(matrix=rot3_y(radians(120)))
    v7 = placer.transform_points(array([v6]))  # H
    v8 = placer.transform_points(v7)           # I

    yval = v6[1] - (ivarray[0][1] - ivarray[3][1])  # y(G) - (y(A) - y(D))
    cvec = dot(rot3_y(radians(60)), array(
        (v6[0], yval, v6[2], 1))) - array((0, yval, 0))
    cvec = abs(ivarray[0][2]) * cvec / norm(cvec)
    v9 = array((cvec[0], yval, cvec[2]))       # J
    placer = Place(matrix=rot3_y(radians(120)))
    vA = placer.transform_points(array([v9]))  # K
    vB = placer.transform_points(vA)           # L

    ivarray = vstack((*ivarray, v6, v7[0], v8[0], v9, vA[0], vB[0]))
    ivarray -= array((0, (ivarray[0][1] + ivarray[11][1]) / 2, 0))

    # TODO: replace with numbers
    from string import ascii_uppercase
    itarray = (
        "ABC",                # cap -
        "AFB", "BDC", "CEA",  # cap ∇
        "JHK", "KIL", "LGJ",  # cap Δ
        "JKL",                # cap -
        "DCG", "EAH", "FBI",  # mid ∇ 1
        "GLD", "HJE", "IKF",  # mid Δ 1
        "BID", "CGE", "AHF",  # mid Δ 2
        "LDI", "JEG", "KFH",  # mid ∇ 2
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)
    itarray = (
        *((ele, 0) for ele in itarray[:8]),
        *((ele, 1) for ele in itarray[8:14]),
        *((ele, 2) for ele in itarray[14:])
    )

    return ivarray, itarray


def icosahedron_geometry_2(h, k, H, K):
    """Calculate vertex coordinates and connectivity of icosahedron with 2-fold symmetry.
    @see: https://www.geogebra.org/3d/ucxunycw
    """
    from operator import itemgetter

    from chimerax.geometry import Place
    from numpy import arccos, array, cross, dot, radians, sqrt, vstack
    from numpy.linalg import norm

    b = array((1, 0))
    Ct = h * b + k * dot(rot2(radians(60)), b)
    Cq = H * dot(rot2(radians(60)), b) + K * dot(rot2(radians(120)), b)

    phi = (1 + sqrt(5)) / 2.0
    b = 0.5
    a = phi * b
    ivarray = array((
        (-b, a, 0),  # A
        (b, a, 0),   # B
        (0, b, a),   # C
        (0, b, -a),  # D
        (-a, 0, b),  # E
        (a, 0, -b)   # F
    ))

    u = ivarray[1] - ivarray[2]
    u /= norm(u)
    v = array((u[1], -u[0], 0))
    w = cross(u, v)
    k = w / norm(w)
    alpha = arccos(dot(Ct, Cq) / (norm(Ct) * norm(Cq)))
    side = norm(Cq) / norm(Ct)
    tip = ivarray[2] + side * rot_rodrigues(u, k, alpha)
    center = ivarray[2] + dot(tip - ivarray[2], u) / dot(u, u) * u
    r_cir = norm(tip - center)
    r_cyl = norm(ivarray[2])
    e = tip - center
    e /= norm(e)
    uxe = cross(u, e)
    uxe /= norm(uxe)

    v6 = min(circle_cylinder_intersection(
        e, uxe, center, r_cir, r_cyl), key=itemgetter(1))
    placer180 = Place(matrix=rot3_y(radians(180)))
    v7 = placer180.transform_points(array([v6]))[0]

    temp = array((ivarray[2][0], 0, ivarray[2][2]))
    theta = arccos(dot(ivarray[5], temp) / (norm(ivarray[5]) * norm(temp)))
    placer = Place(matrix=rot3_y(theta))
    temp = placer.transform_points(
        array([v6]))[0] - array((0, ivarray[2][1], 0))
    c1 = array((0, temp[1], 0))
    v8 = c1 + ivarray[2][2] * (temp - c1)
    v9 = placer180.transform_points(array([v8]))[0]

    temp = Place(matrix=rot3_y(theta + radians(90))
                 ).transform_points(array([v6]))[0] - array((0, ivarray[1][1], 0))
    c2 = array((0, temp[1], 0))
    vA = c2 + ivarray[1][0] * (temp - c2)
    vB = placer180.transform_points(array([vA]))[0]

    ivarray = vstack((*ivarray, v6, v7, v8, v9, vA, vB))
    ivarray -= array((0, (ivarray[5][1] + v6[1]) / 2, 0))

    # TODO: replace with numbers
    from string import ascii_uppercase
    itarray = (
        "ABC", "ACE", "BAD", "BDF", "LKJ", "LJG", "KLI", "KIH",  # cap
        "JKE", "ECJ", "GJC", "CBG", "ILF", "FDI", "HID", "DAH",  # mid 1
        "AEH", "KHE", "BFG", "LGF"                               # mid 2
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)
    itarray = (
        *((ele, 0) for ele in itarray[:8]),
        *((ele, 1) for ele in itarray[8:16]),
        *((ele, 2) for ele in itarray[16:])
    )

    return ivarray, itarray


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
