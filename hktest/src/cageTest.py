# -----------------------------------------------------------------------------
# Produce hexagonal lattices on an icosahedron.  The hexagons are bent where
# they cross the edges of the icosahedron.
#
# These lattices are described at
#
#       http://viperdb.scripps.edu/icos_server.php?icspage=paradigm

import sys

import numpy as np


def show_hk_lattice(session, h, k, radius, orientation='222',
                    color=(255, 255, 255, 255), sphere_factor=0,
                    edge_radius=None, mesh=False, replace=True, alpha=1):

    name = 'Icosahedron h = %d, k = %d' % (h, k)

    R = 1
    r = R * np.sqrt(3.0)/2.0
    b = np.matrix(((2.0 * r, 1.0 * r), (0.0 * R, 1.5 * R)))

    hv = (h * b[:, 0])
    kv = (k * rotmat(60).dot(b[:, 0]))
    origin = np.array((0, 0))
    Ct = (hv + kv)
    Ctp = rotmat(60).T.dot(Ct)
    Ct = Ct.A1
    Ctp = Ctp.A1

    corners = (
        np.array([0, R]),
        np.array([r, R / 2]),
        np.array([r, -R / 2]),
        np.array([0, -R]),
        np.array([-r, -R / 2]),
        np.array([-r, R / 2]),
    )

    xy_vertexes = (origin, Ct, Ctp)
    hk_vertexes = {(0, 0), (h + k, -h), (h, k)}
    print(hk_vertexes)

    varray = []
    tarray = []
    earray = []
    nth = 0

    for j in range(-h, k + 1):  # row
        for i in range(h + k + 1):  # col
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
            # ## ignore case where line intersects exactly at both corners
            # ## ignore case where hexagon is completely within the facet
            if h != k and len(xy_corners) < 6:
                for c1, c2 in iter_ring(p_corners):
                    for v1, v2 in iter_ring(xy_vertexes):
                        ip = np.array(intersection(*c1, *c2, *v1, *v2))
                        if ip.size > 0 and (not cuts or not any(zeroish(np.linalg.norm(ip - ele)) for ele in cuts)):
                            cuts.append(ip)
                assert len(cuts) <= 2
                len(cuts) < 2 and cuts.clear()
                xy_corners.extend(cuts)
            # triangulate
            # ## proceed if hex has a corner in the facet
            if xy_corners:
                # ## update triangulation point and sort corners around it
                if (i, j) not in hk_vertexes and cuts:
                    p = np.mean(xy_corners, axis=0)
                    xy_corners.sort(
                        key=lambda q: np.arctan2(q[1] - p[1], q[0] - p[0])
                    )
                varray.append(p)
                varray.extend(xy_corners)
                # TODO: "fix me please, thanks" - mask
                mask = 7 if (i, j) in hk_vertexes else 2
                for v1, v2 in iter_ring(range(nth + 1, nth + len(xy_corners) + 1)):
                    tarray.append((nth, v1, v2))
                    earray.append(mask)
                nth += len(xy_corners) + 1
            # print(i, j)

    varray = [(*ele, 0) for ele in varray]
    varray = np.array(varray)
    tarray = np.array(tarray)

    assert len(varray) == max(*tarray[-1]) + 1

    hex_edges = earray  # np.array([2] * len(tarray), np.intc)
    print("varray", varray, "tarray", tarray, "hex_edges",
          hex_edges, sep="\n", file=sys.stderr)

    if mesh:
        model = sm = _cage_surface(session, name, replace)
        sm.set_geometry(varray, None, tarray)
        sm.color = color
        sm.display_style = sm.Mesh
        sm.edge_mask = hex_edges  # Hide spokes of hexagons.
        if sm.id is None:
            session.models.add([sm])
    else:
        # Make cage from markers.
        from chimerax.core.models import Surface
        sm = Surface(name, session)
        sm.set_geometry(varray, None, tarray)
        sm.color = color
        sm.display_style = sm.Mesh
        sm.edge_mask = hex_edges  # Hide spokes of hexagons.
        if edge_radius is None:
            edge_radius = .01 * radius
        mset = _cage_markers(session, name) if replace else None
        from chimerax.markers.cmd import markers_from_mesh
        model = markers_from_mesh(session, [sm], color=color,
                                  edge_radius=edge_radius, markers=mset)
        model.name = name
        if mset:
            mset._prev_markers.delete()

    model.hkcage = True

    return model


def triangle_area(p, q, r):
    # Weisstein, Eric W. "Triangle Area." From MathWorld--A Wolfram Web Resource.
    # https://mathworld.wolfram.com/TriangleArea.html
    return 0.5 * np.abs(np.cross(p - q, p - r))


def in_triangle(p, q1, q2, q3):
    return np.isclose(
        triangle_area(q1, q2, q3),
        np.sum(triangle_area(p, *ele) for ele in iter_ring((q1, q2, q3))),
        atol=np.finfo(np.float32).eps
    )


def zeroish(value):
    return np.isclose(value, 0, atol=np.finfo(np.float32).eps, rtol=0)


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


# -----------------------------------------------------------------------------
#


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


# -----------------------------------------------------------------------------
#
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
