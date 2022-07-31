
# Produce hexagonal lattices on an icosahedron.  The hexagons are bent where
# they cross the edges of the icosahedron.
#
# These lattices are described at
#
#       http://viperdb.scripps.edu/icos_server.php?icspage=paradigm


# symmetry: equilateral, 5-fold, 3-fold, 2-fold
symmetry_names = ("e", "5", "3", "2")


def show_hk_lattice(session, h, k, H=None, K=None, symmetry="e", radius=100.0, orientation='222',
                    color=(255, 255, 255, 255),
                    sphere_factor=0, edge_radius=None, mesh=False, replace=True, alpha=1):

    print(*locals().items(), sep="\n")
    name = f'Icosahedron(h={h}, k={k}, H={H}, K={K}, symmetry={symmetry}, alpha={alpha})'
    varray, tarray, hex_edges = hk_icosahedron_lattice(h, k, H, K, symmetry, radius, orientation, alpha)
    # TODO: investigate the commented-out code...
    # interpolate_with_sphere(varray, radius, sphere_factor)

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


def hk_icosahedron_lattice(h, k, H, K, symmetry, radius, orientation, alpha):
    # Find triangles for the hk lattice covering one asymmetric unit equilateral triangle.
    # The asym unit triangle (corners) and hk lattice triangles are in the xy plane in 3-d.
    from itertools import chain

    from chimerax.geometry.icosahedron import icosahedron_geometry
    from numpy import array, intc, multiply

    lattices = {cls.id: cls for cls in (HKTriangle, *all_subclasses(HKTriangle))}
    lattice = lattices.get(alpha, HKTriangle)()

    if symmetry == "e":
        corners_1 = hk3_to_xyz(lattice.corners(h, k, h, k))
        triangles_1, t_hex_edges_1 = zip(*lattice.walk(h, k))
        triangles_1 = list(map(hk3_to_xyz, chain.from_iterable(triangles_1)))
        t_hex_edges_1 = list(chain.from_iterable(t_hex_edges_1))
        # Map the 2d hk asymmetric unit triangles onto each face of an icosahedron
        ivarray, itarray = icosahedron_geometry(orientation)
        faces = ((ivarray[i0], ivarray[i1], ivarray[i2]) for i0, i1, i2 in itarray)
        tlist = list(chain.from_iterable((map_triangles(triangle_map(corners_1, face), triangles_1) for face in faces)))
        # Compute the edge mask to show just the hexagon edges.
        t_hex_edges = t_hex_edges_1 * 20
    elif symmetry == "5":
        corners_1 = hk3_to_xyz(lattice.corners(h, k, h, k))
        triangles_1, t_hex_edges_1 = zip(*lattice.walk(h, k))
        triangles_1 = list(map(hk3_to_xyz, chain.from_iterable(triangles_1)))
        t_hex_edges_1 = list(chain.from_iterable(t_hex_edges_1))
        corners_2 = hk3_to_xyz(lattice.corners(h, k, H, K))
        triangles_2, t_hex_edges_2 = zip(*lattice.walk(h, k, H, K, mode=2))
        triangles_2 = list(map(hk3_to_xyz, chain.from_iterable(triangles_2)))
        t_hex_edges_2 = list(chain.from_iterable(t_hex_edges_2))
        # Map the 2d hk asymmetric unit triangles onto each face of an icosahedron
        ivarray, itarray = icosahedron_geometry_5(h, k, H, K)
        faces = [(ivarray[i0], ivarray[i1], ivarray[i2]) for i0, i1, i2 in itarray]
        tlist = (list(chain.from_iterable((
            *(map_triangles(triangle_map(corners_1, face), triangles_1) for face in faces[:10]),
            *(map_triangles(triangle_map(corners_2, face), triangles_2) for face in faces[10:])
        ))))
        t_hex_edges = t_hex_edges_1 * 10 + t_hex_edges_2 * 10
    elif symmetry == "3":
        corners_1 = hk3_to_xyz(lattice.corners(h, k, h, k))
        triangles_1, t_hex_edges_1 = zip(*lattice.walk(h, k))
        triangles_1 = list(map(hk3_to_xyz, chain.from_iterable(triangles_1)))
        t_hex_edges_1 = list(chain.from_iterable(t_hex_edges_1))
        corners_2 = hk3_to_xyz(lattice.corners(h, k, H, K))
        triangles_2, t_hex_edges_2 = zip(*lattice.walk(h, k, H, K, mode=2))
        triangles_2 = list(map(hk3_to_xyz, chain.from_iterable(triangles_2)))
        t_hex_edges_2 = list(chain.from_iterable(t_hex_edges_2))
        corners_3 = hk3_to_xyz(lattice.corners(h, k, K, h))
        triangles_3, t_hex_edges_3 = zip(*lattice.walk(h, k, K, h, mode=3))
        triangles_3 = list(map(hk3_to_xyz, chain.from_iterable(triangles_3)))
        t_hex_edges_3 = list(chain.from_iterable(t_hex_edges_3))
        # Map the 2d hk asymmetric unit triangles onto each face of an icosahedron
        ivarray, itarray = icosahedron_geometry_3(h, k, H, K)
        faces = [(ivarray[i0], ivarray[i1], ivarray[i2]) for i0, i1, i2 in itarray]
        tlist = (list(chain.from_iterable((
            *(map_triangles(triangle_map(corners_1, face), triangles_1) for face in faces[:8]),
            *(map_triangles(triangle_map(corners_2, face), triangles_2) for face in faces[8:14]),
            *(map_triangles(triangle_map(corners_3, face), triangles_3) for face in faces[14:])
        ))))
        t_hex_edges = t_hex_edges_1 * 8 + t_hex_edges_2 * 6 + t_hex_edges_3 * 6
    elif symmetry == "2":
        corners_1 = hk3_to_xyz(lattice.corners(h, k, h, k))
        triangles_1, t_hex_edges_1 = zip(*lattice.walk(h, k))
        triangles_1 = list(map(hk3_to_xyz, chain.from_iterable(triangles_1)))
        t_hex_edges_1 = list(chain.from_iterable(t_hex_edges_1))
        corners_2 = hk3_to_xyz(lattice.corners(h, k, H, K))
        triangles_2, t_hex_edges_2 = zip(*lattice.walk(h, k, H, K, mode=2))
        triangles_2 = list(map(hk3_to_xyz, chain.from_iterable(triangles_2)))
        t_hex_edges_2 = list(chain.from_iterable(t_hex_edges_2))
        corners_3 = hk3_to_xyz(lattice.corners(h, k, K, h))
        triangles_3, t_hex_edges_3 = zip(*lattice.walk(h, k, K, h, mode=3))
        triangles_3 = list(map(hk3_to_xyz, chain.from_iterable(triangles_3)))
        t_hex_edges_3 = list(chain.from_iterable(t_hex_edges_3))
        # Map the 2d hk asymmetric unit triangles onto each face of an icosahedron
        ivarray, itarray = icosahedron_geometry_2(h, k, H, K)
        faces = [(ivarray[i0], ivarray[i1], ivarray[i2]) for i0, i1, i2 in itarray]
        tlist = (list(chain.from_iterable((
            *(map_triangles(triangle_map(corners_1, face), triangles_1) for face in faces[:8]),
            *(map_triangles(triangle_map(corners_2, face), triangles_2) for face in faces[8:16]),
            *(map_triangles(triangle_map(corners_3, face), triangles_3) for face in faces[16:])
        ))))
        t_hex_edges = t_hex_edges_1 * 8 + t_hex_edges_2 * 8 + t_hex_edges_3 * 4

    # TODO: keep for debug, remove for production...
    from string import ascii_uppercase
    for i, e in enumerate(ivarray):
        print(ascii_uppercase[i], *e, sep="\t")

    # Compute the edge mask to show just the hexagon edges.
    hex_edges = array(t_hex_edges, intc)
    # Convert from triangles defined by 3 vertex points, to an array of
    # unique vertices and triangles as 3 indices into the unique vertex list.
    va, ta = surface_geometry(tlist, tolerance=1e-5)
    # Scale to requested radius
    multiply(va, radius, va)

    return va, ta, hex_edges


# Triangulate the portion of triangle t1 inside t2.  The triangles are specified
# by 3 vertex points and are in 2 dimensions.  Only the cases that occur in the
# hk icosahedral grids are handled.
#
def triangle_intersection(t1, t2, edge_mask):
    interior_vertices = []
    exterior_vertices = []
    boundary_vertices = []
    locations = {-1: exterior_vertices, 0: boundary_vertices, 1: interior_vertices}
    for k in range(3):
        loc = vertex_in_triangle(t1[k], t2)
        locations[loc].append(k)

    iv = len(interior_vertices)
    ev = len(exterior_vertices)
    if iv == 0 and ev > 0:
        return [], []
    if ev == 0:
        return [t1], [edge_mask]

    # Have at least one exterior and one interior vertex.  Need new triangles.
    if iv == 1 and ev == 1:
        i = interior_vertices[0]
        b = boundary_vertices[0]
        e = exterior_vertices[0]
        thalf = list(t1)
        thalf[e] = cut_point(t1[i], t1[e], t2)
        return [thalf], [mask_edge(edge_mask, (e, b))]

    if iv == 1 and ev == 2:
        i = interior_vertices[0]
        e1 = exterior_vertices[0]
        e2 = exterior_vertices[1]
        tpiece = list(t1)
        tpiece[e1] = cut_point(t1[i], t1[e1], t2)
        tpiece[e2] = cut_point(t1[i], t1[e2], t2)
        return [tpiece], [mask_edge(edge_mask, (e1, e2))]

    if iv == 2 and ev == 1:
        i1 = interior_vertices[0]
        i2 = interior_vertices[1]
        e = exterior_vertices[0]
        tpiece1 = list(t1)
        tpiece1[e] = cut_point(t1[i1], t1[e], t2)
        em1 = mask_edge(edge_mask, (i2, e))
        tpiece2 = list(t1)
        tpiece2[e] = cut_point(t1[i2], t1[e], t2)
        tpiece2[i1] = tpiece1[e]
        em2 = mask_edge(edge_mask, (i1, e), (i1, i2))
        return [tpiece1, tpiece2], [em1, em2]

    raise ValueError('hkcage: Unexpected triangle intersection')


# Vertex and triangle are in two dimensions with triangle defined by 3 corner
# vertices.
#
def vertex_in_triangle(v, t):
    v0, v1, v2 = t
    v01p = (-v1[1] + v0[1], v1[0] - v0[0])
    v12p = (-v2[1] + v1[1], v2[0] - v1[0])
    v20p = (-v0[1] + v2[1], v0[0] - v2[0])
    from numpy import dot as inner_product
    from numpy import subtract
    sides = (inner_product(subtract(v, v0), v01p),
             inner_product(subtract(v, v1), v12p),
             inner_product(subtract(v, v2), v20p))
    inside = len([s for s in sides if s > 0])
    outside = len([s for s in sides if s < 0])
    if inside == 3:
        return 1  # Interior
    elif outside == 0:
        return 0  # Boundary

    return -1  # Outside


# Find intersection of segment u->v with triangle boundary.
# u and v must be an interior and an exterior point.
#
def cut_point(u, v, tri):
    for e in range(3):
        ip = segment_intersection(u, v, tri[e], tri[(e + 1) % 3])
        if ip:
            return ip
    raise ValueError('hkcage: No intersection %s %s %s' % (u, v, tri))


# Find intersection of segment ab with segment cd.
# Return (x,y) or None if no intersectin.
#
def segment_intersection(a, b, c, d):
    m11 = b[0] - a[0]
    m21 = b[1] - a[1]
    m12 = c[0] - d[0]
    m22 = c[1] - d[1]
    det = m11 * m22 - m12 * m21
    if det == 0:
        return None

    y1 = c[0] - a[0]
    y2 = c[1] - a[1]
    f = float(m22 * y1 - m12 * y2) / det
    g = float(-m21 * y1 + m11 * y2) / det
    if f < 0 or f > 1 or g < 0 or g > 1:
        return None

    p = (a[0] + m11 * f, a[1] + m21 * f)
    return p


# Mask out edges given by pair of vertex indices (0-2).  Bits 0, 1, and 2
# correspond to edges 0-1, 1-2, and 2-0 respectively.
#
def mask_edge(edge_mask, *edges):
    ebits = {(0, 1): 1, (1, 0): 1, (1, 2): 2, (2, 1): 2, (2, 0): 4, (0, 2): 4}
    emask = edge_mask
    for e in edges:
        emask &= ~ebits[e]
    return emask


# Shear transform 2d hk points to points on the xy plane in 3 dimensions (z=0).
#
def hk3_to_xyz(hklist):
    from math import sqrt
    hx = sqrt(3) / 6
    hy = 0.5 / 3
    ky = 1.0 / 3
    xyz_list = [(h * hx, k * ky + h * hy, 0) for h, k in hklist]
    return xyz_list


# Shear transform 2d hk points to points on the xy plane in 3 dimensions (z=0).
#
def hk2_to_xyz(hklist):
    from math import sqrt
    hx = 1 / 2.0
    hy = 1 / (2 * sqrt(3))
    ky = 1 / sqrt(3)
    xyz_list = [(h * hx, k * ky + h * hy, 0) for h, k in hklist]
    return xyz_list


# Compute the 3 by 4 transform matrix mapping one 3-d triangle to another.
#
def triangle_map(tri1, tri2):
    from numpy import dot as matrix_multiply
    from numpy import float, float64, subtract, zeros

    f1 = zeros((3, 3), float)
    f1[:, 0], f1[:, 1] = subtract(tri1[1], tri1[0]), subtract(tri1[2], tri1[0])
    f1[:, 2] = cross_product(f1[:, 0], f1[:, 1])

    f2 = zeros((3, 3), float)
    f2[:, 0], f2[:, 1] = subtract(tri2[1], tri2[0]), subtract(tri2[2], tri2[0])
    f2[:, 2] = cross_product(f2[:, 0], f2[:, 1])

    from numpy.linalg import inv as inverse
    f1inv = inverse(f1)

    tmap = zeros((3, 4), float64)
    tmap[:, :3] = matrix_multiply(f2, f1inv)
    tmap[:, 3] = subtract(tri2[0], matrix_multiply(tmap[:, :3], tri1[0]))

    from chimerax.geometry import Place
    tf = Place(matrix=tmap)

    return tf


# Apply a 3x4 affine transformation to vertices of triangles.
#
def map_triangles(tmap, triangles):
    tri = [[tmap * v for v in t] for t in triangles]
    return tri


#
def cross_product(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


# Take a list of triangles where each triangle is specified by 3 xyz vertex
# positions and convert to a vertex and triangle array where the triangle
# array contains indices into the vertex array.  Vertices in the original
# triangle data that are close (within tolerance) are merged into a single
# vertex.
#
def surface_geometry(triangles, tolerance=1e-5):
    from numpy import array, intc, reshape
    from numpy import single as floatc
    varray = reshape(triangles, (3 * len(triangles), 3)).astype(floatc)

    uindex = {}
    unique = []
    from chimerax.geometry import find_close_points
    for v in range(len(varray)):
        if v not in uindex:
            i1, i2 = find_close_points(varray[v:v + 1, :], varray, tolerance)
            for i in i2:
                if i not in uindex:
                    uindex[i] = len(unique)
            unique.append(varray[v])

    uvarray = array(unique, floatc)
    tlist = [(uindex[3 * t], uindex[3 * t + 1], uindex[3 * t + 2]) for t in range(len(triangles))]
    tarray = array(tlist, intc)

    return uvarray, tarray


# Radially interpolate vertex points a certain factor towards a sphere of
# given radius.
#
def interpolate_with_sphere(varray, radius, sphere_factor):
    if sphere_factor == 0:
        return

    from math import sqrt
    for v in range(len(varray)):
        x, y, z = varray[v]
        r = sqrt(x * x + y * y + z * z)
        if r > 0:
            ri = r * (1 - sphere_factor) + radius * sphere_factor
            f = ri / r
            varray[v, :] = (f * x, f * y, f * z)


def rot2(theta):
    from numpy import array, cos, sin
    cos_theta, sin_theta = cos(theta), sin(theta)
    return array(
        [
            [cos_theta, sin_theta],
            [sin_theta, cos_theta]
        ]
    )


def rot3_x(theta):
    from numpy import array, cos, sin
    cos_theta, sin_theta = cos(theta), sin(theta)
    return array(
        [
            [1, 0, 0, 0],
            [0, cos_theta, -sin_theta, 0],
            [0, sin_theta, cos_theta, 0]
        ]
    )


def rot3_y(theta):
    from numpy import array, cos, sin
    cos_theta, sin_theta = cos(theta), sin(theta)
    return array(
        [
            [cos_theta, 0, sin_theta, 0],
            [0, 1, 0, 0],
            [-sin_theta, 0, cos_theta, 0],
        ]
    )


def rot3_z(theta):
    from numpy import array, cos, sin
    cos_theta, sin_theta = cos(theta), sin(theta)
    return array(
        [
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0]
        ]
    )


def rot_rodrigues(v, k, t):
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    from numpy import cos, cross, dot, sin
    return v * cos(t) + cross(k, v) * sin(t) + dot(k, dot(k, v)) * (1 - cos(t))


def bisection(f, a, b, tol, iter):
    from numpy import sign
    for _ in range(iter):
        c = (a + b) / 2
        f_of_c = f(c)
        if f_of_c == 0 or (b - a) / 2 < tol:
            return c
        if sign(f_of_c) == sign(f(a)):
            a = c
        else:
            b = c


def brackets(f, a, b, iter):
    from numpy import sign
    frac = b / iter
    prev = sign(f(a))
    for i in range(iter):
        x = a + i * frac
        curr = sign(f(x))
        if prev != curr:
            yield a + (i - 1) * frac, x
            prev = curr


def circle_cylinder_intersection(e, uxe, center, r_cir, r_cyl, iter=10000, tol=10**-15):
    from numpy import array, cos, pi, power, sin
    ux, uy, uz = e
    vx, vy, vz = uxe
    cx, cy, cz = center
    def fy(t): return r_cir * uy * cos(t) + r_cir * vy * sin(t) + cy
    def fx(t): return r_cir * ux * cos(t) + r_cir * vx * sin(t) + cx
    def fz(t): return r_cir * uz * cos(t) + r_cir * vz * sin(t) + cz
    def f(t): return r_cyl * r_cyl - (power(fx(t), 2) + power(fz(t), 2))
    yield from (
        array((fx(t), fy(t), fz(t)))
        for t in (
            bisection(f, a, b, tol=tol, iter=iter)
            for a, b in brackets(f, 0, 2 * pi, iter=iter)
        )
        if t
    )


def icosahedron_geometry_5(h, k, H, K):
    """Calculate vertex coordinates and connectivity of icosahedron with 5-fold symmetry.
    @see: https://www.geogebra.org/3d/ebt6eb7f
    """
    from chimerax.geometry import Place
    from numpy import arccos, arctan2, array, dot, pi, radians, sqrt, vstack
    from numpy.linalg import norm

    b = array((1, 0))
    Ct = h * b + k * dot(rot2(radians(60)), b)
    Cq = H * dot(rot2(radians(60)), b) + K * dot(rot2(radians(120)), b)

    phi = (1 + sqrt(5)) / 2.0
    b = 0.5
    a = phi * b
    ivarray = array((
        (0, b, -a),     # A
        (-b, a, 0),     # B
        (b, a, 0),      # C
        (a, 0, -b),     # D
        (0, -b, -a),    # E
        (-a, 0, -b),    # F
    ))
    theta = pi / 2 - arctan2((1 / (2 * phi)), 0.5)
    ivarray = Place(matrix=rot3_x(theta)).transform_points(ivarray)

    s = norm(Cq) / norm(Ct)
    alpha = arccos(dot(Ct, Cq) / (norm(Ct) * norm(Cq)))
    v1 = ivarray[1]  # B
    tv = v1 + Place(matrix=rot3_z(-alpha)).transform_vector(s * array([1, 0, 0]))

    r1 = norm(v1 - array((0, v1[1], 0)))
    r2 = v1[1] - tv[1]
    v6z = sqrt(r1 * r1 - tv[0] * tv[0])
    v6 = array((tv[0], v1[1] - sqrt(-(v1[2] * v1[2]) + 2 * v1[2] * v6z + r2 * r2 - v6z * v6z), v6z))  # G

    placer = Place(matrix=rot3_y(radians(72)))
    v7 = placer.transform_points(array([v6]))  # H
    v8 = placer.transform_points(v7)           # I
    v9 = placer.transform_points(v8)           # J
    vA = placer.transform_points(v9)           # K

    ivarray = vstack((ivarray, v6, v7[0], v8[0], v9[0], vA[0]))
    ivarray -= array((0, (v1[1] + v6[1]) / 2, 0))
    ivarray = vstack((ivarray, -ivarray[0]))   # L <- -A

    # TODO: replace with numbers
    from string import ascii_uppercase
    itarray = (
        "ABC", "ACD", "ADE", "AEF", "AFB",  # cap Δ
        "LGH", "LHI", "LIJ", "LJK", "LKG",  # cap ∇
        "CBG", "DCH", "EDI", "FEJ", "BFK",  # mid ∇
        "HGC", "GKB", "KJF", "JIE", "IHD",  # mid Δ
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)

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

    v6 = min(circle_cylinder_intersection(e, uxe, center, r_cir, r_cyl), key=itemgetter(1))  # G
    placer = Place(matrix=rot3_y(radians(120)))
    v7 = placer.transform_points(array([v6]))  # H
    v8 = placer.transform_points(v7)           # I

    yval = v6[1] - (ivarray[0][1] - ivarray[3][1])  # y(G) - (y(A) - y(D))
    cvec = dot(rot3_y(radians(60)), array((v6[0], yval, v6[2], 1))) - array((0, yval, 0))
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
        "BID", "CGE", "AHF",  # mid Δ 1
        "ILD", "GJE", "HKF",  # mid ∇ 1
        "DGC", "EHA", "FIB",  # mid ∇ 2
        "DLG", "EJH", "FKI"   # mid Δ 2
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)

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

    v6 = min(circle_cylinder_intersection(e, uxe, center, r_cir, r_cyl), key=itemgetter(1))
    placer180 = Place(matrix=rot3_y(radians(180)))
    v7 = placer180.transform_points(array([v6]))[0]

    temp = array((ivarray[2][0], 0, ivarray[2][2]))
    theta = arccos(dot(ivarray[5], temp) / (norm(ivarray[5]) * norm(temp)))
    placer = Place(matrix=rot3_y(theta))
    temp = placer.transform_points(array([v6]))[0] - array((0, ivarray[2][1], 0))
    c1 = array((0, temp[1], 0))
    v8 = c1 + ivarray[2][2] * (temp - c1)
    v9 = placer180.transform_points(array([v8]))[0]

    temp = Place(matrix=rot3_y(theta + radians(90))).transform_points(array([v6]))[0] - array((0, ivarray[1][1], 0))
    c2 = array((0, temp[1], 0))
    vA = c2 + ivarray[1][0] * (temp - c2)
    vB = placer180.transform_points(array([vA]))[0]

    ivarray = vstack((*ivarray, v6, v7, v8, v9, vA, vB))
    ivarray -= array((0, (ivarray[5][1] + v6[1]) / 2, 0))

    # TODO: replace with numbers
    from string import ascii_uppercase
    itarray = (
        "ABC", "ACE", "BAD", "BDF", "LKJ", "LJG", "KLI", "KIH",
        "EJK", "JEC", "CGJ", "GCB", "FIL", "IFD", "DHI", "HDA",
        "AEH", "KHE", "BFG", "LGF"
    )
    itarray = tuple(tuple(map(ascii_uppercase.find, tri)) for tri in itarray)

    return ivarray, itarray


def all_subclasses(cls):
    # https://stackoverflow.com/a/3862957
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in all_subclasses(c))


class HKTriangle(object):
    id = 1

    def __init__(self):
        self.hex_corner_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))

    def corners(self, h, k, H, K):
        return ((0, 0), (3 * h, 3 * k), (-3 * K, 3 * (H + K)))

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        tri = ((3 * h0, 3 * k0), (3 * h0 + h1o, 3 * k0 + k1o), (3 * h0 + h2o, 3 * k0 + k2o))
        # print(*((*e, *tri[(i+1) % 3]) for i, e in enumerate(tri)), sep="\n")
        yield triangle_intersection(tri, corners, 2)

    def walk(self, h, k, H=None, K=None, mode=1):
        if mode == 1:
            kmax, corners = max(k, h + k), self.corners(h, k, h, k)
        elif mode == 2:
            kmax, corners = max(K, H + K), self.corners(h, k, H, K)
        elif mode == 3:
            kmax, corners = max(K, h + K), self.corners(h, k, K, h)
        else:
            raise ValueError("mode must be in [1, 3]")
        # print(*((*e, *corners[(i+1) % 3]) for i, e in enumerate(corners)), sep="\n")
        yield from (
            (ele[0], ele[1])
            for k0 in range(kmax + 1)
            for h0 in range(-k0, h + 1)
            for c in range(6)
            for ele in self.corner(h0, k0, c, corners)
            if ele
        )


class HKTriangleDual(HKTriangle):
    id = 2

    def __init__(self):
        super().__init__()
        self.hex_corner_offset = ((3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3))


class HKTriangleTrihex(HKTriangle):
    id = 3

    def __init__(self):
        super().__init__()
        self.hex_corner_offset = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))

    def corners(self, h, k, H, K):
        return ((0, 0), (2 * h, 2 * k), (-2 * K, 2 * (H + K)))

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        tri = ((2 * h0, 2 * k0), (2 * h0 + h1o, 2 * k0 + k1o), (2 * h0 + h2o, 2 * k0 + k2o))
        yield triangle_intersection(tri, corners, 2)


class HKTriangleTrihexDual(HKTriangle):
    id = 4

    def __init__(self):
        super().__init__()

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        tri = ((3 * h0, 3 * k0), (3 * h0 + h1o, 3 * k0 + k1o), (3 * h0 + h2o, 3 * k0 + k2o))
        yield triangle_intersection(tri, corners, 1)


class HKTriangleSnub(HKTriangle):
    id = 5

    def __init__(self):
        super().__init__()
        self.tri_corner_offset = ((1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1))

    def corners(self, h, k, H, K):
        return ((0, 0), (5 * h + k, 4 * k - h), (-4 * K + H, 5 * K + 4 * H))

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h1t, k1t = self.tri_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        h2t, k2t = self.tri_corner_offset[(c + 1) % 6]
        tri = (
            (5 * h0 + k0, 4 * k0 - h0),
            (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o),
            (5 * h0 + k0 + h2o, 4 * k0 - h0 + k2o)
        )
        yield triangle_intersection(tri, corners, 2)
        tri = (
            (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o),
            (5 * h0 + k0 + h1o + h1t, 4 * k0 - h0 + k1o + k1t),
            (5 * h0 + k0 + h1o + h2t, 4 * k0 - h0 + k1o + k2t)
        )
        yield triangle_intersection(tri, corners, 2)
        tri = (
            (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o),
            (5 * h0 + k0 + h1o + h1t, 4 * k0 - h0 + k1o + k1t),
            (5 * h0 + k0 + h1o + h2t, 4 * k0 - h0 + k1o + k2t)
        )
        yield triangle_intersection(tri, corners, 3)


class HKTriangleSnubDual(HKTriangleSnub):
    id = 6

    def __init__(self):
        super().__init__()
        self.hex_corner_offset = ((2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2), (2, -2))
        self.tri_start_offset = ((2, -1), (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2))
        self.tri_corner_offset = ((1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1))

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        hst, kst = self.tri_start_offset[c]
        h1t, k1t = self.tri_corner_offset[c]
        h2t, k2t = self.tri_corner_offset[(c + 1) % 6]
        h3t, k3t = self.tri_corner_offset[(c + 2) % 6]
        tri = (
            (5 * h0 + k0, 4 * k0 - h0),
            (5 * h0 + k0 + h1o, 4 * k0 - h0 + k1o),
            (5 * h0 + k0 + h2o, 4 * k0 - h0 + k2o)
        )
        yield triangle_intersection(tri, corners, 1)
        tri = (
            (5 * h0 + k0 + hst, 4 * k0 - h0 + kst),
            (5 * h0 + k0 + hst + h1t, 4 * k0 - h0 + kst + k1t),
            (5 * h0 + k0 + hst + h2t, 4 * k0 - h0 + kst + k2t)
        )
        yield triangle_intersection(tri, corners, 2)
        tri = (
            (5 * h0 + k0 + hst, 4 * k0 - h0 + kst),
            (5 * h0 + k0 + hst + h2t, 4 * k0 - h0 + kst + k2t),
            (5 * h0 + k0 + hst + h3t, 4 * k0 - h0 + kst + k3t)
        )
        yield triangle_intersection(tri, corners, 2)


class HKTriangleRhomb(HKTriangle):
    id = 7

    def __init__(self):
        super().__init__()
        self.tri_corner_offset = ((2, -2), (2, 0), (0, 2), (-2, 2), (-2, 0), (0, -2))

    def corners(self, h, k, H, K):
        return ((0, 0), ((3 + 2) * h, (3 + 2) * k), (-(3 + 2) * K, (3 + 2) * (H + K)))

    def corner(self, h0, k0, c, corners):
        # TODO: fix grid to use square instead of rectangle
        h1o, k1o = self.hex_corner_offset[c]
        h1t, k1t = self.tri_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        h2t, k2t = self.tri_corner_offset[(c + 1) % 6]
        tri = (
            ((3 + 2) * h0, (3 + 2) * k0),
            ((3 + 2) * h0 + h1o, (3 + 2) * k0 + k1o),
            ((3 + 2) * h0 + h2o, (3 + 2) * k0 + k2o)
        )
        yield triangle_intersection(tri, corners, 2)
        tri = (
            ((3 + 2) * h0 + h1o, (3 + 2) * k0 + k1o),
            ((3 + 2) * h0 + h1o + h1t, (3 + 2) * k0 + k1o + k1t),
            ((3 + 2) * h0 + h1o + h2t, (3 + 2) * k0 + k1o + k2t)
        )
        yield triangle_intersection(tri, corners, 1)


class HKTriangleRhombDual(HKTriangle):
    id = 8

    def __init__(self):
        super().__init__()
        self.hex_corner_offset = ((3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3))
        self.tri_corner_offset = ((-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1), (1, 1))

    def corners(self, h, k, H, K):
        return ((0, 0), (6 * h, 6 * k), (-6 * K, 6 * (H + K)))

    def corner(self, h0, k0, c, corners):
        h1o, k1o = self.hex_corner_offset[c]
        h1t, k1t = self.tri_corner_offset[c]
        h2o, k2o = self.hex_corner_offset[(c + 1) % 6]
        tri = (
            (6 * h0, 6 * k0),
            (6 * h0 + h1o, 6 * k0 + k1o),
            (6 * h0 + h2o, 6 * k0 + k2o)
        )
        yield triangle_intersection(tri, corners, 1)
        tri = (
            (6 * h0 + h1o, 6 * k0 + k1o),
            (6 * h0 + h2o, 6 * k0 + k2o),
            (6 * h0 + h1o + h1t, 6 * k0 + k1o + k1t)
        )
        yield triangle_intersection(tri, corners, 2)
