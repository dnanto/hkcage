# -----------------------------------------------------------------------------
# Produce hexagonal lattices on an icosahedron.  The hexagons are bent where
# they cross the edges of the icosahedron.
#
# These lattices are described at
#
#       http://viperdb.scripps.edu/icos_server.php?icspage=paradigm


def show_hk_lattice(session, h, k, H, K, symmetry=5, radius=100.0, tile="hex", color=(255, 255, 255, 255), sphere_factor=0, edge_radius=None, mesh=False, replace=True):
    from democapsid.democapsid import calc_ico, calc_lattice, meshes_to_chimerax

    print(tile)
    meshes = meshes_to_chimerax(calc_ico((h, k, H, K), calc_lattice(tile, radius), a=symmetry, s=sphere_factor))

    if mesh:
        model = sm = _cage_surface(session, name, replace)
        sm.set_geometry(varray, None, tarray)
        sm.color = color
        sm.display_style = sm.Mesh
        sm.edge_mask = hex_edges  # Hide spokes of hexagons.
        sm.id is None and session.models.add([sm])
    else:
        # Make cage from markers.
        from chimerax.core.models import Surface
        from chimerax.markers.cmd import markers_from_mesh
        for i, j, vertices, triangles, edge_mask in meshes: 
            name = f"Polygon[{i}, {j}]"
            sm = Surface(name, session)
            sm.set_geometry(vertices, None, triangles)
            sm.color = color
            sm.display_style = sm.Mesh
            sm.edge_mask = edge_mask  # Hide spokes of hexagons.
            if edge_radius is None:
                edge_radius = .01 * radius
            mset = _cage_markers(session, name) if replace else None
            model = markers_from_mesh(session, [sm], color=color, edge_radius=edge_radius, markers=mset)
            model.name = name
            mset and mset._prev_markers.delete()

    model.hkcage = True

    return model


def _cage_markers(session, name):
    """_summary_

    Args:
        session (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    from chimerax.markers import MarkerSet
    mlist = [m for m in session.models.list(type=MarkerSet) if hasattr(m, 'hkcage')]
    if mlist:
        mset = mlist[0]
        mset._prev_markers = mset.atoms
        mset.name = name
        mset.hkcage = True
        return mset
    return None


def _cage_surface(session, name, replace):
    """Make new surface model or find an existing one.

    Args:
        session (_type_): _description_
        name (_type_): _description_
        replace (_type_): _description_

    Returns:
        _type_: _description_
    """
    sm = None
    from chimerax.core.models import Surface
    if replace:
        mlist = [m for m in session.models.list(type=Surface) if hasattr(m, 'hkcage')]
        if mlist:
            sm = mlist[0]
            sm.name = name
    if sm is None:
        sm = Surface(name, session)
    sm.hkcage = True
    return sm


def surface_geometry(triangles, tolerance=1e-5):
    """Calcualate surface geometry.

    Take a list of triangles where each triangle is specified by 3 xyz vertex
    positions and convert to a vertex and triangle array where the triangle
    array contains indices into the vertex array.  Vertices in the original
    triangle data that are close (within tolerance) are merged into a single
    vertex.

    Args:
        triangles (_type_): _description_
        tolerance (_type_, optional): _description_. Defaults to 1e-5.

    Returns:
        _type_: _description_
    """
    from numpy import array, intc, reshape
    from numpy import single as floatc
    varray = reshape(triangles, (3 * len(triangles), 3)).astype(floatc)

    uindex = {}
    unique = []
    from chimerax.geometry import find_close_points
    for v in range(len(varray)):
        if v not in uindex:
            _, i2 = find_close_points(varray[v:v + 1, :], varray, tolerance)
            for i in i2:
                if i not in uindex:
                    uindex[i] = len(unique)
            unique.append(varray[v])

    uvarray = array(unique, floatc)
    tlist = [(uindex[3 * t], uindex[3 * t + 1], uindex[3 * t + 2]) for t in range(len(triangles))]
    tarray = array(tlist, intc)

    return uvarray, tarray
