# -----------------------------------------------------------------------------
# Produce hexagonal lattices on an icosahedron.  The hexagons are bent where
# they cross the edges of the icosahedron.
#
# These lattices are described at
#
#       http://viperdb.scripps.edu/icos_server.php?icspage=paradigm


def show_hk_lattice(session, h, k, H, K, symmetry=5, radius=100.0, tile="hex", color=(255, 255, 255, 255), sphere_factor=0, edge_radius=1, replace=True):
    from chimerax.core.models import Surface
    from chimerax.markers.cmd import markers_from_mesh
    from pydemocapsid.democapsid import calc_ico, calc_lattice, meshes_to_chimerax


    surfaces = []
    meshes = meshes_to_chimerax(calc_ico((h, k, H, K), calc_lattice(tile, radius), a=symmetry, s=sphere_factor))
    for i, j, vertices, triangles, edge_mask in meshes: 
        surface = Surface(f"Polygon[{i}, {j}]", session)
        surface.set_geometry(vertices, None, triangles)
        surface.color = color
        surface.display_style = surface.Mesh
        surface.edge_mask = edge_mask
        surfaces.append(surface)

    name = f"Capsid[{h}, {k}, {H}, {K}, symmetry={symmetry}, tile={tile}]"
    markers = _cage_markers(session, name) if replace else None
    model = markers_from_mesh(session, surfaces, edge_radius=edge_radius, color=color, markers=markers)
    model.name = name
    model.hkcage = True
    markers and markers._prev_markers.delete()

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
