# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def hkcagetest(session, h, k, H = 1, K = 1, symmetry = 5, radius = 1, tile="hex", color = (255, 255, 255, 255), sphere_factor = 0, edge_radius = None, mesh = False, replace = True):
    if h == 0 and k == 0:
        from chimerax.core.errors import UserError
        raise UserError('h and k must be positive, got %d %d' % (h,k))

    from .cageTest import show_hk_lattice
    show_hk_lattice(session, h, k, H, K, symmetry, radius, tile, color, sphere_factor, edge_radius, mesh, replace)

# -----------------------------------------------------------------------------
#
def register_hkcagetest_command(logger):
    from chimerax.core.commands import (BoolArg, CmdDesc, Color8Arg, EnumOf,
                                        FloatArg, NonNegativeIntArg, StringArg,
                                        register)
    from chimerax.geometry.icosahedron import coordinate_system_names

    desc = CmdDesc(
        required = [
            ('h', NonNegativeIntArg),
            ('k', NonNegativeIntArg),
            ('H', NonNegativeIntArg),
            ('K', NonNegativeIntArg)
        ],
        keyword = [
            ('symmetry', NonNegativeIntArg),
            ('radius', FloatArg),
            ('tile', StringArg),
            ('color', Color8Arg),
            ('sphere_factor', FloatArg),
            ('edge_radius', FloatArg),
            ('mesh', BoolArg),
            ('replace', BoolArg)
        ],
        synopsis = 'Create icosahedral capsid mesh.'
    )
    register('hkcagetest', desc, hkcagetest, logger=logger)
