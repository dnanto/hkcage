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


def hkcagetest(session, h, k, H=1, K=1, edge_radius=100.0, color=(255, 0, 255, 255), replace=True):
    from .cageTest import show_hk_lattice
    show_hk_lattice(session, h, k, H, K, edge_radius, color, replace)


def register_hkcagetest_command(logger):
    from chimerax.core.commands import CmdDesc, register, NonNegativeIntArg, \
        FloatArg, EnumOf, Color8Arg, BoolArg

    from chimerax.geometry.icosahedron import coordinate_system_names
    desc = CmdDesc(
        required=[('h', NonNegativeIntArg),
                  ('k', NonNegativeIntArg)],
        keyword=[
            ('H', NonNegativeIntArg),
            ('K', NonNegativeIntArg),
            ('edge_radius', FloatArg),
            ('color', Color8Arg),
            ('replace', BoolArg)
        ],
        synopsis="Generate an icosahedral mesh."
    )
    register("hkcagetest", desc, hkcagetest, logger=logger)
