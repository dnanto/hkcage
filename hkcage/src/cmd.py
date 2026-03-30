from chimerax.core.commands import (BoolArg, CmdDesc, NonNegativeFloatArg,
                                    NonNegativeIntArg, PercentFloatArg,
                                    StringArg, TupleOf)


def hkmesh(session, h, k, H=None, K=None, symmetry=5, radius=1.0, tile="hex", color=(255, 255, 255, 255), sphere_factor=0, edge_radius=0.01, replace=True):
    from .mesh import show_hkmesh

    H, K = h if H is None else H, k if K is None else K
    show_hkmesh(session, h, k, H, K, symmetry, radius, tile, color, sphere_factor, edge_radius, replace)

cmd_desc = CmdDesc(
    required = [
        ("h", NonNegativeIntArg),
        ("k", NonNegativeIntArg),
    ],
    optional=[
        ("H", NonNegativeIntArg),
        ("K", NonNegativeIntArg),
    ],
    keyword = [
        ("symmetry", NonNegativeIntArg),
        ("radius", NonNegativeFloatArg),
        ("tile", StringArg),
        ("color", TupleOf(NonNegativeIntArg, 4)),
        ("sphere_factor", PercentFloatArg),
        ("edge_radius", NonNegativeFloatArg),
        ("replace", BoolArg),
    ]
)
