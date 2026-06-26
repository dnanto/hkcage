from chimerax.core.toolshed import BundleAPI


class _HKMeshAPI(BundleAPI):

    api_version = 1

    # Override method for registering commands
    @staticmethod
    def register_command(bundle_info, command_info, logger):
        from chimerax.core.commands import register

        from . import cmd
        cmd_desc = cmd.cmd_desc
        cmd_desc.synopsis = command_info.synopsis
        register(command_info.name, cmd_desc, cmd.hkcage, logger=logger)


bundle_api = _HKMeshAPI()
