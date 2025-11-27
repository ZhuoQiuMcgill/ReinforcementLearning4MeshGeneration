"""CLI entrypoints for mesh_rl (v2).

The concrete CLIs live in ``mesh_rl.cli.train_cli`` and
``mesh_rl.cli.eval_cli``. We deliberately avoid importing them here so
that running ``python -m mesh_rl.cli.train_cli`` does not trigger
spurious ``RuntimeWarning`` messages about the module already being
loaded via ``mesh_rl.cli``.
"""

__all__: list[str] = []
