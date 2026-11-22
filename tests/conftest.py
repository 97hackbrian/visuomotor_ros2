"""Configuración común de pytest para la suite.

Aislamiento de la base de experimentos: ``visuomotor_core/utils/tracker.py`` escribe en ``experiments/experiments.db``
(versionada en git) salvo que ``VISUOMOTOR_DB_PATH`` apunte a otra base. Sin este aislamiento, cada corrida de pytest
reescribía en la base real filas de prueba con rutas ``/tmp/test_zarr_merge_*`` (tabla ``datasets``, desde el
2026-11-02; limpiadas el 2026-11-22). Los subprocesos que lanzan las pruebas heredan la variable.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_experiments_db(tmp_path_factory):
    previous = os.environ.get("VISUOMOTOR_DB_PATH")
    os.environ["VISUOMOTOR_DB_PATH"] = str(tmp_path_factory.mktemp("experiments_db") / "experiments_test.db")
    yield
    if previous is None:
        os.environ.pop("VISUOMOTOR_DB_PATH", None)
    else:
        os.environ["VISUOMOTOR_DB_PATH"] = previous
