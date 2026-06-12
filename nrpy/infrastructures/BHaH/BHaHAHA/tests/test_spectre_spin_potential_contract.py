from pathlib import Path


BHHAHA_DIR = Path(__file__).resolve().parents[1]


def test_spectre_spin_diagnostic_computes_z_modes_before_integrating() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()

    helper_idx = source.index("static int bah_compute_spectre_spin_potentials")
    call_idx = source.index("bah_compute_spectre_spin_potentials(commondata, griddata, auxevol_gfs)")
    integration_idx = source.index("#pragma omp parallel\n{{\n    // Private accumulators")

    assert helper_idx < call_idx < integration_idx
    assert "auxevol_gfs[IDX4(ZU0GF, i0, i1, i2)] = z_init;" in source
    assert "dprimme(evals, evecs_red, resnorms, &primme)" in source
    assert "primme.massMatrixMatvec = spectre_spin_primme_M_matvec;" in source
    assert "target_grad_norm = area * area / (6.0 * M_PI)" in source
    assert "K z = lambda M z" in source
    assert "return spin_potential_status;" in source


def test_final_diagnostics_propagates_spectre_spin_failure() -> None:
    source = (BHHAHA_DIR / "diagnostics.py").read_text()

    assert (
        "commondata->error_flag = bah_diagnostics_spectre_spin(commondata, griddata);"
        in source
    )
    assert (
        "if (commondata->error_flag != BHAHAHA_SUCCESS)\n"
        "        return;\n\n"
        "      // Display detailed final iteration diagnostics"
        in source
    )


def test_spectre_spin_potential_failure_has_error_code() -> None:
    source = (BHHAHA_DIR / "error_message.py").read_text()

    for error_code in [
        "DIAG_SPECTRE_SPIN_POTENTIAL_ERROR",
        "DIAG_SPECTRE_SPIN_POTENTIAL_MALLOC_ERROR",
        "DIAG_SPECTRE_SPIN_POTENTIAL_GEOMETRY_ERROR",
        "DIAG_SPECTRE_SPIN_POTENTIAL_PRIMME_ERROR",
        "DIAG_SPECTRE_SPIN_POTENTIAL_NORMALIZATION_ERROR",
    ]:
        assert error_code in source


def test_bhahaha_generator_requires_external_primme() -> None:
    source = (BHHAHA_DIR.parents[2] / "examples" / "bhahaha.py").read_text()

    assert "--primme-dir" in source
    assert 'os.environ.get("PRIMME_DIR")' in source
    assert 'Path(primme_include_dir, "primme.h").is_file()' in source
    assert "include_dirs=[str(primme_include_dir)]" in source
    assert "-L$PRIMME_DIR/lib -lprimme" in source
