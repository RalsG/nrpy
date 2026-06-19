import os
import subprocess
import sys
from pathlib import Path

BHHAHA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BHHAHA_DIR.parents[3]


def test_spectre_spin_diagnostic_computes_z_modes_before_integrating() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()

    helper_idx = source.index("static int bah_compute_spectre_spin_potentials")
    call_idx = source.index(
        "bah_compute_spectre_spin_potentials(commondata, griddata, auxevol_gfs, spectre_spin_gfs)"
    )
    integration_idx = source.index(
        "#pragma omp parallel\n{{\n    // Private accumulators"
    )
    primme_method_idx = source.index(
        "primme_set_method(PRIMME_DEFAULT_MIN_TIME, &primme);"
    )
    primme_init_size_idx = source.index("primme.initSize = 3;")
    primme_seed_idx = source.index(
        "spectre_spin_seed_coordinate_reduced(N, Nred, red_to_full, x_ref, x_centroid, evecs_red);"
    )
    primme_call_idx = source.index("dprimme(evals, evecs_red, resnorms, &primme)")

    assert helper_idx < call_idx < integration_idx
    assert primme_method_idx < primme_init_size_idx < primme_seed_idx < primme_call_idx
    assert "REAL *restrict spectre_spin_gfs" in source
    assert "spectre_spin_gfs[IDX4(ZU0GF, i0, i1, i2)] = z_init;" in source
    assert "auxevol_gfs[IDX4(ZU0GF" not in source
    assert "dprimme(evals, evecs_red, resnorms, &primme)" in source
    assert "primme.massMatrixMatvec = spectre_spin_primme_M_matvec;" in source
    assert "primme.initSize = 3;" in source
    assert "evecs_red[i] = 0.0;" not in source
    assert "SpECTRE spin eigenproblem" in source
    assert "SpECTRE spin coordinate seed" in source
    assert "SpECTRE spin PRIMME setup" in source
    assert "target_grad_norm = area * area / (6.0 * M_PI)" in source
    assert "K z = lambda M z" in source
    assert "return spin_potential_status;" in source


def test_final_diagnostics_treats_spectre_spin_failure_as_optional() -> None:
    source = (BHHAHA_DIR / "diagnostics.py").read_text()

    assert "commondata->error_flag = bah_diagnostics_spectre_spin" not in source
    assert (
        "const int spin_rc = bah_diagnostics_spectre_spin(commondata, griddata);"
        in source
    )
    assert "continuing without spin output" in source
    assert "BHAHAHA_DIAGNOSTIC_UNAVAILABLE" in source


def test_spectre_spin_gridfunctions_are_not_registered_as_auxevol() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()

    for gf_name in ["SE_qDD", "SE_XD", "zU"]:
        gf_idx = source.index(f'"{gf_name}"')
        registration_slice = source[gf_idx : gf_idx + 220]
        assert 'group="AUXEVOL"' not in registration_slice

    assert 'gf_array_name="spectre_spin_gfs"' in source
    assert (
        "_restore_private_spectre_spin_gridfunctions(saved_spectre_spin_gfs)" in source
    )


def test_spectre_spin_scratch_is_poisoned_and_checked() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()

    assert "spectre_spin_gfs[idx] = (REAL)NAN;" in source
    assert "static int spectre_spin_check_finite_scratch_gfs" in source
    assert '"physical precompute"' in source
    assert '"ghost-zone fill"' in source
    assert '"spin-potential solve"' in source
    assert "WARNING: SpECTRE spin scratch check failed after %s" in source


def test_spectre_spin_output_is_dimensionless_chi() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()
    equations = (
        BHHAHA_DIR.parents[2]
        / "equations"
        / "general_relativity"
        / "bhahaha"
        / "SpECTRESpinEstimate.py"
    ).read_text()
    file_output = (BHHAHA_DIR / "diagnostics_file_output.py").read_text()

    assert "REAL S_U[3]" in source
    assert "REAL chi_U[3]" in source
    assert "const REAL M_irr_squared = A / (16.0 * M_PI);" in source
    assert (
        "const REAL M_horizon_squared = M_irr_squared + S * S / (4.0 * M_irr_squared);"
        in source
    )
    assert "chi_U[i] = S_U[i] / M_horizon_squared;" in source
    assert "bhahaha_diags->spin_chi_x_spectre = chi_U[0];" in source
    assert "bhahaha_diags->spin_chi_x_spectre = S_U[0];" not in source
    assert "christodoulou_mass_squared_from_area_and_spin" in equations
    assert "chiU_nominal" in equations
    assert (
        "Dimensionless spin x-component (based on spin function Omega)" in file_output
    )


def test_spectre_spin_unavailable_sentinel_is_initialized() -> None:
    header = (BHHAHA_DIR / "BHaHAHA_header.h").read_text()
    find_horizon = (BHHAHA_DIR / "find_horizon.py").read_text()
    file_output = (BHHAHA_DIR / "diagnostics_file_output.py").read_text()

    assert "#define BHAHAHA_DIAGNOSTIC_UNAVAILABLE (-10.0)" in header
    assert "bah_initialize_diagnostics_struct" in header
    assert "diags->spin_chi_x_spectre = BHAHAHA_DIAGNOSTIC_UNAVAILABLE;" in header
    assert (
        "bah_initialize_diagnostics_struct(commondata.bhahaha_diagnostics);"
        in find_horizon
    )
    assert "diags->spin_chi_x_spectre != BHAHAHA_DIAGNOSTIC_UNAVAILABLE" in file_output


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


def test_bhahaha_generator_uses_internal_primme() -> None:
    source = (BHHAHA_DIR.parents[2] / "examples" / "bhahaha.py").read_text()

    assert "--primme-dir" not in source
    assert "PRIMME_DIR" not in source
    assert "primme_include_dir" not in source
    assert "include_dirs=[str(primme_include_dir)]" not in source
    assert "-lprimme" not in source

    assert "akv_primme_eigensolver" in source
    assert "patch_makefile_for_internal_akv_primme" in source
    assert "BHAHAHA_AKV_PRIMME_DOUBLE_ONLY" in source
    assert "BHAHAHA_AKV_PRIMME_INTERNAL_BLASLAPACK" in source
    assert "PRIMME_WITHOUT_FLOAT" in source
    assert "OBJCOPY ?= objcopy" in source
    assert "linkcheck" in source


def test_spectre_spin_uses_internal_primme_header() -> None:
    source = (BHHAHA_DIR / "spectre_spin_integrator.py").read_text()

    assert '#include "akv_primme.h"' in source
    assert '"akv_primme.h"' in source
    assert "#include <primme.h>" not in source
    assert '"primme.h"' not in source


def test_internal_primme_vendor_tree_is_present() -> None:
    vendor = BHHAHA_DIR / "akv_primme_eigensolver"

    assert (vendor / "akv_primme.h").is_file()
    assert (vendor / "primme_c.c").is_file()
    assert (vendor / "akv_internal_blaslapack.c").is_file()
    assert (vendor / "COPYING.txt").is_file()
    assert "Imported version: 3.2.3" in (vendor / "FROZEN_FORK_FROM.md").read_text()


def test_bhahaha_generation_copies_internal_primme_without_env(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("PRIMME_DIR", None)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if "PYTHONPATH" not in env
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "nrpy.examples.bhahaha",
            "--outrootdir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    project = tmp_path / "BHaHAHA"
    assert (project / "akv_primme_eigensolver" / "primme_c.c").is_file()
    assert (project / "akv_primme_eigensolver" / "akv_primme.h").is_file()

    makefile = (project / "Makefile").read_text()
    assert "-Iakv_primme_eigensolver" in makefile
    assert "akv_primme_eigensolver/primme_c.o" in makefile
    assert "akv_primme_eigensolver/akv_internal_blaslapack.o" in makefile
    assert "OBJCOPY ?= objcopy" in makefile
    assert "linkcheck" in makefile
