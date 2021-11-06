import numpy as np
import pytest

import pilot_implementations as pilot
from pilot_implementations.scripts import taylor

np.set_printoptions(precision=13, linewidth=200, suppress=True)

molecule = {
    "charge": +1,
    "num_unpaired": +1,
    "geom": [
         ('O', (0.000000000000, 0.000000000000, -0.143225816552)),
         ('H', (0.000000000000, 1.638036840407, 1.136548822547)),
         ('H', (0.000000000000, -1.638036840407, 1.136548822547))
         ],
    "basis": "sto-3g"
}

E_SD2 = -74.71471597383582
E_SD3L = -74.71455127608098
E_SD3U = -74.71322275201467
E_SD3V = -74.71262991596943
E_SD3SV = -74.71261320622956
E_SD4L = -74.71360502012332
E_SD4U = -74.71354488766906
E_SD4V = -74.71352488877952
E_SD4SV = -74.71351735573910
E_SD5L = -74.71360427805979
E_SD5U = -74.71355405177935
E_SD5V = -74.71356464935656
E_SD5SV = -74.71356637848228

@pytest.mark.parametrize("inp", [
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_spinorbital.SD2U.simultaneous, "check_minima": True}, id='SD2U'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_unrestricted.SD2U.simultaneous}, id='SD2U_SI'),
    pytest.param({"energy": E_SD3U, "test": False, "solver": taylor.singles_spinorbital.SD3U.simultaneous, "check_minima": True}, id='SD3U'),
    pytest.param({"energy": E_SD3U, "test": False, "solver": taylor.singles_unrestricted.SD3U.simultaneous}, id='SD3U_SI'),
    pytest.param({"energy": E_SD4U, "test": False, "solver": taylor.singles_spinorbital.SD4U.simultaneous, "check_minima": True}, id='SD4U'),
    pytest.param({"energy": E_SD4U, "test": False, "solver": taylor.singles_unrestricted.SD4U.simultaneous}, id='SD4U_SI'),
    pytest.param({"energy": E_SD5U, "test": False, "solver": taylor.singles_spinorbital.SD5U.simultaneous, "check_minima": True}, id='SD5U'),
    pytest.param({"energy": E_SD5U, "test": False, "solver": taylor.singles_unrestricted.SD5U.simultaneous}, id='SD5U_SI'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_spinorbital.SD2V.simultaneous, "check_minima": True}, id='SD2V'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_unrestricted.SD2V.simultaneous}, id='SD2V_SI'),
    pytest.param({"energy": E_SD3V, "test": False, "solver": taylor.singles_spinorbital.SD3V.simultaneous, "check_minima": True}, id='SD3V'),
    pytest.param({"energy": E_SD3V, "test": False, "solver": taylor.singles_unrestricted.SD3V.simultaneous}, id='SD3V_SI'),
    pytest.param({"energy": E_SD4V, "test": False, "solver": taylor.singles_spinorbital.SD4V.simultaneous, "check_minima": True}, id='SD4V'),
    pytest.param({"energy": E_SD4V, "test": False, "solver": taylor.singles_unrestricted.SD4V.simultaneous}, id='SD4V_SI'),
    pytest.param({"energy": E_SD5V, "test": False, "solver": taylor.singles_spinorbital.SD5V.simultaneous, "check_minima": True}, id='SD5V'),
    pytest.param({"energy": E_SD5V, "test": False, "solver": taylor.singles_unrestricted.SD5V.simultaneous}, id='SD5V_SI'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_spinorbital.SD2SV.simultaneous, "check_minima": True}, id='SD2SV'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_unrestricted.SD2SV.simultaneous}, id='SD2SV_SI'),
    pytest.param({"energy": E_SD3SV, "test": False, "solver": taylor.singles_spinorbital.SD3SV.simultaneous, "check_minima": True}, id='SD3SV'),
    pytest.param({"energy": E_SD3SV, "test": False, "solver": taylor.singles_unrestricted.SD3SV.simultaneous}, id='SD3SV_SI'),
    pytest.param({"energy": E_SD4SV, "test": False, "solver": taylor.singles_spinorbital.SD4SV.simultaneous, "check_minima": True}, id='SD4SV'),
    pytest.param({"energy": E_SD4SV, "test": False, "solver": taylor.singles_unrestricted.SD4SV.simultaneous}, id='SD4SV_SI'),
    pytest.param({"energy": E_SD5SV, "test": False, "solver": taylor.singles_spinorbital.SD5SV.simultaneous, "check_minima": True}, id='SD5SV'),
    pytest.param({"energy": E_SD5SV, "test": False, "solver": taylor.singles_unrestricted.SD5SV.simultaneous}, id='SD5SV_SI'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_spinorbital.SD2L.simultaneous, "check_minima": True}, id='SD2L'),
    pytest.param({"energy": E_SD2, "test": False, "solver": taylor.singles_unrestricted.SD2L.simultaneous}, id='SD2L_SI'),
    pytest.param({"energy": E_SD3L, "test": False, "solver": taylor.singles_spinorbital.SD3L.simultaneous, "check_minima": True}, id='SD3L'),
    pytest.param({"energy": E_SD3L, "test": False, "solver": taylor.singles_unrestricted.SD3L.simultaneous}, id='SD3L_SI'),
    pytest.param({"energy": E_SD4L, "test": False, "solver": taylor.singles_spinorbital.SD4L.simultaneous, "check_minima": True}, id='SD4L'),
    pytest.param({"energy": E_SD4L, "test": False, "solver": taylor.singles_unrestricted.SD4L.simultaneous}, id='SD4L_SI'),
    pytest.param({"energy": E_SD5L, "test": False, "solver": taylor.singles_spinorbital.SD5L.simultaneous, "check_minima": True}, id='SD5L'),
    pytest.param({"energy": E_SD5L, "test": False, "solver": taylor.singles_unrestricted.SD5L.simultaneous}, id='SD5L_SI'),
    ]
)
def test_template(inp):
    vals = pilot.subspace(molecule, solver=inp["solver"], test=inp["test"], omp_grad=False, check_minima=inp.get("check_minima", False))[0]
    np.testing.assert_approx_equal(vals["energy"], inp["energy"], significant=10)

