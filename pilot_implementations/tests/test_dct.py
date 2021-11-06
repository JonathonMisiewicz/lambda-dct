import numpy as np
import pytest

import pilot_implementations as pilot
from pilot_implementations.scripts import o_dct

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

E_D2 = -74.71370634648927
E_D3U = -74.71254983207328
E_D4U = -74.71357368209092
E_D5U = -74.71357831044938
E_D3V = -74.71201483416361
E_D4V = -74.71357978993036
E_D4L = -74.71357361279318
E_DT2 = -74.71413729462124
E_D3T2U = -74.71295972715569
E_D3T2V = -74.71241528538496
E_D3T3U = -74.71288153147974
E_D3T3V = -74.71229177767464
E_D3T3SV = -74.71238464151016
E_D4T2L = -74.71399933687455
E_D4T2U = -74.71401040744817
E_D4T2V = -74.7140226669566
E_D4T3L = -74.71399470920358
E_D4T3U = -74.71392349406472
E_D4T3SV = -74.71398635900532
E_D4T3V = -74.71387833209495
E_D3T3L = -74.71413307648207
E_DQ2U = -74.71392234049418

@pytest.mark.parametrize("inp", [
    pytest.param({"energy": E_D2, "test": True, "solver": o_dct.spinorbital.odc12.simultaneous}, id='D2'),
    pytest.param({"energy": E_D2, "test": False, "solver": o_dct.unrestricted.odc12.simultaneous}, id='D2_SI'),
    pytest.param({"energy": E_D3U, "test": True, "solver": o_dct.spinorbital.D3U.simultaneous}, id='D3U'),
    pytest.param({"energy": E_D3U, "test": False, "solver": o_dct.unrestricted.D3U.simultaneous}, id='D3U_SI'),
    pytest.param({"energy": E_D4U, "test": True, "solver": o_dct.spinorbital.D4U.simultaneous}, id='D4U'),
    pytest.param({"energy": E_D4U, "test": False, "solver": o_dct.unrestricted.D4U.simultaneous}, id='D4U_SI'),
    pytest.param({"energy": E_D5U, "test": True, "solver": o_dct.spinorbital.D5U.simultaneous}, id='D5U'),
    pytest.param({"energy": E_D5U, "test": False, "solver": o_dct.unrestricted.D5U.simultaneous}, id='D5U_SI'),
    pytest.param({"energy": E_D3V, "test": True, "solver": o_dct.spinorbital.D3V.simultaneous}, id='D3V'),
    pytest.param({"energy": E_D3V, "test": False, "solver": o_dct.unrestricted.D3V.simultaneous}, id='D3V_SI'),
    pytest.param({"energy": E_D4V, "test": True, "solver": o_dct.spinorbital.D4V.simultaneous}, id='D4V'),
    pytest.param({"energy": E_D4V, "test": False, "solver": o_dct.unrestricted.D4V.simultaneous}, id='D4V_SI'),
    pytest.param({"energy": E_D4L, "test": True, "solver": o_dct.spinorbital.D4L.simultaneous}, id='D4L'),
    pytest.param({"energy": E_D4L, "test": False, "solver": o_dct.unrestricted.D4L.simultaneous}, id='D4L_SI'),
    pytest.param({"energy": E_DT2, "test": True, "solver": o_dct.spinorbital.DT2.simultaneous}, id='DT2'),
    pytest.param({"energy": E_DT2, "test": False, "solver": o_dct.unrestricted.DT2.simultaneous}, id='DT2_SI'),
    pytest.param({"energy": E_D3T2U, "test": True, "solver": o_dct.spinorbital.D3T2U.simultaneous}, id='D3T2U'),
    pytest.param({"energy": E_D3T2U, "test": False, "solver": o_dct.unrestricted.D3T2U.simultaneous}, id='D3T2U_SI'),
    pytest.param({"energy": E_D3T3U, "test": True, "solver": o_dct.spinorbital.D3T3U.simultaneous}, id='D3T3U'),
    pytest.param({"energy": E_D3T3U, "test": False, "solver": o_dct.unrestricted.D3T3U.simultaneous}, id='D3T3U_SI'),
    pytest.param({"energy": E_D4T2U, "test": True, "solver": o_dct.spinorbital.D4T2U.simultaneous}, id='D4T2U'),
    pytest.param({"energy": E_D4T2U, "test": False, "solver": o_dct.unrestricted.D4T2U.simultaneous}, id='D4T2U_SI'),
    pytest.param({"energy": E_D4T3U, "test": True, "solver": o_dct.spinorbital.D4T3U.simultaneous}, id='D4T3U'),
    pytest.param({"energy": E_D4T3U, "test": False, "solver": o_dct.unrestricted.D4T3U.simultaneous}, id='D4T3U_SI'),
    pytest.param({"energy": E_D3T2V, "test": True, "solver": o_dct.spinorbital.D3T2V.simultaneous}, id='D3T2V'),
    pytest.param({"energy": E_D3T2V, "test": False, "solver": o_dct.unrestricted.D3T2V.simultaneous}, id='D3T2V_SI'),
    pytest.param({"energy": E_D3T3V, "test": True, "solver": o_dct.spinorbital.D3T3V.simultaneous}, id='D3T3V'),
    pytest.param({"energy": E_D3T3V, "test": False, "solver": o_dct.unrestricted.D3T3V.simultaneous}, id='D3T3V_SI'),
    pytest.param({"energy": E_D4T2V, "test": True, "solver": o_dct.spinorbital.D4T2V.simultaneous}, id='D4T2V'),
    pytest.param({"energy": E_D4T2V, "test": False, "solver": o_dct.unrestricted.D4T2V.simultaneous}, id='D4T2V_SI'),
    pytest.param({"energy": E_D4T3V, "test": True, "solver": o_dct.spinorbital.D4T3V.simultaneous}, id='D4T3V'),
    pytest.param({"energy": E_D4T3V, "test": False, "solver": o_dct.unrestricted.D4T3V.simultaneous}, id='D4T3V_SI'),
    pytest.param({"energy": E_D3T3SV, "test": True, "solver": o_dct.spinorbital.D3T3SV.simultaneous}, id='D3T3SV'),
    pytest.param({"energy": E_D3T3SV, "test": False, "solver": o_dct.unrestricted.D3T3SV.simultaneous}, id='D3T3SV_SI'),
    pytest.param({"energy": E_D4T3SV, "test": True, "solver": o_dct.spinorbital.D4T3SV.simultaneous}, id='D4T3SV'),
    pytest.param({"energy": E_D4T3SV, "test": False, "solver": o_dct.unrestricted.D4T3SV.simultaneous}, id='D4T3SV_SI'),
    pytest.param({"energy": E_D4T2L, "test": True, "solver": o_dct.spinorbital.D4T2L.simultaneous}, id='D4T2L'),
    pytest.param({"energy": E_D4T2L, "test": False, "solver": o_dct.unrestricted.D4T2L.simultaneous}, id='D4T2L_SI'),
    pytest.param({"energy": E_D3T3L, "test": True, "solver": o_dct.spinorbital.D3T3L.simultaneous}, id='D3T3L'),
    pytest.param({"energy": E_D3T3L, "test": False, "solver": o_dct.unrestricted.D3T3L.simultaneous}, id='D3T3L_SI'),
    pytest.param({"energy": E_D4T3L, "test": True, "solver": o_dct.spinorbital.D4T3L.simultaneous}, id='D4T3L'),
    pytest.param({"energy": E_D4T3L, "test": False, "solver": o_dct.unrestricted.D4T3L.simultaneous}, id='D4T3L_SI'),
    pytest.param({"energy": E_DQ2U, "test": False, "solver": o_dct.spinorbital.DQ2U.simultaneous}, id='DQ2U'),
    pytest.param({"energy": E_DQ2U, "test": True, "solver": o_dct.unrestricted.DQ2U.simultaneous}, id='DQ2U_SI'),
    ]
)
def test_template(inp):
    vals = pilot.subspace(molecule, solver=inp["solver"], test=inp["test"], comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], inp["energy"], significant=10)
    if inp["test"]:
        np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)

