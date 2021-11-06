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

E_D2 = -74.71451994537853
E_D3U = -74.7132519570999
E_D3V = -74.71267115790675
E_D4L = -74.71359969490916
E_D4U = -74.71356449866714
E_D4V = -74.71355137590807
E_D5U = -74.71356911296462
E_D5V = -74.71358636753799
E_DT2U = -74.71506711419485
E_DT2L = -74.71498274149191
E_DT2V = -74.71516631225427
E_D3T2U = -74.71376413843635
E_D3T2V = -74.71325130643126
E_D3T2SV = -74.71309653796828
E_D3T3L = -74.71497723399649
E_D3T3U = -74.71366888458698
E_D3T3SV = -74.71307113184680
E_D3T3V = -74.71306706422961 
E_D4T2L = -74.71403802137613
E_D4T2U = -74.7140914330804
E_D4T2V = -74.71417772881043
E_D4T2SV = -74.71400529929305
E_D4T3L = -74.71403324452561
E_D4T3U = -74.71399275731576 
E_D4T3V = -74.71397304494529 
E_D4T3SV = -74.71397793057569

@pytest.mark.parametrize("inp", [
    pytest.param({"energy": E_D2, "test": True, "solver": taylor.o_spinorbital.D2.simultaneous}, id='D2'),
    pytest.param({"energy": E_D2, "test": False, "solver": taylor.o_unrestricted.D2.simultaneous}, id='D2_SI'),
    pytest.param({"energy": E_D3U, "test": True, "solver": taylor.o_spinorbital.D3U.simultaneous}, id='D3U'),
    pytest.param({"energy": E_D3U, "test": False, "solver": taylor.o_unrestricted.D3U.simultaneous}, id='D3U_SI'),
    pytest.param({"energy": E_D4U, "test": True, "solver": taylor.o_spinorbital.D4U.simultaneous}, id='D4U'),
    pytest.param({"energy": E_D4U, "test": False, "solver": taylor.o_unrestricted.D4U.simultaneous}, id='D4U_SI'),
    pytest.param({"energy": E_D5U, "test": True, "solver": taylor.o_spinorbital.D5U.simultaneous}, id='D5U'),
    pytest.param({"energy": E_D5U, "test": False, "solver": taylor.o_unrestricted.D5U.simultaneous}, id='D5U_SI'),
    pytest.param({"energy": E_D3V, "test": True, "solver": taylor.o_spinorbital.D3V.simultaneous}, id='D3V'),
    pytest.param({"energy": E_D3V, "test": False, "solver": taylor.o_unrestricted.D3V.simultaneous}, id='D3V_SI'),
    pytest.param({"energy": E_D4V, "test": True, "solver": taylor.o_spinorbital.D4V.simultaneous}, id='D4V'),
    pytest.param({"energy": E_D4V, "test": False, "solver": taylor.o_unrestricted.D4V.simultaneous}, id='D4V_SI'),
    pytest.param({"energy": E_D5V, "test": True, "solver": taylor.o_spinorbital.D5V.simultaneous}, id='D5V'),
    pytest.param({"energy": E_D5V, "test": False, "solver": taylor.o_unrestricted.D5V.simultaneous}, id='D5V_SI'),
    pytest.param({"energy": E_D4L, "test": True, "solver": taylor.o_spinorbital.D4L.simultaneous}, id='D4L'),
    pytest.param({"energy": E_D4L, "test": False, "solver": taylor.o_unrestricted.D4L.simultaneous}, id='D4L_SI'),
    pytest.param({"energy": E_DT2U, "test": True, "solver": taylor.o_spinorbital.DT2U.simultaneous}, id='DT2U'),
    pytest.param({"energy": E_DT2U, "test": False, "solver": taylor.o_unrestricted.DT2U.simultaneous}, id='DT2U_SI'),
    pytest.param({"energy": E_D3T2U, "test": True, "solver": taylor.o_spinorbital.D3T2U.simultaneous}, id='D3T2U'),
    pytest.param({"energy": E_D3T2U, "test": False, "solver": taylor.o_unrestricted.D3T2U.simultaneous}, id='D3T2U_SI'),
    pytest.param({"energy": E_D3T3U, "test": True, "solver": taylor.o_spinorbital.D3T3U.simultaneous}, id='D3T3U'),
    pytest.param({"energy": E_D3T3U, "test": False, "solver": taylor.o_unrestricted.D3T3U.simultaneous}, id='D3T3U_SI'),
    pytest.param({"energy": E_D4T2U, "test": True, "solver": taylor.o_spinorbital.D4T2U.simultaneous}, id='D4T2U'),
    pytest.param({"energy": E_D4T2U, "test": False, "solver": taylor.o_unrestricted.D4T2U.simultaneous}, id='D4T2U_SI'),
    pytest.param({"energy": E_D4T3U, "test": True, "solver": taylor.o_spinorbital.D4T3U.simultaneous}, id='D4T3U'),
    pytest.param({"energy": E_D4T3U, "test": False, "solver": taylor.o_unrestricted.D4T3U.simultaneous}, id='D4T3U_SI'),
    pytest.param({"energy": E_DT2V, "test": True, "solver": taylor.o_spinorbital.DT2V.simultaneous}, id='DT2V'),
    pytest.param({"energy": E_DT2V, "test": False, "solver": taylor.o_unrestricted.DT2V.simultaneous}, id='DT2V_SI'),
    pytest.param({"energy": E_D3T2V, "test": True, "solver": taylor.o_spinorbital.D3T2V.simultaneous}, id='D3T2V'),
    pytest.param({"energy": E_D3T2V, "test": False, "solver": taylor.o_unrestricted.D3T2V.simultaneous}, id='D3T2V_SI'),
    pytest.param({"energy": E_D3T3V, "test": True, "solver": taylor.o_spinorbital.D3T3V.simultaneous}, id='D3T3V'),
    pytest.param({"energy": E_D3T3V, "test": False, "solver": taylor.o_unrestricted.D3T3V.simultaneous}, id='D3T3V_SI'),
    pytest.param({"energy": E_D4T2V, "test": True, "solver": taylor.o_spinorbital.D4T2V.simultaneous}, id='D4T2V'),
    pytest.param({"energy": E_D4T2V, "test": False, "solver": taylor.o_unrestricted.D4T2V.simultaneous}, id='D4T2V_SI'),
    pytest.param({"energy": E_D4T3V, "test": True, "solver": taylor.o_spinorbital.D4T3V.simultaneous}, id='D4T3V'),
    pytest.param({"energy": E_D4T3V, "test": False, "solver": taylor.o_unrestricted.D4T3V.simultaneous}, id='D4T3V_SI'),
    pytest.param({"energy": E_D3T2SV, "test": True, "solver": taylor.o_spinorbital.D3T2SV.simultaneous}, id='D3T2SV'),
    pytest.param({"energy": E_D3T2SV, "test": False, "solver": taylor.o_unrestricted.D3T2SV.simultaneous}, id='D3T2SV_SI'),
    pytest.param({"energy": E_D3T3SV, "test": True, "solver": taylor.o_spinorbital.D3T3SV.simultaneous}, id='D3T3SV'),
    pytest.param({"energy": E_D3T3SV, "test": False, "solver": taylor.o_unrestricted.D3T3SV.simultaneous}, id='D3T3SV_SI'),
    pytest.param({"energy": E_D4T2SV, "test": True, "solver": taylor.o_spinorbital.D4T2SV.simultaneous}, id='D4T2SV'),
    pytest.param({"energy": E_D4T2SV, "test": False, "solver": taylor.o_unrestricted.D4T2SV.simultaneous}, id='D4T2SV_SI'),
    pytest.param({"energy": E_D4T3SV, "test": True, "solver": taylor.o_spinorbital.D4T3SV.simultaneous}, id='D4T3SV'),
    pytest.param({"energy": E_D4T3SV, "test": False, "solver": taylor.o_unrestricted.D4T3SV.simultaneous}, id='D4T3SV_SI'),
    pytest.param({"energy": E_DT2L, "test": True, "solver": taylor.o_spinorbital.DT2L.simultaneous}, id='DT2L'),
    pytest.param({"energy": E_DT2L, "test": False, "solver": taylor.o_unrestricted.DT2L.simultaneous}, id='DT2L_SI'),
    pytest.param({"energy": E_D3T3L, "test": True, "solver": taylor.o_spinorbital.D3T3L.simultaneous}, id='D3T3L'),
    pytest.param({"energy": E_D3T3L, "test": False, "solver": taylor.o_unrestricted.D3T3L.simultaneous}, id='D3T3L_SI'),
    pytest.param({"energy": E_D4T2L, "test": True, "solver": taylor.o_spinorbital.D4T2L.simultaneous}, id='D4T2L'),
    pytest.param({"energy": E_D4T2L, "test": False, "solver": taylor.o_unrestricted.D4T2L.simultaneous}, id='D4T2L_SI'),
    pytest.param({"energy": E_D4T3L, "test": True, "solver": taylor.o_spinorbital.D4T3L.simultaneous}, id='D4T3L'),
    pytest.param({"energy": E_D4T3L, "test": False, "solver": taylor.o_unrestricted.D4T3L.simultaneous}, id='D4T3L_SI'),]
)
def test_template(inp):
    vals = pilot.subspace(molecule, solver=inp["solver"], test=inp["test"], comp_grad=False)[0]
    np.testing.assert_approx_equal(vals["energy"], inp["energy"], significant=10)
    if inp["test"]:
        np.testing.assert_allclose(vals["mu"], vals["deriv"], atol=1.0e-10)


