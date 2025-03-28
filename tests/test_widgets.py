import numpy as np
import pytest
from unittest.mock import patch
import ipysim.widgets as maglev_widgets


@pytest.mark.solara
@pytest.mark.parametrize(
    "kp_value, kd_value",
    [
        (100.0, 20.0),
        (300.0, 50.0),
        (600.0, 75.0),
        (800.0, 100.0),
        (1000.0, 200.0),
    ]
)
def test_maglev_slider_cases(solara_test, page_session, kp_value, kd_value):
    with patch("ipysim.widgets.simulate_maglev") as mock_sim:
        mock_sim.return_value = (
            np.linspace(0, 1, 1000),
            np.zeros((1000, 6))
        )

        solara_test.render(maglev_widgets.MaglevControl())

        kp_slider = page_session.get_by_role("slider", name="Kp")
        kd_slider = page_session.get_by_role("slider", name="Kd")

        kp_slider.wait_for()
        kd_slider.wait_for()

        def bump_slider(slider, from_val, to_val, step):
            steps = int(round((to_val - from_val) / step))
            key = "ArrowRight" if steps >= 0 else "ArrowLeft"
            for _ in range(abs(steps)):
                slider.press(key)

        bump_slider(kp_slider, 600.0, kp_value, 10.0)
        bump_slider(kd_slider, 30.0, kd_value, 5.0)

        page_session.wait_for_timeout(300)

        mock_sim.assert_called_with(
            kp_value,
            kd_value,
            1.0,
            0.001,
            maglev_widgets.default_state0,
            maglev_widgets.default_params,
        )
