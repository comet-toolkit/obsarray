from obsarray.err_corr import register_err_corr_form, BaseErrCorrForm
import xarray as xr
import numpy as np
import obsarray


@register_err_corr_form("new")
class NewForm(BaseErrCorrForm):

    form = "new"
    is_systematic = True

    def build_matrix(self, idx: np.ndarray) -> np.ndarray:
        return "abc"


ds = xr.open_dataset("obs_example.nc")

ds.unc["temperature"]["u_sys_temperature"] = (
    ["x", "y", "time"],
    ds.temperature * 0.03,
    {
        "err_corr": [
            {
                "dim": "x",
                "form": "new",
                "params": [],
            },
            {
                "dim": "y",
                "form": "systematic",
                "params": [],
            },
            {
                "dim": "time",
                "form": "systematic",
                "params": [],
            },
        ]
    },
)

p = ds.unc["temperature"]["u_sys_temperature"]

pass
