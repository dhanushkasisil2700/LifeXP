"""lifexp: Actuarial life experience analysis toolkit."""

__version__ = "0.1.0"

# Trigger built-in table registration (A_1967_70) at package import time.
import lifexp.core.tables as _tables  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from lifexp.core.data_model import ClaimDataset, PolicyDataset  # noqa: E402
from lifexp.core.study_period import StudyPeriod                # noqa: E402
from lifexp.core.tables import TableRegistry                    # noqa: E402
from lifexp.mortality.study import MortalityStudy               # noqa: E402
from lifexp.lapse.study import LapseStudy                       # noqa: E402
from lifexp.morbidity.study import MorbidityStudy               # noqa: E402
from lifexp.reinsurance.study import RIStudy                    # noqa: E402
from lifexp.expense.study import ExpenseStudy                   # noqa: E402
from lifexp.reporting.html_report import HTMLReport             # noqa: E402
from lifexp.reporting.excel_report import ExcelReport           # noqa: E402

__all__ = [
    "PolicyDataset",
    "ClaimDataset",
    "StudyPeriod",
    "MortalityStudy",
    "LapseStudy",
    "MorbidityStudy",
    "RIStudy",
    "ExpenseStudy",
    "TableRegistry",
    "HTMLReport",
    "ExcelReport",
]
