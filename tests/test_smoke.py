"""Smoke tests verifying the package can be imported and exposes the expected version."""

import lifexp


def test_version():
    assert lifexp.__version__ == "0.1.0"


def test_subpackage_imports():
    import lifexp.core
    import lifexp.mortality
    import lifexp.lapse
    import lifexp.morbidity
    import lifexp.reinsurance
    import lifexp.expense
    import lifexp.graduation
    import lifexp.reporting
    import lifexp.datasets
