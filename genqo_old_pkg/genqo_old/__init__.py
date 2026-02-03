try:
    from .genqo_old_dev import tools, TMSV, SPDC, ZALM, DWDM, SIGSAG, SIGSAG_BS, ZEROSAG, QM, TYP_PARAMS
except ModuleNotFoundError:
    from .genqo_old_pip import tools, TMSV, SPDC, ZALM, TYP_PARAMS
