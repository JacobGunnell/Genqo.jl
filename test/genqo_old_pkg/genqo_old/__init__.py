try:
    from .genqo_old_dev import tools, TMSV, SPDC, ZALM, DWDM, SIGSAG, SIGSAG_BS, ZEROSAG, QM, TYP_PARAMS
    GENQO_OLD_DEV = True
except ModuleNotFoundError:
    from .genqo_old_pip import tools, TMSV, SPDC, ZALM, SIGSAG_BS, TYP_PARAMS
    GENQO_OLD_DEV = False
