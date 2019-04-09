import os
import numpy as np


__all__ = ('_PseudoLogger', '_config', '_get_line_lists', '_gen_sex')


class _PseudoLogger:
    @staticmethod
    def info(s):
        print('INFO:', s)

    @staticmethod
    def warning(s):
        print('WARNING:', s)

    @staticmethod
    def error(s):
        print('ERROR:', s)


def _config(conf_path=None, nb=True):
    CONFIG = ('default.sex', 'default.conv', 'default.nnw', 'default.param')

    if conf_path is None:
        conf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 '_muselet_data')

    config_files = []
    for f in CONFIG:
        if nb:
            f = 'nb_' + f
        f = os.path.join(conf_path, f)
        config_files.append(f)

    return config_files


def _get_line_lists(conf_path=None):
    if conf_path is None:
        conf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 '_muselet_data')

    eml = dict(np.loadtxt(os.path.join(conf_path, 'emlines'),
                          dtype={'names': ('lambda', 'lname'),
                                 'formats': ('f', 'S20')}))
    eml2 = dict(np.loadtxt(os.path.join(conf_path, 'emlines_small'),
                           dtype={'names': ('lambda', 'lname'),
                                  'formats': ('f', 'S20')}))
    return eml, eml2


def _gen_sex(cmd_sex, f, cat_name, weight_name, nb=True, conf_path=None):
    conf, conv, nnw, param = _config(conf_path, nb)
    if not isinstance(f, str):
        f = f[0] + '","' + f[1]

    cmd = cmd_sex
    cmd += ' "{}"'.format(f)
    cmd += ' -c "{}"'.format(conf)
    cmd += ' -PARAMETERS_NAME "{}"'.format(param)
    cmd += ' -FILTER_NAME "{}"'.format(conv)
    cmd += ' -STARNNW_NAME "{}"'.format(nnw)
    cmd += ' -CATALOG_NAME "{}"'.format(cat_name)
    cmd += ' -CATALOG_TYPE ASCII_HEAD'
    cmd += ' -WEIGHT_IMAGE "{}"'.format(weight_name)

    return cmd