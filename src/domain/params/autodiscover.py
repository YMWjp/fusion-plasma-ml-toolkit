import importlib
import pkgutil

from src.domain.params import groups


# src/domain/params/groups/ 以下に置いたパラメータ関数が自動登録される。
# functions under src/domain/params/groups/ will be automatically registered.
def autodiscover():
    pkg = groups
    prefix = pkg.__name__ + "."
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix):
        importlib.import_module(modname)