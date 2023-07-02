"""Write down basic config here
"""
import os
from types import SimpleNamespace


class Config(SimpleNamespace):
    """SimpeNamespace with get method
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, k, replace=None):
        """Get method
        Args:
            k: Key attribute
            replace: Replacement
        Returns:
            v: Key or replacement
        """
        v = None
        try:
            v = self.__getattribute__(k)
        except AttributeError:
            v = replace
        return v
    
    def set(self, k, v):
        self.__setattr__(k, v)
    
    def update(self, dic):
        """Update config
        Args:
            dic: new dictionary
        Returns:
            None
        """

        if not isinstance(dic, dict):
            dic = vars(dic)
        
        assert isinstance(dic, dict), "Type error"

        for k, v in dic.items():
            self.set(k, v)


### head
cfg = Config(**{})

### body
# data path
cfg.root_path = "/data/private/recsys-challenge-2023/sharechat_recsys2023_data/"
cfg.train_file = os.path.join(cfg.root_path, "train/train.parquet")
cfg.test_file = os.path.join(cfg.root_path, "test/000000000000.csv")

# utils
cfg.seed = 42