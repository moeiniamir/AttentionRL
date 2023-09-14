from pack_existing_segs import *
from abc import ABC, abstractmethod

class AbstractHistory(ABC):
    @abstractmethod
    def append(self, patch, row, col):
        pass

    @abstractmethod
    def get_history_dict(self):
        pass
