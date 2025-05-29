"""
Miscellaneous Utility Functions
"""
import click
import warnings
import logging
from torch.utils.data import Dataset
def colorful_print(string: str, *args, **kwargs) -> None:
    logging.info(click.style(string, *args, **kwargs))

def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs))
