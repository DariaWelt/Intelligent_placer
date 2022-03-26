import os
from dataclasses import dataclass
from typing import Union

import pytest

from intelligent_placer_lib import check_image


@dataclass(frozen=True)
class IntelligentPlacerCase:
    image_path: Union[str, os.PathLike]
    result: bool


def load_cases(test_dir_path: Union[str, os.PathLike]):
    cases = []
    for root, dirs, files in os.walk(test_dir_path):
        for file in files:
            try:
                tmp = os.path.splitext(file)[-2]
                result = bool(int(os.path.splitext(file)[-2][-1]))
                cases.append(IntelligentPlacerCase(os.path.join(test_dir_path, file), result))
            except ValueError:
                raise ValueError('Invalid test image name. '
                                 'Last symbol should be 0 or 1 and should specify placer result/')
    return cases


CASES = load_cases('data')


@pytest.mark.parametrize('case', CASES, ids=str)
def test_intelligent_placer(case: IntelligentPlacerCase):
    assert check_image(case.image_path, verbose=True) == case.result
