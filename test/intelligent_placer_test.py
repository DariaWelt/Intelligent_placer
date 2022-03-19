import json
import os
from dataclasses import dataclass
from typing import Union, List

import pytest

from intelligent_placer_lib import check_image


@dataclass(frozen=True)
class IntelligentPlacerCase:
    image_path: Union[str, os.PathLike]
    polygon_data: List
    polygon_mode: str
    result: bool


def load_cases(test_specification_path: Union[str, os.PathLike]):
    cases = []
    dir_path = os.path.dirname(test_specification_path)
    with open(test_specification_path) as f:
        data = json.load(f)
    if data.get('data') is None:
        raise ValueError('Invalid test specification. Example of specification json in /examples.test_spec.json ')
    for test in data['data']:
        cases.append(IntelligentPlacerCase(os.path.join(dir_path, test['path']), test['polygon']['data'], test['polygon']['mode'],
                                           test['groundTruth']))
    return cases


CASES = load_cases('data/image_spec.json')


@pytest.mark.parametrize('case', CASES, ids=str)
def test_intelligent_placer(case: IntelligentPlacerCase):
    assert check_image(case.image_path, case.polygon_data, case.polygon_mode) == case.result
