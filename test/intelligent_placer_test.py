import json
import os
from typing import Union

from intelligent_placer_lib import check_image


def test_intelligent_placer(test_specification_path: Union[str, os.PathLike]='data/image_spec.json'):
    with open(test_specification_path) as f:
        data = json.load(f)
    if data.get('data') is None:
        raise ValueError('Invalid test specification. Example of specification json in /examples.test_spec.json ')
    for test in data['data']:
        assert (check_image('data/'+test['path'], test['polygon']['data'], mode=test['polygon']['mode']) == test['groundTruth'])


if __name__ == '__main__':
    test_intelligent_placer('data/image_spec.json')
