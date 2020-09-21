import pytest
from input import generate_overlay


@pytest.mark.parametrize("start_idx, nesting, output",
                         [
                             (
                                 0, [5, 4],
                                 [[i + j*4 for i in range(4)]
                                  for j in range(5)]
                             ),
                         ])
def test_generate_overlay(nesting, start_idx, output):
    assert generate_overlay(nesting, start_idx) == output
