import os
import pathlib
import unittest

import pytest

os.chdir(pathlib.Path.cwd() / 'test/test_model')

pytest.main()
