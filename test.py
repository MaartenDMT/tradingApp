import os
import pathlib
import pytest
import unittest
os.chdir(pathlib.Path.cwd() / 'test')

pytest.main()
