import unittest
from data_types.value import Value 

class TestValueOperations(unittest.TestCase):

    def test_value_addition(self):
        a, b = 17, 96
        res = Value(a) + Value(b)
        self.assertEqual(res.data, a + b)

    def test_value_int_addition(self):
        a, b = 17, 96
        res = Value(a) + b
        self.assertEqual(res.data, a + b)

    def test_int_value_addition(self):
        a, b = 17, 96
        res = a + Value(b)
        self.assertEqual(res.data, a + b)
