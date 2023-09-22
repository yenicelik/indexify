import unittest
from indexify_extractors.entity_extractor import EntityExtractor


class TestEntityExtractor(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEntityExtractor, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls._entityextractor = EntityExtractor()

    def test_extractor(self):
        input = "My name is Wolfgang and I live in Berlin"
        entities = self._entityextractor.extract(input)
        print(entities)
        self.assertEqual(len(entities), 2)

if __name__ == "__main__":
    unittest.main()
