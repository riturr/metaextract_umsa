import unittest

import spacy

from demo import DemoApp


class MyTestCase(unittest.TestCase):
    def test_cover_page_text_extraction(self):
        app = DemoApp(spacy.blank('en'), None)
        page = app.extract_cover_page('pdf_samples/PG-8601.pdf')
        text = app.extract_text_from_cover_page(page)
        self.assertIn('ESTUDIO BASICO PARA LA EXTRACCIÓN DE FLAVONOIDES DEL', text)

    def test_abstract_page_text_extraction(self):
        app = DemoApp(spacy.blank('en'), None)
        page = app.extract_abstract_page('pdf_samples/PG-8601.pdf')
        text = app.extract_text_from_abstract_page(page)
        self.assertIn('En el presente proyecto de grado se realizó un estudio básico para la obtención de flavonoides a partir de los cálices de Chilto (Physalis peruviana L.).', text)

if __name__ == '__main__':
    unittest.main()
