from functools import lru_cache
from typing import Any
import xml.etree.ElementTree as ET
from PIL import Image
from PIL.ImageDraw import ImageDraw
from pytesseract import pytesseract
from pypdf import PdfReader
import gradio as gr
from unidecode import unidecode
import cv2
import numpy as np
import spacy
from spacy import displacy, Language
from spacy.tokens import Doc
from pdf2image import convert_from_path


class DemoApp:
    def __init__(self, ner_model: Language, llm_model: Any | None = None, pos_model: Language | None = None):
        self.ui = self.build_ui()
        self.ner_model = ner_model
        self.llm_model, self.tokenizer = llm_model if llm_model else (None, None)
        self.pos_model = pos_model

    @lru_cache(maxsize=1)
    def extract_abstract_page(self, pdf_file_path) -> Image:
        reader = PdfReader(pdf_file_path)
        pattern = "resumen"
        max_pages_to_scan = 25
        for page_index in range(0, max_pages_to_scan):
            if page_index > len(reader.pages) - 1:
                raise "No se pudo extraer el resumen del proyecto"
            page_text = unidecode(reader.pages[page_index].extract_text()).lower()
            is_page_toc = page_text.count('tabla') > 2 or page_text.count('figura') > 2 or page_text.count('.') > 50

            if pattern in page_text and not is_page_toc:
                image = convert_from_path(pdf_file_path, fmt='png', first_page=page_index + 1, last_page=page_index + 1)[0]
                print(f"Successfully extracted abstract page from {pdf_file_path}")
                image.save('abstract_page.jpg')
                return image
        raise "No se pudo extraer el resumen del proyecto"

    @lru_cache(maxsize=1)
    def extract_cover_page(self, pdf_file_path) -> Image:
        # get first page of PDF
        image = convert_from_path(pdf_file_path, fmt='png', first_page=0, last_page=1)[0]
        print(f"Successfully extracted cover page from {pdf_file_path}")
        # image.save('cover_page.jpg')
        return image

    def load_preview_images(self, pdf):
        if not pdf:
            return None, None
        abstract_page_img = self.extract_abstract_page(pdf)
        cover_page_img = self.extract_cover_page(pdf)
        return gr.Image(cover_page_img, width=400, height=600), gr.Image(abstract_page_img, width=400, height=600)

    def extract_text_from_abstract_page(self, abstract_page_img: Image) -> str:
        return pytesseract.image_to_string(abstract_page_img, lang="spa").replace('\n', ' ')

    def extract_text_from_cover_page(self, cover_page_img: Image) -> str:
        return pytesseract.image_to_string(self.mask_logos(cover_page_img), lang="spa")

    def mask_logos(self, image: Image) -> Image:
        cv_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        width = cv_image.shape[1]
        cv2_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, cv2_binary_image = cv2.threshold(cv2_gray, 200, 255, cv2.THRESH_BINARY)
        cv2_inverted_image = cv2.bitwise_not(cv2_binary_image)
        contours, _ = cv2.findContours(cv2_inverted_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        image1 = ImageDraw(image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (width/10)**2:
                x, y, w, h = cv2.boundingRect(contour)
                if w < width/3:
                    image1.rectangle([(x, y), (x + w, y + h)], fill="#ffff33")

        return image

    def recognize_entities(self, text) -> Doc:
        return self.ner_model(text)

    def render_doc_entities_as_html(self, doc: Doc) -> str:
        colors = {
            'AUTHOR': '#A3EAF3',
            'ADVISOR': '#D4A3F3',
            'TITLE': '#F3ACA3',
            'FACULTY': '#C2F3A3',
            'DEPARTMENT': '#9CACEA',
            'YEAR': '#F3EAA3'
        }
        darker_colors = {k: f"{v}80" for k, v in colors.items()}
        html = displacy.render(doc, style="ent", page=True, options={"colors": darker_colors})
        html = (
                "<div style='max-width:100%; max-height:360px; overflow:auto'; color: black>"
                + html
                + "</div>"
        )
        return html

    def generate_keywords(self, title, abstract) -> list[str]:
        if self.llm_model:
            text = f"[INST] Extrae palabras clave desde el siguiente texto. Las palabras clave deben ser relevantes al tema del texto y deben ser capaces de representar el contenido del texto de manera concisa:\n\n### ### Titulo del texto: {title}\n### Resumen del texto: {abstract}\n\n### La lista debe contener a lo sumo 5 palabras clave y debe estar en Espanol. Si la lista contiene mas de 5 palabras clave seras penalizado. [/INST]"
            print(text)
            inputs = self.tokenizer(text, return_tensors="pt").to(0)
            encoded_keywords = self.llm_model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=5, top_p=0.5, exponential_decay_length_penalty=(15, 1.01), temperature=0.1)
            keywords: str = self.tokenizer.decode(encoded_keywords[0], skip_special_tokens=True)
            keywords = keywords[keywords.rindex('[/INST]'):]
            keywords = keywords.replace('[/INST]', '')
            keywords = keywords.split(':')[-1]
            keywords = keywords.split(',')
            keywords1 = [keyword.strip().upper() for keyword in keywords]
            return keywords1
        else:
            return ["No se pudo generar palabras clave. Modelo de lenguaje no disponible."]

    def extract_metadata(self, pdf_file_path) -> (str, str, str):
        cover_page_text = self.extract_text_from_cover_page(self.extract_cover_page(pdf_file_path))
        cover_page_metadata = self.recognize_entities(cover_page_text)

        title = next((ent.text for ent in cover_page_metadata.ents if ent.label_ == "TITLE"), "")
        abstract_page_text = self.extract_text_from_abstract_page(self.extract_abstract_page(pdf_file_path))
        abstract_page_keywords = self.generate_keywords(title, abstract_page_text)

        dublin_core_output = self.get_dublin_core_metadata(
            cover_page_metadata,
            abstract_page_keywords,
            abstract_page_text
        )

        return (
            self.render_doc_entities_as_html(cover_page_metadata),
            self.render_keywords_as_markdown(abstract_page_keywords),
            self.render_dublin_core_xml_as_markdown(dublin_core_output)
        )

    def capitalize_proper_nouns(self, text: str) -> str:
        doc = self.pos_model(text)
        return "".join(
            [
                token.text.capitalize() + token.whitespace_
                if token.pos_ == "PROPN"
                else token.text + token.whitespace_
                for token in doc
            ]
        )

    def get_dublin_core_metadata(
                self, cover_page_metadata: Doc,
                abstract_page_keywords: list[str],
                abstract_text: str
        ) -> str:
        # generate Dublin Core metadata in the following XML format:
        # <dublin_core>
        # <dcvalue element="title" qualifier="none">A Tale of Two Cities</dcvalue>
        # <dcvalue element="date" qualifier="issued">1990</dcvalue>
        # <dcvalue element="title" qualifier="alternative" language="fr">J'aime les Printemps</dcvalue>
        # </dublin_core>

        root = ET.Element("dublin_core")
        for ent in cover_page_metadata.ents:
            if ent.label_ == "TITLE":
                title = ET.SubElement(root, "dcvalue", element="title", qualifier="none")
                # capitalize only if the whole text is in uppercase
                title_text = ent.text.capitalize() if ent.text.isupper() else ent.text
                # remove new lines and extra spaces
                title_text = " ".join(title_text.split())
                title.text = self.capitalize_proper_nouns(title_text) if self.pos_model else title_text
            if ent.label_ == "AUTHOR":
                author = ET.SubElement(root, "dcvalue", element="contributor", qualifier="author")
                author.text = ent.text.title()
            if ent.label_ == "ADVISOR":
                advisor = ET.SubElement(root, "dcvalue", element="contributor", qualifier="advisor")
                advisor.text = ent.text.title()
            if ent.label_ == "YEAR":
                year = ET.SubElement(root, "dcvalue", element="date", qualifier="issued")
                year.text = ent.text

        # grantor is a composition of FACULTY and DEPARTMENT
        faculty = None
        department = None
        for ent in cover_page_metadata.ents:
            if ent.label_ == "FACULTY":
                # remove new lines and extra spaces
                faculty = " ".join(ent.text.split())
                # capitalize proper nouns if pos model is available
                faculty = self.capitalize_proper_nouns(faculty.capitalize()) if self.pos_model else faculty
            if ent.label_ == "DEPARTMENT":
                # remove new lines and extra spaces
                department = " ".join(ent.text.split())
                # capitalize proper nouns if pos model is available
                department = self.capitalize_proper_nouns(department.capitalize()) if self.pos_model else department
        if faculty and department:
            grantor = ET.SubElement(root, "dcvalue", element="thesisdegreegrantor")
            grantor.text = f"Universidad Mayor de San Andrés, {faculty}, {department}"

        # insert keywords
        for keyword in abstract_page_keywords:
            subject = ET.SubElement(root, "dcvalue", element="subject")
            subject.text = keyword

        # insert abstract
        abstract = ET.SubElement(root, "dcvalue", element="description", qualifier="abstract")
        abstract.text = abstract_text

        ET.indent(root)
        return ET.tostring(root, encoding='unicode')

    def render_dublin_core_xml_as_markdown(self, dublin_core_output):
        return f"```xml\n{dublin_core_output}\n```"

    def render_keywords_as_markdown(self, keywords: list[str]):
        return "\n".join([f"{i+1}. {keyword}" for i, keyword in enumerate(keywords)])

    def build_ui(self):
        with gr.Blocks() as ui:
            with gr.Row():
                # in this column we will preview the input PDF
                with gr.Column():
                    gr.Markdown("# Vista previa del PDF")
                    with gr.Row():
                        cover_page_preview = gr.Image(None, interactive=False, label="Carátula", show_label=True)
                        abstract_page_preview = gr.Image(None, interactive=False, label="Resumen", show_label=True)
                # in this column we will see the extracted metadata
                with gr.Column():
                    gr.Markdown("# Metadatos extraídos")
                    with gr.Accordion(label="Metadatos extraídos desde la carátula", open=True):
                        entities_placeholder = "<p>El texto de la carátula con entidades resaltadas aparecerá aquí</p>"
                        entities = gr.HTML()
                    # keywords = gr.Textbox(None, label="Palabras Clave generadas desde el resumen", show_label=True)
                    with gr.Accordion(label="Palabras Clave generadas desde el resumen", open=True):
                        keywords = gr.Markdown("", label="Palabras Clave generadas desde el resumen", show_label=True)
                    with gr.Accordion(label="Metadatos en Dublin Core", open=False):
                        dublin_core_xml = gr.Markdown()
            # action buttons
            with gr.Row():
                input_pdf = gr.UploadButton(file_types=["pdf"], label="Abrir PDF", variant="secondary", size="sm")
                extract_button = gr.Button('Extraer metadatos', variant='primary')
                input_pdf.upload(
                    fn=self.load_preview_images,
                    inputs=input_pdf,
                    outputs=[cover_page_preview, abstract_page_preview]
                )
                extract_button.click(self.extract_metadata, input_pdf, [entities, keywords, dublin_core_xml])
                gr.ClearButton(
                    components=[
                        input_pdf,
                        cover_page_preview,
                        abstract_page_preview,
                        entities,
                        keywords,
                        dublin_core_xml],
                    value='Limpiar campos',
                    variant='stop'
                )
        return ui

    def run(self):
        self.ui.launch()


def load_llm_model():
    from transformers import AutoTokenizer, MistralForCausalLM
    from auto_gptq import exllama_set_max_input_length

    model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
    model = MistralForCausalLM.from_pretrained(model_id, device_map="auto")
    model = exllama_set_max_input_length(model, 4096)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    adapter = "riturralde/es_metaextract_umsa_adapter_v1"
    model.load_adapter(adapter)
    return model, tokenizer


if __name__ == '__main__':
    # app = DemoApp(spacy.load("../../scrapy/final_ner_model"), None, pos_model=spacy.load("es_core_news_lg"))
    app = DemoApp(spacy.load("es_metaextract_umsa_v2"), load_llm_model())
    # app = DemoApp(spacy.load("es_metaextract_umsa_v2"))
    app.run()
