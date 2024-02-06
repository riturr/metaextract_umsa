from typing import Any

import scrapy
from scrapy.http import Response, Request

BASE_URL = 'https://repositorio.umsa.bo'
DSPACE_COMMUNITY_PAGES_TO_CRAWL = {
    'Área Vicerrectorado': f'{BASE_URL}/handle/123456789/17064/recent-submissions',
    'Facultad de Agronomía': f'{BASE_URL}/handle/123456789/3797/recent-submissions',
    'Facultad de Arquitectura, Artes, Diseño y Urbanismo': f'{BASE_URL}/handle/123456789/308/recent-submissions',
    'Facultad de Ciencias Económicas y Financieras': f'{BASE_URL}/handle/123456789/309/recent-submissions',
    'Facultad de Ciencias Farmacéuticas y Bioquímicas': f'{BASE_URL}/handle/123456789/5/recent-submissions',
    'Facultad de Ciencias Geológicas': f'{BASE_URL}/handle/123456789/19871/recent-submissions',
    'Facultad de Ciencias Puras y Naturales': f'{BASE_URL}/handle/123456789/313/recent-submissions',
    'Facultad de Ciencias Sociales': f'{BASE_URL}/handle/123456789/322/recent-submissions',
    'Facultad de Derecho y Ciencias Políticas': f'{BASE_URL}/handle/123456789/336/recent-submissions',
    'Facultad de Humanidades y Ciencias de la Educación': f'{BASE_URL}/handle/123456789/327/recent-submissions',
    'Facultad de Ingenieria': f'{BASE_URL}/handle/123456789/20341/recent-submissions',
    'Facultad de Medicina, Enfermería, Nutrición y Tecnología Médica': f'{BASE_URL}/handle/123456789/354/recent-submissions',
    'Facultad de Odontología': f'{BASE_URL}/handle/123456789/1687/recent-submissions',
    'Facultad de Tecnología': f'{BASE_URL}/handle/123456789/362/recent-submissions'
}


class DSpaceCommunitySpider(scrapy.Spider):
    name = 'dspace_spider'
    base_url = 'https://repositorio.umsa.bo'
    start_urls = [url for faculty, url in DSPACE_COMMUNITY_PAGES_TO_CRAWL.items()]

    custom_settings = {
        'ITEM_PIPELINES': {'scrapy.pipelines.files.FilesPipeline': 1},
        'FILES_STORE': './files'
    }

    def parse(self, response: Response, **kwargs: Any) -> Any:
        for title in response.css('h4.artifact-title'):
            path = title.xpath('a/@href').get()
            yield Request(
                url=self.base_url + path,
                callback=self.parse_document_page,
                cb_kwargs={'community_page': response.url}
            )

        for next_page in response.css('a.next-page-link'):
            yield response.follow(next_page, self.parse)

    def parse_document_page(self, response: Response, **kwargs: Any) -> Any:
        document_url = BASE_URL + (response
                                   .css('#aspect_artifactbrowser_ItemViewer_div_item-view')
                                   .css('.item-page-field-wrapper.table.word-break')
                                   .xpath('.//a/@href').get())

        metadata_file_url = BASE_URL + (
            response.css('#aspect_artifactbrowser_ItemViewer_div_item-view')
            .xpath('.//comment()')
            .re(r'/metadata/handle/\d+/\d+/.+\.xml')[0]
        )
        # Get breadcrumb path, so we know to which section in DSpace this document belongs to
        breadcrumb = list(filter(
            lambda item: item.strip() != '',
            response.css('ul.breadcrumb').xpath('.//text()').getall()
        ))
        yield {
            'community_page': response.cb_kwargs['community_page'],
            'document_page': response.url,
            'breadcrumb': breadcrumb,
            'file_urls': [document_url, metadata_file_url]
        }
