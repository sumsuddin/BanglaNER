# -*- coding: utf-8 -*-
import click
import logging
import wget
import gdown
from utils import delete_files_and_directories


def download(url, output_path):
    logger = logging.getLogger(__name__)

    logger.info('downloading the raw data from {url}'.format(url=url))
    filename = wget.download(url, out=output_path)
    logger.info('saved at {filename}'.format(filename=filename))


@click.command()
@click.argument('output_dir', type=click.Path())
def main(output_dir):
    """ Downloads and syncs the raw dataset to the given
        `output_dir` directory
    """

    delete_files_and_directories(output_dir)

    url = 'https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl'
    download(url, output_dir)

    for i in range(1, 21):
        url = 'https://raw.githubusercontent.com/Rifat1493/Bengali-NER/master/annotated data/{}.txt'.format(i)
        download(url, output_dir)

    print("Done")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
