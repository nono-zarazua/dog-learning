#!/usr/bin/env python3
import argparse

from dog_crawler import run_crawler
from manual_image_filter import run_manual
from feature_extraction import run_extraction

DIR_HELP = "Path to the target directory containing your images."

def handle_crawl(args):
    run_crawler(
        output_dir=args.output_dir,
        search_queries=args.queries,
        max_num=args.max_num
    )

def handle_manual(args):
    run_manual(
        source_dir=args.source_dir
    )

def handle_extract(args):
    run_extraction(
        source_dir=args.source_dir,
        output_csv=args.output_csv
    )
def main():
    parser = argparse.ArgumentParser(
        description = "Scrape images from Google fro your bioimaging dataset."
    )

    subparsers = parser.add_subparsers(
        dest='command',
        required=True,
        help="Pipeline allows to either download fresh dataset, manual filter, or feature extraction."
    )

    # ==========================================
    # SUBCOMMAND: crawl
    # ==========================================
    parser_crawl = subparsers.add_parser('crawl', help="Scrape image data from the web.")
    parser_crawl.add_argument('output_dir', type=str, help=DIR_HELP)
    parser_crawl.add_argument('--queries', type=str, nargs='+', help="Space-separated list of search queries enclosed in quotes (e.g., 'sleeping puppy', 'plain bagel')")
    parser_crawl.add_argument('--max-num', type=int, default=20, help="Maximum number of images to download per query (default: 20).")
    parser_crawl.set_defaults(func=handle_crawl)

    # ==========================================
    # SUBCOMMAND: manual filter
    # ==========================================

    parser_manual = subparsers.add_parser('manual', help="Open manual filter for dataset.")
    parser_manual.add_argument('source_dir', type=str, help=DIR_HELP)
    parser_manual.set_defaults(func=handle_manual)

    # ==========================================
    # SUBCOMMAND: extract
    # ==========================================
    parser_extract = subparsers.add_parser('extract', help="Extract features from images.")
    parser_extract.add_argument('source_dir', type=str, help=DIR_HELP)
    parser_extract.add_argument('output_csv', type=str, help="Path to output.")
    parser_extract.set_defaults(func=handle_extract)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()


