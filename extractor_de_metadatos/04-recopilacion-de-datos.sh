#!/bin/bash

scrapy runspider recopilar-datos.py \
    -o all_collected_data.jsonl \
    --logfile all_collect_data.log \
    --set JOBDIR=crawls/community-spider-4 \
    --set CONCURRENT_REQUESTS_PER_DOMAIN=2 \
    --set LOG_LEVEL=INFO \
    --set DOWNLOAD_DELAY=0.5
