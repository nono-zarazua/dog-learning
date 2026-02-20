from icrawler.builtin import BingImageCrawler
import os

# Pfad erstellen, falls er nicht existiert

output_dir = '/home/jona/Schreibtisch/Studium/Semester3/Bioimaging/Practical_week/Data/dataset_puppy_donuts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste der "Donut-Hund" Suchbegriffe
search_queries = [
    'curled up puppy sleeping top view',
    'dog sleeping in a basket',
    'pug curled up sleeping',
    'french bulldog sleeping ball',
    'sleeping puppy round shape'
]
# search_queries_bagels = [
#     'single bagel top view',
#     'sesame bagel isolated',
#     'bagel on wooden plate top view',
#     'freshly baked bagels',
#     'plain bagel isolated'
# ]

for query in search_queries:
    print(f"Lade Bilder für: {query}...")
    
    # Der Google-Crawler
    # 'max_num' kannst du erhöhen, wenn du mehr Daten brauchst
    google_crawler = BingImageCrawler(storage={'root_dir': output_dir})
    
    google_crawler.crawl(
        keyword=query, 
        max_num=20, 
        min_size=(400, 400), # Filtert zu kleine Vorschaubilder aus
        file_idx_offset='auto'
    )

print(f"Fertig! Deine Bilder liegen im Ordner: {output_dir}")