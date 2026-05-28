import json
from pathlib import Path

JSON_DIR = r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\json"

json_paths = list(Path(JSON_DIR).rglob("*.json"))
print(f"Found {len(json_paths)} JSON files\n")

# check metadata across all files
print("--- Metadata check ---")
for json_path in json_paths:
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    print(
        f"{doc['filename'][:60]:<60} | "
        f"company: {doc['company_name']:<25} | "
        f"year: {doc['year']} | "
        f"pages: {doc['pages_extracted']}/{doc['total_pages']} | "
        f"ocr: {doc['pages_ocr']}"
    )

# inspect one document in detail
print("\n--- Detailed check (first file) ---")
with open(json_paths[0], encoding="utf-8") as f:
    doc = json.load(f)

print("Company:", doc["company_name"])
print("Year:", doc["year"])
print("Report type:", doc["report_type"])

# check page 5
page = doc["pages"][5]
print(f"\nPage {page['page_number']} preview:")
print(page["text"][:300])
print("\nTables on this page:")
for row in page["tables"][:5]:
    print(" -", row)

# check how many pages have structured tables vs how many have table-like text
import json, glob
json_files = glob.glob(r"C:\Users\dina_\Desktop\esg_verification_draft\data\processed\json\**\*.json", recursive=True)

total_pages = 0
pages_with_tables = 0
table_rows_total = 0
for path in json_files:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    for page in doc["pages"]:
        total_pages += 1
        if page["tables"]:
            pages_with_tables += 1
            table_rows_total += len(page["tables"])

print(f"Total pages extracted: {total_pages}")
print(f"Pages with structured tables: {pages_with_tables} ({pages_with_tables/total_pages*100:.1f}%)")
print(f"Total table rows captured: {table_rows_total}")