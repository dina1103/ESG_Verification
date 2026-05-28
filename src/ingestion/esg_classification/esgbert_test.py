from transformers import pipeline

e_clf = pipeline("text-classification", model="ESGBERT/EnvironmentalBERT-environmental", truncation=True, max_length=512)
s_clf = pipeline("text-classification", model="ESGBERT/SocialBERT-social", truncation=True, max_length=512)
g_clf = pipeline("text-classification", model="ESGBERT/GovernanceBERT-governance", truncation=True, max_length=512)

# test sentences — we know what these should be
sentences = [
    "We reduced Scope 1 emissions by 20% in 2023.",         # clearly Environmental
    "We provide health insurance to all employees.",          # clearly Social
    "The board consists of 40% independent directors.",      # clearly Governance
    "The loan agreement was signed on January 1st.",          # clearly None
]


print("--- EnvironmentalBERT ---")
for sent, res in zip(sentences, e_clf(sentences)):
    print(f"  {res}  →  {sent[:60]}")

print("\n--- SocialBERT ---")
for sent, res in zip(sentences, s_clf(sentences)):
    print(f"  {res}  →  {sent[:60]}")

print("\n--- GovernanceBERT ---")
for sent, res in zip(sentences, g_clf(sentences)):
    print(f"  {res}  →  {sent[:60]}")