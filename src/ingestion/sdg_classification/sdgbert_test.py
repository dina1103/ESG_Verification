from transformers import pipeline

print("Loading sdgBERT...")
clf = pipeline("text-classification", model="sadickam/sdgBERT", truncation=True, max_length=512)
print("Loaded\n")


test_sentences = {
    "SDG12": [
        "We reduced packaging waste by 30% through circular economy initiatives.",
        "Our products are designed for reuse, repair, and recycling at end of life.",
        "We are committed to responsible consumption and sustainable supply chains.",
        "The company adopted life-cycle assessment methods across its product lines.",
        "We achieved a 65% recycled content rate in our aluminum components.",
        "Sustainable procurement policies require suppliers to meet environmental standards.",
        "Food waste in our canteens was reduced by 40% compared to the 2020 baseline.",
        "We published our first report in line with the GRI 306 waste standard.",
        "Our zero-landfill program now covers 80% of our manufacturing sites.",
        "Responsible sourcing of critical raw materials is a core pillar of our strategy.",
    ],
    "SDG13": [
        "We reduced Scope 1 emissions by 20% compared to our 2020 baseline.",
        "Our climate strategy targets net zero carbon emissions by 2050.",
        "We have committed to science-based targets aligned with the Paris Agreement.",
        "Our TCFD disclosure identifies physical and transition climate risks.",
        "Greenhouse gas emissions from our operations decreased year over year.",
        "We purchased renewable energy certificates covering 100% of our electricity.",
        "Climate change adaptation measures were integrated into our risk management framework.",
        "Our SBTi-validated target commits us to a 1.5 degree pathway.",
        "We measure and report our carbon footprint using the GHG Protocol.",
        "The company invested in decarbonization technologies for its manufacturing processes.",
    ],
    "SDG16": [
        "We have a zero tolerance policy for corruption and bribery.",
        "Our whistleblower policy protects employees who report misconduct.",
        "The audit committee ensures transparency and accountability in governance.",
        "Anti-corruption training was completed by 98% of employees this year.",
        "Our Code of Conduct aligns with the UN Global Compact principles.",
        "Compliance with the OECD Anti-Bribery Convention is monitored by internal audit.",
        "We published our first transparency report on political contributions and lobbying activities.",
        "The board of directors maintains oversight of our ethics and compliance program.",
        "Our due diligence procedures include human rights and anti-bribery risk assessment.",
        "We do not make political donations or contributions to political parties.",
    ],
    # Confounders — these might confuse the model because they use overlapping language
    "Confounders": [
        # Tech/innovation sounding — might get pulled to SDG9 (Industry, Innovation, Infrastructure)
        "We developed innovative carbon capture technology for our cement plants.",
        "Our climate action plan includes investment in infrastructure upgrades.",
        # Governance-of-climate — could be SDG13 or SDG16
        "The board's climate committee oversees our decarbonization strategy.",
        # Social-adjacent — could be SDG8 (decent work) or SDG12 (supply chain)
        "Supplier audits verify compliance with labor and environmental standards.",
        # Actually non-ESG, stress-test
        "Revenue from our financial services division grew by 12% last year.",
    ],
}

total_correct = 0
total_tested = 0

for expected, sentences in test_sentences.items():
    print(f"=== Expected: {expected} ===")
    results = clf(sentences)
    correct_this_group = 0
    for sent, res in zip(sentences, results):
        got = res["label"].lower()
        score = res["score"]

        # simple correctness check — only for the three in-scope categories
        if expected != "Confounders":
            expected_label = expected.lower()
            is_correct = got == expected_label
            marker = "✓" if is_correct else "✗"
            if is_correct:
                correct_this_group += 1
                total_correct += 1
            total_tested += 1
        else:
            marker = "?"

        print(f"  {marker} {got:<8} ({score:.3f})  →  {sent[:90]}")

    if expected != "Confounders":
        print(f"  Correct: {correct_this_group}/{len(sentences)}")
    print()

print(f"=== OVERALL (excluding confounders): {total_correct}/{total_tested} correct ({total_correct/total_tested*100:.1f}%) ===")