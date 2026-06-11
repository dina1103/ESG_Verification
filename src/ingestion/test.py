import json
def norm(s): return " ".join(str(s).split()).strip().lower()
d = json.load(open(r"src\ingestion\ml_promise_dataset\English_test.json", encoding="utf-8"))
train = json.load(open(r"src\ingestion\ml_promise_dataset\PromiseEval_Trainset_English.json", encoding="utf-8"))
tt = set(norm(r["data"]) for r in train)
print(len(d), "records |", sum(1 for r in d if norm(r["data"]) in tt), "train-overlap")
# want: 388 records | 0 train-overlap