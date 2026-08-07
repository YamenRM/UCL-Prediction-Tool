import re
import json
import pandas as pd
from Configuration import Review_PATH , json_mapping_path, Need_Review_PATH

review_df = pd.read_csv(Review_PATH)

# Check if the shorter name is a real match within the longer name, considering word boundaries and ignoring case. This helps avoid false positives where one name is a substring of another but not actually the same team.

def is_real_match(short: str, long_: str) -> bool:
    pattern = re.escape(short.lower())
    return re.search(rf'(^|\W){pattern}(\W|$)', long_.lower()) is not None

auto_accept = {}
needs_review = []

for _, row in review_df.iterrows():
    our_name = row['our_name']
    aliases = set()

    if row['elo_score'] == 100 and row['elo_guess'] != our_name:
        if is_real_match(our_name, row['elo_guess']) or is_real_match(row['elo_guess'], our_name):
            aliases.add(row['elo_guess'])

    if row['sofifa_score'] == 100 and row['sofifa_guess'] != our_name:
        if is_real_match(our_name, row['sofifa_guess']) or is_real_match(row['sofifa_guess'], our_name):
            aliases.add(row['sofifa_guess'])

    if aliases:
        auto_accept[our_name] = sorted(aliases)
    elif row['elo_score'] < 100 or row['sofifa_score'] < 100:
        needs_review.append(row.to_dict())

with open(json_mapping_path, "w", encoding="utf-8") as f:
    json.dump(auto_accept, f, indent=2, ensure_ascii=False)

pd.DataFrame(needs_review).to_csv(Need_Review_PATH, index=False)
print(f"Auto-accepted {len(auto_accept)} clean mappings")
print(f"{len(needs_review)} still genuinely need a human look")