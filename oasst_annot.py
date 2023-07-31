# ds = load_dataset("OpenAssistant/oasst1")
# load dataset from local jsonl
path = "oasst/2023-04-12_oasst_all.messages.jsonl"

import pandas as pd

df = pd.read_json(path, lines=True)

sdf = df[df["synthetic"]]
oasst_df = sdf[sdf["role"] == "assistant"]

for row in oasst_df.iloc:
    pid = row["parent_id"]
    texts = [row["role"].upper() + ": " + row["text"]]
    while not pd.isnull(pid):
        parent = df[df["message_id"] == pid].iloc[0]
        texts.append(parent["role"].upper() + ": " + parent["text"])
        pid = parent["parent_id"]

    texts.reverse()
    transcript = "\n\n".join(texts)
    print(transcript)
    if input() == "q":
        break
    # clear console
    print("\033[H\033[J")
