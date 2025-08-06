import json

# Read the original file
with open("test.json2", "r", encoding="utf-8") as f:
    lines = f.readlines()

stripped_lines = []
for line in lines:
    # Parse the JSON object
    obj = json.loads(line)

    # For each message sequence
    for msg in obj["messages"]:
        # If the message is from the assistant and contains a label, clear it
        if msg.get("role") == "assistant" and msg.get("content") in ("center", "left", "right"):
            msg["content"] = ""  # or you can set to None if you prefer

    # Serialize back to JSON
    stripped_lines.append(json.dumps(obj, ensure_ascii=False))

# Write to a new file
with open("test_no_labels.json2", "w", encoding="utf-8") as f:
    for line in stripped_lines:
        f.write(line + "\n")