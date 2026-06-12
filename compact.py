LAYER1_LEN = 30

def compact_layer1(messages: list):
    if len(messages) < LAYER1_LEN:
        return messages
    for msg in messages[2:-5]:
        if msg["role"] == "tool":
            msg["content"] = f"[Previous: used {msg["name"]}"



