# Ensures complex objects (e.g., Document, LLM responses) are converted to plain dicts and strings so JSON can be written.
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v) for k, v in obj.items()
        }  # Convert keys and values to strings in a dictionary recursively
    elif isinstance(obj, list):
        return [
            make_json_safe(i) for i in obj
        ]  # Convert values to strings in a list recursively
    elif hasattr(obj, "page_content"):
        return {
            "page_content": obj.page_content,
            "metadata": obj.metadata,
        }  # Convert Document data to a dictionary
    elif hasattr(obj, "__dict__"):
        return str(
            obj
        )  # If it’s any other object with a __dict__ convert the entire object to its string representation
    else:
        return obj  # Strings, numbers, booleans, and None are already serializable, so they pass through unchanged.
