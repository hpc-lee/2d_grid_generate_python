


def remove_comment_keys(data):
    """Filter out keys starting with '#' from
       nested dictionaries (JSON-loaded data)."""
    if isinstance(data, dict):
        new_dict = data.copy()
        for key in list(new_dict.keys()):
            if key.startswith('#'):
                del new_dict[key]
            else:
                new_dict[key] = remove_comment_keys(new_dict[key])
        return new_dict
    elif isinstance(data, list):
        return [remove_comment_keys(item) for item in data]
    else:
        return data