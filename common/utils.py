



def print_quality_checks(cfgs: dict) -> None:
    """Print enabled grid quality check flags."""
    if cfgs.get('check_orth') == 1:
        print("------- check grid orthogonality-------")
    if cfgs.get('check_jac') == 1:
        print("------- check grid jacobi-------")
    if cfgs.get('check_ratio') == 1:
        print("------- check grid ratio-------")
    if cfgs.get('check_step_xi') == 1:
        print("------- check grid step xi direction-------")
    if cfgs.get('check_step_zt') == 1:
        print("------- check grid step zt direction-------")
    if cfgs.get('check_smooth_xi') == 1:
        print("------- check grid smooth xi direction-------")
    if cfgs.get('check_smooth_zt') == 1:
        print("------- check grid smooth zt direction-------")


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