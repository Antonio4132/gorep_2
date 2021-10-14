def intersection(list1, list2):
    if not list1:
        return list2
    if not list2:
        return list1

    return list(set(list1) & set(list2))