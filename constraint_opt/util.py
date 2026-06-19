def args2str(x):
    return "_".join(list(map(str, x)))

def unifies(x, args_p, args_a) -> bool:
    if len(x) != len(args_p):
        return False
    for i in range(0, len(x)):
        if args_a[x[i] - 1] != args_p[i]:
            return False
    return True