from itertools import permutations, product
from typing import List, Dict, Tuple

import torch

from planning_structs.domain import Domain, ActionSchema
from planning_structs.instance import Instance
from planning_structs.traces import ObservationM


def sig_match(domain: Domain, name_list1: List[str], name_list2: List[str]) -> bool:
    for n1, n2 in zip(name_list1, name_list2):
        types1 = [t.name for t in domain.get_action_schema(n1).types]
        types2 = [t.name for t in domain.get_action_schema(n2).types]
        if sorted(types1) != sorted(types2):
            return False
    return True


def type_match(x: List[int], a: ActionSchema) -> bool:
    for i in range(0, len(x)):
        if a.types[x[i] - 1] != a.types[i]:
            return False
    return True


def perm_agree(domain: Domain,
               name_2_to_1: Dict[str, str],
               args_2_to_1: Dict[str, Tuple[int]],
               model1: ObservationM,
               model2: ObservationM) -> float:
    dist = 0
    item_num = 0
    for a_2, p, x_2 in model2.pre:
        proba_pre = model2.pre[a_2, p, x_2]
        proba_add = model2.add[a_2, p, x_2]
        proba_del = model2.dele[a_2, p, x_2]
        a_1 = domain.get_action_schema(name_2_to_1[a_2.name])
        x_1 = tuple(args_2_to_1[a_2.name][i - 1] for i in x_2)  # x_ext is 1-based index
        dist += proba_pre * model1.pre[(a_1, p, x_1)] + (1 - proba_pre) * (1 - model1.pre[(a_1, p, x_1)]) + \
                proba_add * model1.add[(a_1, p, x_1)] + (1 - proba_add) * (1 - model1.add[(a_1, p, x_1)]) + \
                proba_del * model1.dele[(a_1, p, x_1)] + (1 - proba_del) * (1 - model1.dele[(a_1, p, x_1)])
        item_num += 3
    return dist / item_num


def perm_error(domain: Domain,
               name_2_to_1: Dict[str, str],
               args_2_to_1: Dict[str, Tuple[int]],
               model1: ObservationM,
               model2: ObservationM) -> int:
    _error = 0
    for a_2, p, x_2 in model2.pre:
        proba_pre = model2.pre[a_2, p, x_2]
        proba_add = model2.add[a_2, p, x_2]
        proba_del = model2.dele[a_2, p, x_2]
        case_1 = torch.tensor([1 - proba_pre - proba_add, proba_add, proba_pre - proba_del, proba_del]).argmax()

        a_1 = domain.get_action_schema(name_2_to_1[a_2.name])
        x_1 = tuple(args_2_to_1[a_2.name][i - 1] for i in x_2)
        proba_pre = model1.pre[(a_1, p, x_1)]
        proba_add = model1.add[(a_1, p, x_1)]
        proba_del = model1.dele[(a_1, p, x_1)]
        case_2 = torch.tensor([1 - proba_pre - proba_add, proba_add, proba_pre - proba_del, proba_del]).argmax()

        if case_1 != case_2:
            _error += 1
    return _error


def model_permutation(domain: Domain, model1: ObservationM, model2: ObservationM) \
        -> Tuple[Dict[str, str], Dict[str, Tuple[int]], float, int]:
    names = [a.name for a in domain.action_schemas]
    names_perms = [n for n in list(permutations(names)) if sig_match(domain, names, n)]
    args_perms = []
    for name in names:
        a = domain.get_action_schema(name)
        args_perms.append([x for x in list(permutations(list(range(1, a.arity + 1)))) if type_match(x, a)])

    best_name_mapping = None
    best_args_mapping = None
    highest_agree = -float("inf")
    lowest_error = float("inf")
    for names_perm in names_perms:
        for args_perm in product(*args_perms):
            name_mapping = {n1: n2 for n1, n2 in zip(names, names_perm)}
            args_mapping = {name: args for name, args in zip(names, args_perm)}
            agree = perm_agree(domain, name_mapping, args_mapping, model1, model2)
            error = perm_error(domain, name_mapping, args_mapping, model1, model2)
            if error < lowest_error:
                lowest_error = error
            if agree > highest_agree:
                best_name_mapping = name_mapping
                best_args_mapping = args_mapping
                highest_agree = agree

    return best_name_mapping, best_args_mapping, highest_agree, lowest_error