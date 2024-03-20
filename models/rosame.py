import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import math
import string
import json
from json import JSONEncoder


class Type:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def is_child(self, another_type):
        if self.name == another_type.name:
            return True
        elif self.parent is None:
            return False
        else:
            return self.parent.is_child(another_type)


class Predicate:
    def __init__(self, name, params):
        self.name = name
        # params are dicts {Type: num}
        self.params = params
        self.params_types = sorted(params.keys(), key=lambda x: x.name)

    def proposition(self, sorted_obj_lists):
        return (
            self.name
            + " "
            + " ".join(
                [
                    f"{self.params_types[i].name} "
                    + f" {self.params_types[i].name} ".join(sorted_obj_lists[i])
                    for i in range(len(sorted_obj_lists))
                ]
            )
        ).strip()

    def ground(self, objects):
        """
        Input a list of objects in the form {Type: []}
        Return all the propositions grounded from this predicates with the objects
        """
        propositions = []
        obj_lists_per_params = {params_type: [] for params_type in self.params_types}
        for params_type in self.params_types:
            for obj_type in objects.keys():
                if obj_type.is_child(params_type):
                    obj_lists_per_params[params_type].extend(objects[obj_type])
        for obj_lists in itertools.product(
            *[
                itertools.permutations(
                    obj_lists_per_params[params_type], self.params[params_type]
                )
                for params_type in self.params_types
            ]
        ):
            propositions.append(self.proposition(obj_lists))
        return propositions


class Action_Schema(nn.Module):
    def __init__(self, name, params):
        super(Action_Schema, self).__init__()
        self.name = name
        # params are dicts {Type: num}
        self.params = params
        self.params_types = sorted(params.keys(), key=lambda x: x.name)
        # predicates that are relevant
        self.predicates = []

    def initialise(self, predicates, device):
        """
        Input all predicates and generate the deep learning model for the action schema
        """
        n_features = 0
        for predicate in predicates:
            # A predicate is relevant to an action schema iff for each of its param type,
            # the number of objects required is leq the number of objects there is
            # for the same type or children type in the action schema
            is_relevant = True
            # Also calculate how many propositions there are when predicate is grounded on "variables"
            # e.g. on X Y; on Y X when X and Y are variables
            n_ground = 1
            for params_type in predicate.params_types:
                n_params = 0
                for model_params_type in self.params:
                    if model_params_type.is_child(params_type):
                        n_params += self.params[model_params_type]
                if predicate.params[params_type] > n_params:
                    is_relevant = False
                    break
                else:
                    n_ground *= math.perm(n_params, predicate.params[params_type])
            if is_relevant:
                self.predicates.append(predicate)
                n_features += n_ground
        n_features = int(n_features)

        self.randn = torch.randn(n_features, 128, device=device, requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=1),
        )
        self.mlp.to(device)

    def forward(self):
        return self.mlp(self.randn)

    def ground(self, objects, is_single_action=False):
        if is_single_action:
            propositions = []
            for predicate in self.predicates:
                propositions.extend(predicate.ground(objects))
            return propositions
        else:
            propositions = []
            obj_lists_per_params = {
                params_type: [] for params_type in self.params_types
            }
            for params_type in self.params_types:
                for obj_type in objects.keys():
                    if obj_type.is_child(params_type):
                        obj_lists_per_params[params_type].extend(objects[obj_type])
            for obj_list in itertools.product(
                *[
                    itertools.permutations(
                        obj_lists_per_params[params_type], self.params[params_type]
                    )
                    for params_type in self.params_types
                ]
            ):
                objects_per_action = {}
                for i in range(len(self.params_types)):
                    objects_per_action[self.params_types[i]] = obj_list[i]
                propositions_per_action = []
                for predicate in self.predicates:
                    propositions_per_action.extend(predicate.ground(objects_per_action))
                propositions.append(propositions_per_action)
            return propositions

    def pretty_print(self):
        var = {}
        n = 0
        for param_type in self.params_types:
            var[param_type] = list(string.ascii_lowercase)[
                n : n + self.params[param_type]
            ]
            n += self.params[param_type]
        print(
            f"{self.name}"
            + " "
            + " ".join([k.name + " " + v for k in var.keys() for v in var[k]])
        )
        propositions = self.ground(var, True)
        precon_list = []
        addeff_list = []
        deleff_list = []
        result = torch.argmax(self(), dim=1)
        for i in range(len(propositions)):
            if result[i] == 1:
                addeff_list.append(propositions[i])
            elif result[i] == 2:
                precon_list.append(propositions[i])
            elif result[i] == 3:
                precon_list.append(propositions[i])
                deleff_list.append(propositions[i])
        print(", ".join(precon_list))
        print(", ".join(addeff_list))
        print(", ".join(deleff_list))


class Domain_Model(nn.Module):
    def __init__(self, types, predicates, action_schemas, device):
        super(Domain_Model, self).__init__()
        self.types = types
        self.predicates = predicates
        self.action_schemas = action_schemas
        self.device = device
        for action_schema in action_schemas:
            action_schema.initialise(predicates, self.device)

    def ground(self, objects):
        # Ground predicates to propositions
        # Record in a dictionary with values as indices, for later lookup
        self.propositions = {}
        for predicate in self.predicates:
            for proposition in predicate.ground(objects):
                self.propositions[proposition] = len(self.propositions)

        # For each action schema, ground to actions and then find the indices
        self.indices = []
        # Also need to know which action schema each action is from
        self.action_to_schema = []
        for action_schema in self.action_schemas:
            for propositions in action_schema.ground(objects):
                self.indices.append([self.propositions[p] for p in propositions])
                self.action_to_schema.append(action_schema)

    def build(self, actions):
        """
        actions is a list of numbers
        """
        precon = torch.zeros(
            (len(actions), len(self.propositions)),
            device=self.device,
            requires_grad=False,
        )
        addeff = torch.zeros(
            (len(actions), len(self.propositions)),
            device=self.device,
            requires_grad=False,
        )
        deleff = torch.zeros(
            (len(actions), len(self.propositions)),
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(actions)):
            y_indices = self.indices[actions[i]]
            schema = self.action_to_schema[actions[i]]
            y_indices_set = set(y_indices)

            schema_prams = schema()
            schema_precon = schema_prams @ torch.tensor(
                [0.0, 0.0, 1.0, 1.0], device=self.device
            )
            schema_addeff = schema_prams @ torch.tensor(
                [0.0, 1.0, 0.0, 0.0], device=self.device
            )
            schema_deleff = schema_prams @ torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device
            )

            if len(y_indices) > len(y_indices_set):
                # There are duplicate indices in y_indices
                # Multiple predicates are grounded to one same proposition
                # We need to combine the contribution from different predicates to one proposition
                applied = set()
                for y_idx in y_indices:
                    if y_idx not in applied:
                        precon[i, y_idx] += schema_precon[y_idx]
                        addeff[i, y_idx] += schema_addeff[y_idx]
                        deleff[i, y_idx] += schema_deleff[y_idx]
                        applied.add(y_idx)
                    else:
                        # The multiple effects are combined with "or"
                        # p v q = not ((not p)^(not q))
                        precon[i, y_idx] = 1 - (1 - precon[i, y_idx]) * (
                            1 - schema_precon[y_idx]
                        )
                        addeff[i, y_idx] = 1 - (1 - addeff[i, y_idx]) * (
                            1 - schema_addeff[y_idx]
                        )
                        deleff[i, y_idx] = 1 - (1 - deleff[i, y_idx]) * (
                            1 - schema_deleff[y_idx]
                        )
            else:
                x_indices = [i] * len(y_indices)
                precon[x_indices, y_indices] += schema_precon
                addeff[x_indices, y_indices] += schema_addeff
                deleff[x_indices, y_indices] += schema_deleff
        return precon, addeff, deleff

    @staticmethod
    def create_from_json(json_dict, device):
        type_dict = {}
        predicates = []
        action_schemas = []
        for t in json_dict["types"]:
            type_dict[t["name"]] = Type(t["name"], t["parent"])
        for p in json_dict["predicates"]:
            predicates.append(
                Predicate(
                    p["name"],
                    {type_dict[param]: num for param, num in p["params"].items()},
                )
            )
        for a in json_dict["action_schemas"]:
            action_schemas.append(
                Action_Schema(
                    a["name"],
                    {type_dict[param]: num for param, num in a["params"].items()},
                )
            )
        return Domain_Model(
            list(type_dict.values()), predicates, action_schemas, device
        )


class DomainModelEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Type):
            return {"name": o.name, "parent": o.parent}
        elif isinstance(o, Predicate) or isinstance(o, Action_Schema):
            return {
                "name": o.name,
                "params": {param.name: num for param, num in o.params.items()},
            }
        elif isinstance(o, Domain_Model):
            return {
                "types": o.types,
                "predicates": o.predicates,
                "action_schemas": o.action_schemas,
            }
        else:
            return o.__dict__


def dump_model(domain_model, file_pth):
    with open(file_pth, "w") as f:
        f.write(json.dumps(domain_model, cls=DomainModelEncoder, indent=4))


def load_model(file_pth, device):
    with open(file_pth, "r") as f:
        domain_model = Domain_Model.create_from_json(json.load(f), device)
    return domain_model
