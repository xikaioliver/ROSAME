# translator.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch

from planning_structs.domain import ActionSchema, Predicate
from planning_structs.traces import ObservationP, ObservationA, ObservationM, ObservationT


class Translator:
    """
    Translator between:
      (A) deep learning predictions -> CP/constraint-optimization objectives (ObservationM, ObservationT), and
      (B) CP/constraint-optimization solutions -> deep learning pseudo-labels.

    CPMpy compatibility contract:
      - decision variables are CPMpy expressions/vars that expose `.value()` after solve.
      - the `problem`/`encoding` object passed to `extract_*` provides:
          problem.n_traces: int
          problem.max_t: int
          problem.instance: Instance-like (has propositions/actions)
          problem.hol[(i,t,p)] : CPMpy var for holds
          problem.act[(i,t,a)] : CPMpy var for action occurs
          problem.pre[(a,p,x)], problem.add[(a,p,x)], problem.dele[(a,p,x)] : CPMpy vars
        Optional (if you keep permutation logic):
          problem.name_ext_int, problem.args_ext_int
          problem.instance.get_permuted_action(action, name_ext_int, args_ext_int)
          problem.mip_domain.get_permuted_action_schema(...)
    """

    def __init__(self, mip_domain, rosame_domain, instance):
        self.mip_domain = mip_domain
        self.rosame_domain = rosame_domain
        self.instance = instance

    # -------------------------
    # (A) DL -> objectives (obs_t, obs_m)
    # -------------------------

    def trans_full_state(self, state: torch.Tensor):
        return [prop for value, prop in zip(state, self.instance.propositions) if value == 1]

    def trans_obs_tr(
        self,
        state_preds: torch.Tensor,
        action_preds: torch.Tensor,
        init: torch.Tensor,
        goal: torch.Tensor,
        eps: float = 1e-5,
    ) -> ObservationT:
        """
        Translate one trace worth of DL predictions into an ObservationT.

        Mirrors your existing logic:
          - clamp predictions
          - build obs_p for intermediate steps (t=2..T)
          - set full init row at t=1 and full goal row at t=T+1
          - build obs_a for action distributions at t=1..T
        """
        state_preds = state_preds.clamp(eps, 1 - eps).cpu().tolist()
        action_preds = action_preds.clamp(eps, 1 - eps).cpu().tolist()

        step = len(state_preds) + 1

        obs_p = {
            t + 2: [ObservationP(prop, prob) for prop, prob in zip(self.instance.propositions, probs)]
            for t, probs in enumerate(state_preds)
        }

        init_props = self.trans_full_state(init)
        obs_p[1] = [
            ObservationP(prop, 1 - eps) if prop in init_props else ObservationP(prop, eps)
            for prop in self.instance.propositions
        ]

        goal_props = self.trans_full_state(goal)
        obs_p[step + 1] = [
            ObservationP(prop, 1 - eps) if prop in goal_props else ObservationP(prop, eps)
            for prop in self.instance.propositions
        ]

        obs_a = {
            t + 1: [ObservationA(act, prob) for act, prob in zip(self.instance.actions, probs)]
            for t, probs in enumerate(action_preds)
        }

        return ObservationT(step, init_props, obs_p, goal_props, obs_a)

    def trans_rosame(self) -> ObservationM:
        """
        Translate current ROSAME action model parameters into an ObservationM (pre/add/del probabilities).
        """
        obs_pre: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], float] = {}
        obs_add: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], float] = {}
        obs_del: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], float] = {}

        for schema in self.rosame_domain.action_schemas:
            schema_ros_pred = self.rosame_domain.activation(schema())
            schema_ros_pre = schema_ros_pred @ torch.tensor([0.0, 0.0, 1.0, 1.0], device=schema_ros_pred.device)
            schema_ros_add = schema_ros_pred @ torch.tensor([0.0, 1.0, 0.0, 0.0], device=schema_ros_pred.device)
            schema_ros_del = schema_ros_pred @ torch.tensor([0.0, 0.0, 0.0, 1.0], device=schema_ros_pred.device)

            a = self.mip_domain.get_action_schema(schema.name)
            idx = 0
            for predicate in schema.predicates:
                p = self.mip_domain.get_predicate(predicate.name)
                for x in self.mip_domain.predicate_arguments[(a, p)]:
                    obs_pre[(a, p, x)] = float(schema_ros_pre[idx].item())
                    obs_add[(a, p, x)] = float(schema_ros_add[idx].item())
                    obs_del[(a, p, x)] = float(schema_ros_del[idx].item())
                    idx += 1

        return ObservationM(obs_pre, obs_add, obs_del)

    # -------------------------
    # (B) CP solution -> pseudo-labels
    # -------------------------
    def extract_sol_label(self, problem, name_dl_cp, args_dl_cp) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Extract per-trace state/action pseudo-labels from a solved CPMpy model.

        This corresponds to your previous `extract_sol_label` :contentReference[oaicite:6]{index=6},
        but reads CPMpy variable values rather than Gurobi `.X`.
        """
        state_labels = []
        action_labels = []

        for i in range(1, problem.n_traces + 1):
            state_label = torch.zeros(problem.max_t - 2, len(self.instance.propositions))
            action_label = torch.zeros(problem.max_t - 1, dtype=torch.long)

            # states: t = 2..max_t-1 (exclude init=1 and goal=max_t+1)
            for t in range(2, problem.max_t):
                for j, p in enumerate(self.instance.propositions):
                    state_label[t - 2, j] = 1.0 if problem._bool_value(problem.hol[(i, t, p)]) else 0.0

            # actions: pick the first action with act==1 at each t
            for t in range(1, problem.max_t):
                chosen = 0  # default; keep current behavior
                found = False
                for j, action in enumerate(self.instance.actions):
                    a = problem.instance.get_permuted_action(action, name_dl_cp, args_dl_cp)
                    if problem._bool_value(problem.act[(i, t, a)]):
                        chosen = j
                        found = True
                        break
                action_label[t - 1] = chosen if found else 0

            state_labels.append(state_label)
            action_labels.append(action_label)

        return state_labels, action_labels

    def extract_sol_model(self, problem, name_dl_cp, args_dl_cp) -> Dict[str, torch.Tensor]:
        """
        Extract pseudo-labels for the learned action model (pre/add/del/none) from a solved CPMpy model.
        """
        model_label: Dict[str, torch.Tensor] = {}

        for schema in self.rosame_domain.action_schemas:
            schema_label = torch.zeros(schema.randn.shape[0], 4)
            a_cp = self.mip_domain.get_permuted_action_schema(schema.name, name_dl_cp)

            idx = 0
            for predicate in schema.predicates:
                p = self.mip_domain.get_predicate(predicate.name)
                for x in self.mip_domain.predicate_arguments[(a_cp, p)]:
                    if args_dl_cp is not None:
                        x_cp = tuple(args_dl_cp[schema.name][i - 1] for i in x)
                    else:
                        x_cp = x

                    is_pre = problem._bool_value(problem.pre[(a_cp, p, x_cp)])
                    is_add = problem._bool_value(problem.add[(a_cp, p, x_cp)])
                    is_del = problem._bool_value(problem.dele[(a_cp, p, x_cp)])

                    # one-hot with your precedence order (add > del > pre > none)
                    if is_add:
                        schema_label[idx, 1] = 1
                    elif is_del:
                        schema_label[idx, 3] = 1
                    elif is_pre:
                        schema_label[idx, 2] = 1
                    else:
                        schema_label[idx, 0] = 1
                    idx += 1

            model_label[schema.name] = schema_label

        return model_label
