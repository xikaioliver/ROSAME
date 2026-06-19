# convertor.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch

from constraint_opt.factory import resolve
from planning_structs.instance import Instance
from planning_structs.traces import Traces
from planning_structs.util import load_domain_from_json
from util.model_perm import model_permutation

from .translator import Translator
from .pseudo_label import PseudoLabels

# Optional legacy diagnostic: keep if you still use it
from util.pddl_parsing import parse_pddl_domain


class Convertor:
    """
    General Convertor:
      - DL predictions -> CP objectives (Translator)
      - CP solve (CPMpy)
      - CP solution -> pseudo-labels (Translator)
    """

    def __init__(
        self,
        rosame_domain,
        objectives,
        domain_name: str,
        problem_type: str,
    ):
        self.rosame_domain = rosame_domain
        self.objectives = objectives

        # ---- legacy domain name normalization (keep as requested) ----
        domain_name = "blocksworld" if domain_name == "blocks" else domain_name

        # ---- load domain + instance (legacy behavior retained) ----
        spec_path = Path(Path(__file__).parent.parent, "planning_structs", "specs", domain_name)
        domain_path = spec_path / "domain.json"
        self.domain = load_domain_from_json(domain_path)

        self.objects = [(o, t.name) for t, obj_list in rosame_domain.objects.items() for o in obj_list]
        self.instance = Instance(self.domain, self.objects)

        # ground-truth action model for permutation diagnostics
        self.gt_am = parse_pddl_domain(self.domain, Path(__file__).parent.parent / "pddl" / domain_name / "domain.pddl")

        # ---- translator + pseudo labels ----
        self.translator = Translator(self.domain, rosame_domain, self.instance)
        self.pseudo_labels = PseudoLabels()

        # factory lookup; raises if invalid
        self.problem_builder = resolve(problem_type)

    def encode_cp_problem(self, traces: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        """
        Build CP objectives from per-trace packs: trace_packs = [(state, action, init, goal), ...]
        """
        obs_m = self.translator.trans_rosame()
        obs_t = [self.translator.trans_obs_tr(s, a, i, g) for (s, a, i, g) in traces]

        # Delegate encoding/build to factory-resolved builder
        return self.problem_builder(self.domain, Traces(self.instance, obs_m, obs_t), self.objectives)

    def create_pseudo_label(self, problem, indices, names_dl_cp, args_dl_cp):
        """
        Extract pseudo-labels from a *solved* problem and update the pseudo-label store.
        This method does NOT solve; it only reads solution values.
        """
        state_labels, action_labels = self.translator.extract_sol_label(problem, names_dl_cp, args_dl_cp)
        model_label = self.translator.extract_sol_model(problem, names_dl_cp, args_dl_cp)
        self.pseudo_labels.update(list(indices), state_labels, action_labels, model_label)

    def run_fixer(self, trace_selector, time_limit: Optional[int] = None):
        """
        Top-level controller:
          - collect traces + indices
          - build CP problem
          - solve CP problem
          - create/update pseudo-labels
          - optionally return model permutation diagnostic
        """
        traces = trace_selector.collect_traces()
        if len(traces) == 0:
            return None

        indices = trace_selector.collect_indices()
        problem = self.encode_cp_problem(traces)

        # problem wrapper is responsible for solving with the chosen solver
        hassol = problem.solve(time_limit=time_limit)
        if not hassol:
            return None
        else:
            action_model_sol = problem.action_model_sol()
            name_dl_cp, args_dl_cp, _, _ = model_permutation(problem.domain, action_model_sol, problem.traces.obs_m)

        self.create_pseudo_label(problem, indices, name_dl_cp, args_dl_cp)

        # return highest_agree and lowest_error for logging
        if self.gt_am is not None:
            _, _, highest_agree, lowest_error = model_permutation(problem.domain, action_model_sol, self.gt_am)
            return 1-highest_agree # for legacy matching
        else:
            return None
