# constraint_opt/encodings/cp_sat.py
from __future__ import annotations

import time
from typing import Dict, Tuple, Any, Optional

import cpmpy as cp
from cpmpy import Model, boolvar, SolverLookup
from cpmpy.solvers.solver_interface import ExitStatus

from constraint_opt.util import args2str, unifies
from planning_structs.domain import ActionSchema, Predicate
from planning_structs.instance import Action, Proposition
from planning_structs.traces import ObservationM


class CPSAT:
    """
    CPMpy-backed refactor of problem.py:Problem with identical encoding.

    Notes:
      - Variables are CPMpy boolvar; after solving, use var.value().
    """
    def __init__(self, domain, traces, objectives):
        self.domain = domain
        self.traces = traces
        self.instance = traces.instance
        self.n_traces = len(traces.obs_t)
        self.max_t = traces.obs_t[0].step + 1
        self.objectives = objectives

        self.model = Model()

        self.pre: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}
        self.add: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}
        self.dele: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Any] = {}

        self.act: Dict[Tuple[int, int, Action], Any] = {}
        self.hol: Dict[Tuple[int, int, Proposition], Any] = {}

        self.stepadd: Dict[Tuple[int, int, Proposition], Any] = {}
        self.stepdel: Dict[Tuple[int, int, Proposition], Any] = {}
        self.steppre: Dict[Tuple[int, int, Proposition], Any] = {}

        self.build_domain_variables()
        self.build_domain_constraints()

        self.build_trace_variables()
        self.build_trace_constraints(traces)

        self.obj_scale = 1e10 # hard-coded scale for CP-SAT integer objective
        self.build_objectives(traces)

    # ---------- problem encoding utilities ----------
    def lodge_time(self, mess):
        self.prev_time = time.time()
        print(mess + "...", end="")

    def report_time(self):
        print(round(time.time() - self.prev_time, 4))

    # ---------- variable construction ----------
    def build_domain_variables(self) -> None:
        self.lodge_time("Building Domain Variables")
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    self.pre[(a, p, x)] = boolvar(name=f"pre_{a.name}_{p.name}_{args2str(x)}")
                    self.add[(a, p, x)] = boolvar(name=f"pos_{a.name}_{p.name}_{args2str(x)}")
                    self.dele[(a, p, x)] = boolvar(name=f"neg_{a.name}_{p.name}_{args2str(x)}")
        self.report_time()

    def build_trace_variables(self) -> None:
        # Holds: hol[(i,t,p)]
        self.lodge_time("Building Hol Variables")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t + 1):
                    self.hol[(i, t, p)] = boolvar(name=f"hol_{i}_{t}_{p}")
        self.report_time()

        # Occurs: act[(i,t,a)]
        self.lodge_time("Building Act Variables")
        for a in self.instance.actions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.act[(i, t, a)] = boolvar(name=f"act_{i}_{t}_{a}")
        self.report_time()

        # Step vars: steppre/stepadd/stepdel
        self.lodge_time("Building Step Variables")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.stepadd[(i, t, p)] = boolvar(name=f"stepprod_{i}_{t}_{p}")
                    self.stepdel[(i, t, p)] = boolvar(name=f"stepclob_{i}_{t}_{p}")
                    self.steppre[(i, t, p)] = boolvar(name=f"stepuser_{i}_{t}_{p}")
        self.report_time()

    # ---------- constraints ----------
    def build_domain_constraints(self) -> None:
        self.lodge_time("Building Domain Constraints")
        cons = []
        for a in self.domain.action_schemas:
            epre_terms = []
            eadd_terms = []
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    # dele -> pre
                    cons.append(self.dele[(a, p, x)].implies(self.pre[(a, p, x)]))
                    # not(pre & add)
                    cons.append(~(self.pre[(a, p, x)] & self.add[(a, p, x)]))
                    epre_terms.append(self.pre[(a, p, x)])
                    eadd_terms.append(self.add[(a, p, x)])
            cons.append(cp.sum(epre_terms) >= 1)
            cons.append(cp.sum(eadd_terms) >= 1)
        self.model += cons
        self.report_time()

    def build_trace_constraints(self, traces) -> None:
        cons = []

        # Initial state at t=1
        self.lodge_time("Building Initial State Constraints")
        for i in range(1, self.n_traces + 1):
            for p in self.instance.propositions:
                if p in traces.obs_t[i - 1].init:
                    cons.append(self.hol[(i, 1, p)])
                else:
                    cons.append(~self.hol[(i, 1, p)])
        self.report_time()

        # Goal state at t=max_t
        self.lodge_time("Building Goal Constraints")
        for i in range(1, self.n_traces + 1):
            for p in self.instance.propositions:
                if p in traces.obs_t[i - 1].goal:
                    cons.append(self.hol[(i, self.max_t, p)])
                else:
                    cons.append(~self.hol[(i, self.max_t, p)])
        self.report_time()

        # Only one action per step
        self.lodge_time("Building Action Exclusion Constraints")
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                cons.append(cp.sum(self.act[(i, t, a)] for a in self.instance.actions) == 1)
        self.report_time()

        # Use action execution as indicator
        # stepadd(p,t) = 1 iff exist (a, x) such that x unifies the arguments of p and a and
        # p.predicate is in add[a.action_schema] and act(a, t)
        self.lodge_time("Building Step Pre/Add/Del Constraints")
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                for p in self.instance.propositions:
                    cand_add = []
                    cand_del = []
                    cand_pre = []
                    for a in self.instance.actors[p]:
                        for x in [
                            x for (a1, pred, x) in self.add
                            if a1.name == a.action_schema.name
                               and pred.name == p.predicate.name
                               and unifies(x, p.args, a.args)
                        ]:
                            a_schema = a.action_schema
                            p_pred = p.predicate
                            cand_add.append(self.act[(i, t, a)] & self.add[(a_schema, p_pred, x)])
                            cand_del.append(self.act[(i, t, a)] & self.dele[(a_schema, p_pred, x)])
                            cand_pre.append(self.act[(i, t, a)] & self.pre[(a_schema, p_pred, x)])
                    cons.append(self.stepadd[(i, t, p)] == cp.any(cand_add))
                    cons.append(self.stepdel[(i, t, p)] == cp.any(cand_del))
                    cons.append(self.steppre[(i, t, p)] == cp.any(cand_pre))
                    cons.append(~(self.stepadd[(i, t, p)] & self.hol[(i, t, p)]))
        self.report_time()

        # Successor state + precondition constraints
        self.lodge_time("Building Successor State and Precondition Constraints")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    cons.append(self.stepadd[(i, t, p)].implies(self.hol[(i, t + 1, p)]))
                    cons.append(self.stepdel[(i, t, p)].implies(~self.hol[(i, t + 1, p)]))
                    cons.append(self.steppre[(i, t, p)].implies(self.hol[(i, t, p)]))
        self.report_time()

        # Frame axioms
        self.lodge_time("Building Frame Constraints")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    cons.append((~self.hol[(i, t, p)] & self.hol[(i, t + 1, p)]).implies(self.stepadd[(i, t, p)]))
                    cons.append((self.hol[(i, t, p)] & ~self.hol[(i, t + 1, p)]).implies(self.stepdel[(i, t, p)]))
        self.report_time()

        self.model += cons

    # ---------- objective ----------
    def build_objectives(self, traces) -> None:
        self.lodge_time("Building Objectives")

        def w(prob: float) -> int:
            # prob in [0,1] -> weight in [0 S]
            return int(round((2.0 * float(prob) - 1.0) * self.obj_scale))

        terms = []
        obs_t = traces.obs_t

        if "state" in self.objectives:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t + 1):
                    for op in obs_t[i - 1].obs_p[t]:
                        coef = w(op.prob)
                        if coef:  # skip exact zeros to keep the model smaller
                            terms.append(coef * self.hol[(i, t, op.proposition)])

        if "action" in self.objectives:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    for oa in obs_t[i - 1].obs_a[t]:
                        coef = w(oa.prob)
                        if coef:
                            terms.append(coef * self.act[(i, t, oa.action)])

        if "model" in self.objectives:
            obs_m = traces.obs_m
            for a in self.domain.action_schemas:
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        c_pre = w(obs_m.pre[a, p, x])
                        if c_pre:
                            terms.append(c_pre * self.pre[(a, p, x)])

                        c_add = w(obs_m.add[a, p, x])
                        if c_add:
                            terms.append(c_add * self.add[(a, p, x)])

                        c_del = w(obs_m.dele[a, p, x])
                        if c_del:
                            terms.append(c_del * self.dele[(a, p, x)])

        if terms:
            self.model.maximize(cp.sum(terms))

        self.report_time()

    # ---------- solving ----------
    def make_solution_hints(self, threshold: float = 0.5):
        """
        Provide Gurobi *solution hints* (VarHintVal) via CPMpy's `solution_hint`,
        based on probabilistic observations.

        Parameters
        ----------
        threshold : float
            Probability threshold to map probabilities to {0,1} hints.
        """
        hint_vars = []
        hint_vals = []

        def add_hint(var, val: int):
            # CPMpy expects lists of CPMpy variables and corresponding values.
            # We avoid duplicates to keep the hint list clean; however, if duplicates
            # occur, later duplicates are harmless but wasteful.
            hint_vars.append(var)
            hint_vals.append(int(val))

        # ----- State / proposition hints -----
        obs_t = self.traces.obs_t
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t + 1):
                for op in obs_t[i - 1].obs_p[t]:
                    var = self.hol[(i, t, op.proposition)]  # CPMpy BoolVar/IntVar
                    add_hint(var, 1 if op.prob > threshold else 0)

        # ----- Action hints -----
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                for oa in obs_t[i - 1].obs_a[t]:
                    var = self.act[(i, t, oa.action)]
                    add_hint(var, 1 if oa.prob > threshold else 0)

        # ----- Action model hints (pre/add/del) -----
        obs_m = self.traces.obs_m
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    add_hint(self.pre[(a, p, x)], 1 if obs_m.pre[a, p, x] > threshold else 0)
                    add_hint(self.add[(a, p, x)], 1 if obs_m.add[a, p, x] > threshold else 0)
                    add_hint(self.dele[(a, p, x)], 1 if obs_m.dele[a, p, x] > threshold else 0)

        return hint_vars, hint_vals

    def solve(self, time_limit: Optional[int] = None) -> bool:
        solver = SolverLookup.get("ortools", self.model)
        hint_vars, hint_vals = self.make_solution_hints()
        solver.solution_hint(hint_vars, hint_vals)
        solver.solve(time_limit=time_limit, log_search_progress=True)

        st = solver.status()

        if st.exitstatus == ExitStatus.UNKNOWN:
            print("No feasible solution found within time limit.")
            return False
        elif st.exitstatus == ExitStatus.UNSATISFIABLE:
            print("Problem is infeasible.")
            return False
        else:
            self.print_solution()
            return True

    # ---------- post-processing utilities ----------
    def print_solution(self):
        for a in self.domain.action_schemas:
            print("{}()".format(a.name, list(range(1, a.arity + 1))))
            print("  Preconditions:")
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    if (a, p, x) in self.pre and bool(self.pre[(a, p, x)].value()):
                        print("    {}({})".format(p.name, x))
            print("  Add effects:")
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    if (a, p, x) in self.add and bool(self.add[(a, p, x)].value()):
                        print("    {}({})".format(p.name, x))
            print("  Delete effects:")
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    if (a, p, x) in self.dele and bool(self.dele[(a, p, x)].value()):
                        print("    {}({})".format(p.name, x))

        print("Other variables")

        for i in range(1, self.n_traces + 1):
            print(f"Trace {i}")

            print("\n Action occurrences")
            for t in range(1, self.max_t):
                for a in self.instance.actions:
                    if (i, t, a) in self.act and bool(self.act[(i, t, a)].value()):
                        print("{} occurs at {}".format(a, t))

            print("\n Propositions holding")
            for t in range(1, self.max_t + 1):
                for p in self.instance.propositions:
                    if (i, t, p) in self.hol and bool(self.hol[(i, t, p)].value()):
                        print("{} holds at {}".format(p, t))

    def action_model_sol(self) -> ObservationM:
        pre = {(a, p, x): int(self.pre[(a, p, x)].value()) for (a, p, x) in self.pre}
        add = {(a, p, x): int(self.add[(a, p, x)].value()) for (a, p, x) in self.add}
        dele = {(a, p, x): int(self.dele[(a, p, x)].value()) for (a, p, x) in self.dele}
        return ObservationM(pre, add, dele)

    def _bool_value(self, var) -> bool:
        return var.value()

def build_cp_sat_encoding(domain, traces, objectives):
    """Factory entrypoint: build a CPMpy encoding for the MIP-style fixer."""
    return CPSAT(domain=domain, traces=traces, objectives=objectives)