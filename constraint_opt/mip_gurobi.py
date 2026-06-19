from typing import Dict, Tuple, Optional

from gurobipy import *
import time

from constraint_opt.util import args2str, unifies
from planning_structs.domain import ActionSchema, Predicate
from planning_structs.instance import Action, Proposition
from planning_structs.traces import ObservationM


class MIPGurobi:
    def __init__(self, domain, traces, objectives):
        self.domain = domain
        self.traces = traces
        self.instance = traces.instance
        self.n_traces = len(traces.obs_t)
        self.max_t = traces.obs_t[0].step + 1
        self.objectives = objectives

        self.model = Model()

        self.pre: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Var] = {}
        self.add: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Var] = {}
        self.dele: Dict[Tuple[ActionSchema, Predicate, Tuple[int, ...]], Var] = {}

        self.act: Dict[Tuple[int, int, Action], Var] = {}
        self.hol: Dict[Tuple[int, int, Proposition], Var] = {}

        self.stepadd: Dict[Tuple[int, int, Proposition], Var] = {}
        self.stepdel: Dict[Tuple[int, int, Proposition], Var] = {}
        self.steppre: Dict[Tuple[int, int, Proposition], Var] = {}

        self.build_domain_variables()
        self.build_domain_constraints()

        self.build_trace_variables()
        self.build_trace_constraints(traces)

        self.build_objectives(traces)

        self.warm_start()

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
                    self.pre[(a, p, x)] = self.model.addVar(vtype=GRB.BINARY, name=f"pre_{a.name}_{p.name}_{args2str(x)}")
                    self.add[(a, p, x)] = self.model.addVar(vtype=GRB.BINARY, name=f"add_{a.name}_{p.name}_{args2str(x)}")
                    self.dele[(a, p, x)] = self.model.addVar(vtype=GRB.BINARY, name=f"del_{a.name}_{p.name}_{args2str(x)}")
        self.report_time()

    def build_trace_variables(self) -> None:
        # Holds: hol[(i,t,p)]
        self.lodge_time("Building Hol Variables")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t + 1):
                    self.hol[(i, t, p)] = self.model.addVar(vtype=GRB.BINARY, name=f"hol_{i}_{t}_{p}")
        self.report_time()

        # Occurs: act[(i,t,a)]
        self.lodge_time("Building Act Variables")
        for a in self.instance.actions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.act[(i, t, a)] = self.model.addVar(vtype=GRB.BINARY, name=f"act_{i}_{t}_{a}")
        self.report_time()

        # Step vars: steppre/stepadd/stepdel
        self.lodge_time("Building Step Variables")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.stepadd[(i, t, p)] = self.model.addVar(vtype=GRB.BINARY, name=f"stepadd_{i}_{t}_{p}")
                    self.stepdel[(i, t, p)] = self.model.addVar(vtype=GRB.BINARY, name=f"stepdel_{i}_{t}_{p}")
                    self.steppre[(i, t, p)] = self.model.addVar(vtype=GRB.BINARY, name=f"steppre_{i}_{t}_{p}")
        self.report_time()

    # ---------- constraints ----------
    def build_domain_constraints(self) -> None:
        self.lodge_time("Building Domain Constraints")
        for a in self.domain.action_schemas:
            epre = LinExpr()
            eadd = LinExpr()
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    self.model.addConstr(self.pre[(a, p, x)] >= self.dele[(a, p, x)],f"DelIsPre_{a.name}_{p.name}_{args2str(x)}")
                    self.model.addConstr(self.pre[(a, p, x)] + self.add[(a, p, x)] <= 1, f"AddIsNotPre_{a.name}_{p.name}_{args2str(x)}")
                    epre.add(self.pre[(a, p, x)], 1.0)
                    eadd.add(self.add[(a, p, x)], 1.0)
            self.model.addConstr(epre >= 1, "PreIsNotEmpty_" + str(a.name))
            self.model.addConstr(eadd >= 1, "AddIsNotEmpty_" + str(a.name))
        self.report_time()

    def build_trace_constraints(self, traces) -> None:
        # Initial state at t=1
        self.lodge_time("Building Initial State Constraints")
        for i in range(1, self.n_traces + 1):
            for p in self.instance.propositions:
                if p in traces.obs_t[i - 1].init:
                    self.model.addConstr(self.hol[(i, 1, p)] == 1, f"InitT_{i}_{p}")
                else:
                    self.model.addConstr(self.hol[(i, 1, p)] == 0, f"InitF_{i}_{p}")
        self.report_time()

        # Goal state at t=max_t
        self.lodge_time("Building Goal Constraints")
        for i in range(1, self.n_traces + 1):
            for p in self.instance.propositions:
                if p in traces.obs_t[i - 1].goal:
                    self.model.addConstr(self.hol[(i, self.max_t, p)] == 1, f"GoalT_{i}_{p}")
                else:
                    self.model.addConstr(self.hol[(i, self.max_t, p)] == 0, f"GoalF_{i}_{p}")
        self.report_time()

        # Only one action per step
        self.lodge_time("Building Action Exclusion Constraints")
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                et = LinExpr()
                for a in self.instance.actions:
                    et.add(self.act[i, t, a], 1.0)
                self.model.addConstr(et == 1, f"OneAction_{i}_{t}")
        self.report_time()

        # Use action execution as indicator
        # stepadd(p,t) = 1 iff exist a such that act(a, t) and exist (pred, x) in add[a] such that x unifies the arguments of p and a
        # act(a, t) = 1 implies stepadd(p, t) = self.add[(a.action_schema, p.predicate, x)], otherwise stepadd(p, t) is free
        # finally no such action executed implies stepadd(p, t) = 0
        self.lodge_time("Building Step Pre/Add/Del Constraints")
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                for p in self.instance.propositions:
                    eact = LinExpr()
                    for a in self.instance.actors[p]:
                        eact.add(self.act[(i, t, a)], 1.0)
                        for x in [
                            x for (a1, pred, x) in self.add
                            if a1.name == a.action_schema.name
                               and pred.name == p.predicate.name
                               and unifies(x, p.args, a.args)
                        ]:
                            a_schema = a.action_schema
                            p_pred = p.predicate
                            self.model.addConstr(
                                self.stepadd[(i, t, p)] <= self.add[(a_schema, p_pred, x)] + 1 - self.act[(i, t, a)],
                                f"StepAddDefa_{i}_{t}_{p}_{a}")
                            self.model.addConstr(
                                self.stepadd[(i, t, p)] >= self.add[(a_schema, p_pred, x)] + self.act[(i, t, a)] - 1,
                                f"StepAddDefb_{i}_{t}_{p}_{a}")
                            self.model.addConstr(
                                self.stepdel[(i, t, p)] <= self.dele[(a_schema, p_pred, x)] + 1 - self.act[(i, t, a)],
                                f"StepDelDefa_{i}_{t}_{p}_{a}")
                            self.model.addConstr(
                                self.stepdel[(i, t, p)] >= self.dele[(a_schema, p_pred, x)] + self.act[(i, t, a)] - 1,
                                f"StepDelDefb_{i}_{t}_{p}_{a}")
                            self.model.addConstr(
                                self.steppre[(i, t, p)] <= self.pre[(a_schema, p_pred, x)] + 1 - self.act[(i, t, a)],
                                f"StepPreDefa_{i}_{t}_{p}_{a}")
                            self.model.addConstr(
                                self.steppre[(i, t, p)] >= self.pre[(a_schema, p_pred, x)] + self.act[(i, t, a)] - 1,
                                f"StepPreDefb_{i}_{t}_{p}_{a}")
                    self.model.addConstr(self.stepadd[(i, t, p)] <= eact, f"StepAddDefc_{i}_{t}_{p}")
                    self.model.addConstr(self.stepdel[(i, t, p)] <= eact, f"StepDelDefc_{i}_{t}_{p}")
                    self.model.addConstr(self.steppre[(i, t, p)] <= eact, f"StepPreDefc_{i}_{t}_{p}")
                    self.model.addConstr(self.stepadd[(i, t, p)] + self.hol[(i, t, p)] <= 1, f"StepAddPre_{i}_{t}_{p}")
        self.report_time()

        # Successor state + precondition constraints
        self.lodge_time("Building Successor State and Precondition Constraints")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.model.addConstr(self.hol[(i, t + 1, p)] >= self.stepadd[(i, t, p)], f"PosAdd_{i}_{t}_{p}")
                    self.model.addConstr(1 - self.hol[(i, t + 1, p)] >= self.stepdel[(i, t, p)], f"NegDel_{i}_{t}_{p}")
                    self.model.addConstr(self.hol[(i, t, p)] >= self.steppre[(i, t, p)], f"PosPre_{i}_{t}_{p}")
        self.report_time()

        # Frame axioms
        self.lodge_time("Building Frame Constraints")
        for p in self.instance.propositions:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    self.model.addConstr(self.stepadd[(i, t, p)] >= self.hol[(i, t + 1, p)] - self.hol[(i, t, p)],
                                         f"FrameAdd_{i}_{t}_{p}")
                    self.model.addConstr(self.stepdel[(i, t, p)] >= self.hol[(i, t, p)] - self.hol[(i, t + 1, p)],
                                         f"FrameDel_{i}_{t}_{p}")
        self.report_time()

    # ---------- objective ----------
    def build_objectives(self, traces) -> None:
        self.lodge_time("Building Objectives")
        obj_terms = LinExpr()
        obs_t = traces.obs_t

        if "state" in self.objectives:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t + 1):
                    for op in obs_t[i - 1].obs_p[t]:
                        obj_terms.add(self.hol[(i, t, op.proposition)], 2 * op.prob - 1)

        if "action" in self.objectives:
            for i in range(1, self.n_traces + 1):
                for t in range(1, self.max_t):
                    for oa in obs_t[i - 1].obs_a[t]:
                        obj_terms.add(self.act[(i, t, oa.action)], 2 * oa.prob - 1)

        if "model" in self.objectives:
            obs_m = traces.obs_m
            for a in self.domain.action_schemas:
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        obj_terms.add(self.pre[(a, p, x)], 2 * obs_m.pre[a, p, x] - 1)
                        obj_terms.add(self.add[(a, p, x)], 2 * obs_m.add[a, p, x] - 1)
                        obj_terms.add(self.dele[(a, p, x)], 2 * obs_m.dele[a, p, x] - 1)
        self.model.setObjective(obj_terms, GRB.MAXIMIZE)
        self.report_time()

    # ---------- solve ----------
    def warm_start(self):
        self.lodge_time("Assigning Warm Start Values")
        obs_t = self.traces.obs_t

        # ----- State / proposition hints -----
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t + 1):
                for op in obs_t[i - 1].obs_p[t]:
                    self.hol[(i, t, op.proposition)].Start = 1 if op.prob > 0.5 else 0

        # ----- Action hints -----
        for i in range(1, self.n_traces + 1):
            for t in range(1, self.max_t):
                for oa in obs_t[i - 1].obs_a[t]:
                    self.act[(i, t, oa.action)].Start = 1 if oa.prob > 0.5 else 0

        # ----- Action model hints (pre/add/del) -----
        obs_m = self.traces.obs_m
        for a in self.domain.action_schemas:
            for p in self.domain.predicates:
                for x in self.domain.predicate_arguments[(a, p)]:
                    self.pre[(a, p, x)].Start = 1 if obs_m.pre[a, p, x] > 0.5 else 0
                    self.add[(a, p, x)].Start = 1 if obs_m.add[a, p, x] > 0.5 else 0
                    self.dele[(a, p, x)].Start = 1 if obs_m.dele[a, p, x] > 0.5 else 0
        self.report_time()

    def solve(self, time_limit: Optional[int] = None) -> bool:
        self.model.update()
        self.model.setParam('TimeLimit', time_limit)
        self.model.optimize()
        self.report_time()
        if self.model.SolCount == 0:
            print("No feasible solution found within time limit.")
            return False
        elif self.model.status == GRB.INFEASIBLE:
            print("Problem is infeasible.")
            self.model.computeIIS()
            if self.model.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
            return False
        else:
            for a in self.domain.action_schemas:
                print("{}()".format(a.name, list(range(1, a.arity + 1))))
                print("  Preconditions:")
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        if (a, p, x) in self.pre and self.pre[(a, p, x)].X:
                            print("    {}({})".format(p.name, x))
                print("  Add effects:")
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        if (a, p, x) in self.add and self.add[(a, p, x)].X:
                            print("    {}({})".format(p.name, x))
                print("  Delete effects:")
                for p in self.domain.predicates:
                    for x in self.domain.predicate_arguments[(a, p)]:
                        if (a, p, x) in self.dele and self.dele[(a, p, x)].X:
                            print("    {}({})".format(p.name, x))

            print("Other variables")

            for i in range(1, self.n_traces + 1):
                print(f"Trace {i}")

                print("\n Action occurrences")
                for t in range(1, self.max_t):
                    for a in self.instance.actions:
                        if (i, t, a) in self.act and self.act[(i, t, a)].X:
                            print("{} occurs at {}".format(a, t))

                print("\n Propositions holding")
                for t in range(1, self.max_t + 1):
                    for p in self.instance.propositions:
                        if (i, t, p) in self.hol and self.hol[(i, t, p)].X:
                            print("{} holds at {}".format(p, t))
            return True

    # ---------- post-processing utilities ----------
    def action_model_sol(self) -> ObservationM:
        pre = {(a, p, x): self.pre[(a, p, x)].X for (a, p, x) in self.pre}
        add = {(a, p, x): self.add[(a, p, x)].X for (a, p, x) in self.add}
        dele = {(a, p, x): self.dele[(a, p, x)].X for (a, p, x) in self.dele}
        return ObservationM(pre, add, dele)

    def _bool_value(self, var) -> bool:
        return var.X > 0.5

def build_mip_gurobi_encoding(domain, traces, objectives):
    """Factory entrypoint: build a CPMpy encoding for the MIP-style fixer."""
    return MIPGurobi(domain=domain, traces=traces, objectives=objectives)