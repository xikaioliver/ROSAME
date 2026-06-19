class ObservationP:
    """Create ObservationP (an observation of a proposition)
       __init__: Proposition x real -> ObservationP
      parameters:
      - proposition: Proposition that we're observing
      - proba: probability of the observation
      result:
      - an proposition Observation
    """

    def __init__(self, proposition, prob):
        self.proposition = proposition
        self.prob = prob

    def __str__(self):
        return ("Pr(" + str(self.proposition) + ") = " + str(self.prob))


class ObservationA:
    """Create ObservationA (an observation of an Action)
       __init__: Action x real -> ObservationA
      parameters:
      - action: Action that we're observing
      - proba: probability of the observation
      result:
      - an action Observation
    """

    def __init__(self, action, prob):
        self.action = action
        self.prob = prob

    def __str__(self):
        return ("Pr(" + str(self.action) + ") = " + str(self.prob))


class ObservationM:
    def __init__(self, obs_pre, obs_add, obs_del):
        self.pre = obs_pre
        self.add = obs_add
        self.dele = obs_del


class ObservationT:
    def __init__(self, step, init, obs_p, goal, obs_a):
        self.step = step
        self.obs_a = obs_a
        self.obs_p = obs_p
        self.init = init
        self.goal = goal

    def __str__(self):
        result = ""
        for t in range(1, self.step + 2):
            result += "time = " + str(t) + ":\n"
            result += "\n  ".join(list(map(lambda o: str(o), self.obs_p[t]))) + "\n"
            if t <= self.step:
                result += "\n  ".join(list(map(lambda o: str(o), self.obs_a[t]))) + "\n"
                # REMOVE (was if we knew the action)
                # result +="==> "+str(self.observation_a[t])+"\n"
        return (result)


class Traces:
    ''' Create a planning problem Trace
    __init__: Instance x (dict[int]->list[ObservationA] x dict[int]->list[ObservationP]) x list[str] -> Instance
      parameters:
      - obs_m: action model observations
      - obs_t: trace sequence observations
    '''
    def __init__(self, instance, obs_m, obs_t):
        self.instance = instance
        self.obs_m = obs_m
        self.obs_t = obs_t