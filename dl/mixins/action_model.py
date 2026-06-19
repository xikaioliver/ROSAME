from ..util.ROSAME.rosame import get_domain_model, extract_pddl
import torch.nn.functional as F

class ROSAMEMixin:
    def _build_aux_around(self, input_shape):
        super()._build_aux_around(input_shape)

    def _build_around(self, input_shape):
        # Build the core domain model.
        self.domain_model = get_domain_model(self.parameters["domain"])
        self.action_list = list(range(len(self.domain_model.actions)))
        self.parameters["prop_dim"] = len(self.domain_model.propositions)
        self.parameters["action_dim"] = len(self.domain_model.actions)
        super()._build_around(input_shape)

    def adim(self):
        return self.parameters["action_dim"]

    def apply(self, z_pre, actions):
        actions = actions.argmax(dim=-1)  # [B, T]
        _, all_addeffs, all_deleffs = self.domain_model(self.action_list)  # [action_dim, symbol_dim]
        addeffs, deleffs = all_addeffs[actions], all_deleffs[actions]  # [B, T, symbol_dim]
        return z_pre * (1 - deleffs) + (1 - z_pre) * addeffs

    def dump_actions(self, *args, **kwargs):
        self.net.eval()
        with open(self.local('domain_model.pddl'), 'w') as file:
            print(extract_pddl(self.domain_model, domain_name=self.parameters["domain"]), file=file)

        with open(self.local('domain_model_parameters.txt'), 'w') as file:
            for schema in self.domain_model.action_schemas:
                print(F.softmax(schema(), dim=-1), file=file)
        return
