import json
import os.path

from .mixins.pair import TraceWrapper
from .mixins.plot import TracePlotMixin
from .network import Network
from .mixins.encoder_decoder import *
from .mixins.action import *
from .mixins.action_model import *
from .mixins.output import *
from .util import dapply


# Style guide
# * Each section is divided by 3 newlines
# * Each class is separated by 2 newline (so that hideshow will separate them by 1 newline when folded)
# utilities ####################################################################

def get(name):
    return globals()[name]

def get_ae_type(directory):
    with open(os.path.join(directory,"aux.json"),"r") as f:
        return json.load(f)["class"]

def load(directory,allow_failure=False):
    if allow_failure:
        try:
            classobj = get(get_ae_type(directory))
        except FileNotFoundError as c:
            print("Error skipped:", c)
            return None
    else:
        classobj = get(get_ae_type(directory))
    return classobj(directory).load(allow_failure=allow_failure)

# Network mixins ###############################################################
class AE(Network):
    """Autoencoder class. Supports SAVE and LOAD, as well as REPORT methods.
    Additionally, provides ENCODE / DECODE / AUTOENCODE / AUTODECODE methods.
    The latter two are used for verifying the performance of the AE.
    """
    def encode(self, input, **kwargs):
        return self.encoder(input, **kwargs)

    def decode(self, input, **kwargs):
        return self.decoder(input, **kwargs)

    def autoencode(self, input, **kwargs):
        return self.autoencoder(input, **kwargs)


class StateAE(EncoderDecoderMixin, AE):
    """"An Autoencoder whose latent layer is BinaryConcrete.
    Note: references to self.parameters[key] are all hyperparameters.
    """
    def zdim(self):
        return self.fdim()

    def _build_around(self,input_shape):
        super()._build_around(input_shape)
        self.output = VanillaRenderer()
        self.output.parameters = self.parameters

    @torch.no_grad()
    def render(self,x):
        return self.output.render(x)


class ROSAMEGoal(TracePlotMixin, ROSAMEMixin, DetActionMixin, TraceWrapper, StateAE, ResNetMixin):
    def _build_around(self, input_shape):
        self.input_shape = input_shape
        super()._build_around(input_shape)

    def _build_primary(self, input_shape):
        # Use a helper method to create the `Net` instance
        self.net = self._create_net()
        # Register metrics
        self._register_metrics()

    def _create_net(self):
        class Net(nn.Module):
            def __init__(self, feature_encoder, symbol_net, feature_composer, feature_decoder, domain_model, action, action_activation):
                super(Net, self).__init__()
                self.feature_encoder = feature_encoder
                self.symbol_net = symbol_net
                self.feature_composer = feature_composer
                self.feature_decoder = feature_decoder
                self.domain_model = domain_model
                self.action = action
                self.action_activation = action_activation
                self.action_list = list(range(len(self.domain_model.actions)))

            def forward(self, x, a, i, g):
                """
                Args:
                    x: [B, T, *image_dims]    - a batch of T-step image traces
                    a: [B, T, *action_dims]   - a batch of T-step action traces
                    i: [B, *symbol_dims]      - a batch of initial states
                    g: [B, *symbol_dims]      - a batch of goal states
                """
                B, T, *image_dims = x.shape

                # -----------------------------------------------------------
                # 1. Symbol extraction for the T-step image trace
                # -----------------------------------------------------------
                # Encode each image -> extract symbol logits -> turn into probabilities
                features = dapply(x, self.feature_encoder)   # [B, T, feature_dim]
                z = torch.sigmoid(dapply(features, self.symbol_net)) # [B, T, symbol_dim]

                # -----------------------------------------------------------
                # 2. Reconstruction of the original trace images
                # -----------------------------------------------------------
                # Reconstruct the T actual images from z. There's no (T+1)-th image.
                y = self.feature_decoder(
                    self.feature_composer(z.view(B * T, -1))
                )
                y = y.view(B, T, *image_dims)  # [B, T, C, H, W]

                # -----------------------------------------------------------
                # 3. (Optional) predict actions, override the given actions
                # -----------------------------------------------------------
                z_ext = torch.cat([i.unsqueeze(1), z, g.unsqueeze(1)], dim=1)
                a_logit = self.action(torch.cat([z_ext[:, :-1, :], z_ext[:, 1:, :]], dim=-1))  # [B, T+1, action_dim]
                a = self.action_activation(a_logit)

                # -----------------------------------------------------------
                # 4. Forward inference (applier) across the T+1 pairs (including the initial state)
                # -----------------------------------------------------------
                z_ext = torch.cat([i.unsqueeze(1), z], dim=1)
                z_unsqueezed = z_ext.unsqueeze(2)
                all_precons, all_addeffs, all_deleffs = self.domain_model(self.action_list)  # [action_dim, symbol_dim]
                z_suc_aae = z_unsqueezed * (1 - all_deleffs) + (1 - z_unsqueezed) * all_addeffs # [B, T+1, action_dim, symbol_dim]

                # -----------------------------------------------------------
                # 5. applicability
                # -----------------------------------------------------------
                p_applicable = 1 - all_precons * (1 - z_unsqueezed)  # [B, T+1, action_dim, symbol_dim]

                # --------------------------------------------------------
                # 6. Return outputs as needed
                # --------------------------------------------------------
                return {
                    'x': x,
                    'z': z,
                    'y': y,
                    'a_logit': a_logit,
                    'a': a,
                    'z_suc_aae': z_suc_aae,
                    'p_applicable': p_applicable,
                    'all_precons': all_precons,
                }

        return Net(self.feature_encoder, self.symbol_net, self.feature_composer,
                   self.feature_decoder, self.domain_model, self.action, self.action_activation)

    def _register_metrics(self):
        # Register default metrics
        self.add_metric("state_acc", self.state_accuracy)
        self.add_metric("action_acc", self.action_accuracy)

    def loss_pred(self, outputs, goals):
        z_suc_aae = outputs['z_suc_aae']  # [B, T+1, action_dim, symbol_dim]
        a = outputs['a'].unsqueeze(2)  # [B, T+1, 1, action_dim]

        # For the prefix loss:
        #   Target for mse_loss is outputs['z'], which has shape [B, T, symbol_dim].
        #   We unsqueeze at dim=2 to get [B, T, 1, symbol_dim] and then expand it to match
        #   the shape of z_suc_aae[:, :-1, ...] which is [B, T, action_dim, symbol_dim].
        target_prefix = outputs['z'].unsqueeze(2).expand_as(z_suc_aae[:, :-1, ...])
        mse_prefix = F.mse_loss(z_suc_aae[:, :-1, ...], target_prefix, reduction='none')
        # a[:, :-1, :] has shape [B, T-1, 1, action_dim]
        # The batched matrix multiplication gives a result of shape [B, T-1, 1, symbol_dim]
        loss_prefix = (a[:, :-1, :] @ mse_prefix).sum(dim=(1, 2, 3))

        # For the last loss:
        #   Target is goals, with shape [B, symbol_dim]. Unsqueeze to [B, 1, symbol_dim] and
        #   expand to match z_suc_aae[:, -1, ...] which is [B, action_dim, symbol_dim].
        target_last = goals.unsqueeze(1).expand_as(z_suc_aae[:, -1, ...])
        mse_last = F.mse_loss(z_suc_aae[:, -1, ...], target_last, reduction='none')
        # a[:, -1, :] has shape [B, 1, action_dim], and the matrix multiplication yields [B, 1, symbol_dim]
        loss_last = (a[:, -1, :] @ mse_last).sum(dim=(1, 2))

        return loss_prefix + self.parameters["gamma"] * loss_last

    def loss_app(self, outputs):
        p_applicable = outputs['p_applicable']  # [B, T+1, action_dim, symbol_dim]
        a = outputs['a'].unsqueeze(2)  # [B, T+1, 1, action_dim]

        p_applicable_suffix = p_applicable[:, 1:, ...]  # [B, T, action_dim, symbol_dim]
        a_suffix = a[:, 1:, ...]  # [B, T, 1, action_dim]
        loss_suffix = (a_suffix @ F.mse_loss(p_applicable_suffix, torch.ones_like(p_applicable_suffix), reduction='none')).sum(dim=(1, 2, 3))

        loss_first = (a[:, 0, :] @ F.mse_loss(p_applicable[:, 0, :], torch.ones_like(p_applicable[:, 0, :]), reduction='none')).sum(dim=(1, 2))
        return self.parameters["gamma"] * loss_first + loss_suffix

    # def loss_pred(self, outputs, goals):
    #     preds = outputs['z_suc_aae']
    #     targets = torch.cat([outputs['z'][:, 1:, :], goals.unsqueeze(1)], dim=1)
    #     targets = targets.unsqueeze(2)
    #     match_prob = preds * targets + (1 - preds) * (1 - targets)
    #
    #     # Compute log probability and sum over proposition dimension.
    #     # Adding eps to avoid log(0).
    #     log_match = torch.log(match_prob + 1e-8)
    #     log_match_sum = log_match.sum(dim=-1)  # shape: [B, T, action_dim]
    #
    #     # Weight the log probabilities by the predicted action distribution and sum over actions
    #     weighted_log_prob = (outputs['a'] * log_match_sum).sum(dim=-1)  # shape: [B, T]
    #
    #     # Increase the weight for the final time step by gamma
    #     weighted_log_prob[:, -1] *= self.parameters["gamma"]
    #
    #     return -weighted_log_prob.mean()
    #
    # def loss_app(self, outputs):
    #     log_p_app = torch.log(outputs['p_applicable'] + 1e-8)
    #     log_p_app_sum = log_p_app.sum(dim=-1)  # shape: [B, T, action_dim]
    #
    #     weighted_log_prob = (outputs['a'] * log_p_app_sum).sum(dim=-1)
    #     return -weighted_log_prob.mean()

    def pseudo_label(self, outputs, goals):
        preds = outputs['z_suc_aae']
        targets = torch.cat([outputs['z'], goals.unsqueeze(1)], dim=1)
        targets = targets.unsqueeze(2)
        match_prob = preds * targets + (1 - preds) * (1 - targets)

        # Compute log probability and sum over proposition dimension.
        # Adding eps to avoid log(0).
        log_match = torch.log(match_prob + 1e-8)
        log_match_sum = log_match.sum(dim=-1)  # shape: [B, T+1, action_dim]

        log_p_app = torch.log(outputs['p_applicable'] + 1e-8)
        log_p_app_sum = log_p_app.sum(dim=-1)  # shape: [B, T+1, action_dim]

        # Sum the tensors and compute target indices
        logits = log_match_sum + log_p_app_sum
        pseudo_labels = logits.argmax(dim=-1)
        return pseudo_labels

    def loss_reconst(self, outputs):
        return F.mse_loss(outputs["y"], outputs["x"], reduction='mean')

    def loss_prior(self, outputs):
        all_precons = outputs['all_precons'] # [action_dim, symbol_dim]
        a = outputs['a'].unsqueeze(2) # [B, T+1, 1, action_dim]
        B, T_plus_1, _, _ = a.shape
        return  (a @ F.mse_loss(all_precons, torch.ones_like(all_precons), reduction='none')).sum() / (B * T_plus_1)

    def loss(self, outputs, targets, indices=[], mip_pseudo_labels=None):
        B, T, _ = outputs["z"].shape

        # Compute each loss component
        loss_prior_val = self.loss_prior(outputs)
        goals = targets[0][:, -1, :]
        loss_pred_val = self.loss_pred(outputs, goals) # (B, ), unnormalized
        loss_app_val = self.loss_app(outputs) # (B, ), unnormalized
        loss_reconst_val = self.loss_reconst(outputs)
        loss_pred_val = loss_pred_val.sum() / (B * (T+1))
        loss_app_val = loss_app_val.sum() / (B * (T+1))

        # Assemble final loss with weights
        loss_dl_relaxed = (self.parameters["lambda"] * loss_prior_val +
                      self.parameters["beta_pred"] * loss_pred_val +
                      self.parameters["beta_app"] * loss_app_val +
                      self.parameters["beta_reconst"] * loss_reconst_val)


        preds_a = outputs['a_logit'].view(-1, self.adim())
        pseudo_labels_a = self.pseudo_label(outputs, goals)
        weight_mask_a = torch.ones_like(pseudo_labels_a, dtype=torch.float32)

        trace_labels = mip_pseudo_labels.traces if mip_pseudo_labels is not None else {}
        loss_pseudo_s = 0
        # loss_pseudo_a = 0

        for batch_id, idx in enumerate(indices):
            idx = idx.item()
            if idx in trace_labels:
                weight, state_label, action_label = trace_labels[idx]
                if 'state' in self.parameters["MIP_to_DL"]:
                    state_label = state_label.to(outputs["z"].device)
                    loss_pseudo_s += F.binary_cross_entropy(outputs["z"][batch_id], state_label) * weight
                if 'action' in self.parameters["MIP_to_DL"]:
                    action_label = action_label.to(outputs["a_logit"].device)
                    # loss_pseudo_a += F.cross_entropy(outputs["a_logit"][batch_id], action_label) * weight
                    pseudo_labels_a[batch_id] = action_label
                    weight_mask_a[batch_id] = weight
                trace_labels[idx] = (weight * self.parameters["pseudo_weight_decay"], state_label, action_label)

        if len(trace_labels) > 0:
            loss_pseudo_s /= len(trace_labels)
            # loss_pseudo_a /= len(trace_labels)
        loss_pseudo_a = (weight_mask_a.view(-1) * F.cross_entropy(preds_a, pseudo_labels_a.reshape(-1), reduction='none')).mean()

        model_label = mip_pseudo_labels.model if mip_pseudo_labels is not None else None
        loss_pseudo_m = 0
        if 'model' in self.parameters["MIP_to_DL"] and model_label is not None:
            len_model = 0
            for schema in self.domain_model.action_schemas:
                schema_prams = self.domain_model.activation(schema())
                target = model_label[schema.name].to(schema_prams.device)
                loss_pseudo_m += F.cross_entropy(schema_prams, target, reduction='sum')
                len_model += schema.randn.shape[0]
            loss_pseudo_m /= len_model

        total_loss = loss_dl_relaxed + loss_pseudo_a + loss_pseudo_s + loss_pseudo_m

        # Option 1: Return a dictionary with all losses
        return {
            'total_loss': total_loss,
            'loss_pred': loss_pred_val,
            'loss_app': loss_app_val,
            'loss_reconst': loss_reconst_val,
            'loss_prior': loss_prior_val,
        }

    def state_accuracy(self, outputs, targets):
        z = outputs['z']
        pred_states = (z >= 0.5).float()
        target_states = targets[0][:, 1:-1, :]
        correct = (pred_states == target_states).float()
        return correct.mean().item() # Average over batch, step, and propositions

    def action_accuracy(self, outputs, targets, action_mapping=None):
        a = outputs['a']
        pred_actions = a.argmax(dim=-1)
        if action_mapping is not None:
            pred_actions = torch.tensor(action_mapping, device=pred_actions.device)[pred_actions]
        target_actions = targets[1].argmax(dim=-1)
        correct = (pred_actions == target_actions).float()
        return correct.mean().item() # Average over batch and step