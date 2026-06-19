import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from convertor.convertor import Convertor
from convertor.selector import TraceSelector
from util.model_perm import model_permutation
from .util import NpEncoder
from .util.dataset import TraceDataset, ColumnPermute, ItemPermute, RoomBallPermute, BlocksworldPilePermute, \
    TraceResize
from .util.plot import plot_grid
from tqdm import tqdm
from itertools import permutations, product


class Network:
    """Base class for various neural networks including GANs, AEs and Classifiers.
    Provides an interface for saving / loading the trained weights as well as hyperparameters.

    Each instance corresponds to a directory (specified in the `path` variable in the initialization),
    which contains the learned weights as well as several json files containing additional
    information such as hyperparameters.
    """

    def __init__(self, path, parameters={}):
        self.built = False
        self.built_aux = False
        self.loaded = False
        self.parameters = parameters
        self.past_parameters = []
        self.metrics = {}
        self.epoch = 0
        if not os.path.exists(path):
            os.makedirs(path)
        if "hash" in parameters:
            self.path = os.path.join(path, "logs", str(self.parameters["hash"]))
        else:
            # During resume, we provide the old hash in path and update the path after loading the network
            self.path = path

    def build(self, *args, **kwargs):
        """An interface for building a network. Input-shape: list of dimensions.
        Users should not overload this method; Define _build_around() for each subclass instead.
        This function calls _build bottom-up from the least specialized class.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        if self.built:
            print("Avoided building {} twice.".format(self))
            return
        print("Building networks")
        self._build_around(*args, **kwargs)
        self.built = True
        print("Network built")
        return self

    def _build_around(self, *args, **kwargs):
        """An interface for building a network.
        This function is called by build() only when the network is not build yet.
        Users may define a method for each subclass for adding a new build-time feature.
        Each method should call the _build_around() method of the superclass in turn.
        Users are not expected to call this method directly. Call build() instead.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        return self._build_primary(*args, **kwargs)

    def _build_primary(self, *args, **kwargs):
        pass

    def build_aux(self, *args, **kwargs):
        """An interface for building an additional network not required for training.
        To be used after the training.
        Input-shape: list of dimensions.
        Users should not overload this method; Define _build_around() for each subclass instead.
        This function calls _build bottom-up from the least specialized class.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        if self.built_aux:
            print("Avoided building {} twice.".format(self))
            return
        print("Building auxiliary networks")
        self._build_aux_around(*args, **kwargs)
        self.built_aux = True
        print("Auxiliary network built")
        return self

    def _build_aux_around(self, *args, **kwargs):
        """An interface for building an additional network not required for training.
        To be used after the training.
        Input-shape: list of dimensions.
        This function is called by build_aux() only when the network is not build yet.
        Users may define a method for each subclass for adding a new build-time feature.
        Each method should call the _build_aux_around() method of the superclass in turn.
        Users are not expected to call this method directly. Call build() instead.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        return self._build_aux_primary(*args, **kwargs)

    def _build_aux_primary(self, *args, **kwargs):
        pass

    def local(self, path=""):
        """A convenient method for converting a relative path to the learned result directory
        into a full path."""
        return os.path.join(self.path, path)

    def save(self, path=""):
        """An interface for saving a network.
        Users should not overload this method; Define _save() for each subclass instead.
        This function calls _save bottom-up from the least specialized class.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        print("Saving the network to {}".format(self.local(path)))
        os.makedirs(self.local(path), exist_ok=True)
        self._save(path)
        print("Network saved")
        return self

    def _save(self, path=""):
        """An interface for saving a network.
        Users may define a method for each subclass for adding a new save-time feature.
        Each method should call the _save() method of the superclass in turn.
        Users are not expected to call this method directly. Call save() instead.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        torch.save(self.net.state_dict(), self.local(os.path.join(path, f"net.pth")))

        torch.save(self.optimizer.state_dict(), self.local(os.path.join(path, f"opt.pth")))

        with open(self.local(os.path.join(path, "aux.json")), "w") as f:
            json.dump(
                {
                    "parameters": self.parameters,
                    'past_parameters': self.past_parameters,
                    "class": self.__class__.__name__,
                    "epoch": self.epoch,
                    "input_shape": self.input_shape,
                },
                f,
                skipkeys=True,
                cls=NpEncoder,
                indent=2,
            )

    def load(self, allow_failure=False, path=""):
        """An interface for loading a network.
        Users should not overload this method; Define _load() for each subclass instead.
        This function calls _load bottom-up from the least specialized class.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        if self.loaded:
            print("Avoided loading {} twice.".format(self))
            return

        if allow_failure:
            try:
                print("Loading networks from {} (with failure allowed)".format(self.local(path)))
                self._load(path)
                self.loaded = True
                print("Network loaded")
            except Exception as e:
                print("Exception {} during load(), ignored.".format(e))
        else:
            print("Loading networks from {} (with failure not allowed)".format(self.local(path)))
            self._load(path)
            self.loaded = True
            print("Network loaded")
        return self

    def _load(self, path=""):
        """An interface for loading a network.
        Users may define a method for each subclass for adding a new load-time feature.
        Each method should call the _load() method of the superclass in turn.
        Users are not expected to call this method directly. Call load() instead.
        Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods.
        """
        with open(self.local(os.path.join(path, "aux.json")), "r") as f:
            data = json.load(f)
            checkpoint_params = data["parameters"]
            self.past_parameters = data.get("past_parameters", [])
            self.past_parameters.append(checkpoint_params)
            # Merge the checkpoint parameters with the new ones
            merged_params = checkpoint_params.copy()
            merged_params.update(self.parameters)  # New values override the old ones
            self.parameters = merged_params
            self.build(torch.Size(data["input_shape"]))
            self.built = True
            self.build_aux(torch.Size(data["input_shape"]))
            self.built_aux = True
            self.epoch = data["epoch"]

        # We load the network to return to the exact status when we saved it, including the device
        device = torch.device(self.parameters["device"])
        state_dict = torch.load(self.local(os.path.join(path, f"net.pth")), map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.optimizer = getattr(optim, self.parameters["optimizer"])(self.net.parameters(), lr=self.parameters["lr"])
        self.optimizer.load_state_dict(torch.load(self.local(os.path.join(path, f"opt.pth")), map_location=device))
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def add_metric(self, name, loss=None, force=False):
        """loss should be a function that accepts net's outputs and targets (optional)"""
        if name in self.metrics and not force:
            print("Metric '", name, "' already registered, not added.")
            return
        if loss is None:
            loss = eval(name)
        self.metrics[name] = loss
        return

    def _run_training(self, train_data, val_data):
        # Create datasets from the provided training and validation data.
        if self.parameters["domain"] == "blocks":
            if "pddlgym" in self.parameters["data_loc"]:
                train_dataset = TraceDataset(train_data, transforms=BlocksworldPilePermute())
            elif "ogbench" in self.parameters["data_loc"]:
                train_dataset = TraceDataset(train_data)
            else:
                train_dataset = TraceDataset(train_data, transforms=ColumnPermute())
        elif self.parameters["domain"] == "gripper":
            train_dataset = TraceDataset(train_data, transforms=RoomBallPermute())
        elif self.parameters["domain"] == "logistics":
            train_dataset = TraceDataset(train_data, transforms=ItemPermute())
        else:
            train_dataset = TraceDataset(train_data)

        batch_size = self.parameters["batch_size"]
        batch_size = min(batch_size, len(train_dataset))
        # Create DataLoaders for training and validation.
        train_loader = DataLoader(
            train_dataset, batch_size,
            shuffle=True, num_workers=64, prefetch_factor=8,
            persistent_workers=True
        )
        device = torch.device(self.parameters["device"])

        # Create a progress bar starting at the correct epoch.
        bar = tqdm(range(self.parameters["epoch"]))
        logs = {}
        self.file_writer = SummaryWriter(self.path)

        convertor = Convertor(self.net.domain_model, self.parameters["DL_to_MIP"], self.parameters["domain"], self.parameters["cp_type"])  # just for loading domain
        trace_selector = TraceSelector(self.parameters["mip_traces"])

        best_train_loss = float("inf")
        for epoch in bar:
            self.net.train()
            running_log = {}  # to accumulate training losses
            n_batches = 0
            trace_selector.clear()

            for img_traces, state_traces, action_traces, indices in train_loader:
                n_batches += 1
                img_traces, state_traces, action_traces = img_traces.to(device), state_traces.to(
                    device), action_traces.to(device)
                inits = state_traces[:, 0, :]
                goals = state_traces[:, -1, :]
                targets = [state_traces, action_traces]
                self.optimizer.zero_grad()
                outputs = self.net(img_traces, action_traces, inits, goals)
                loss_dict = self.loss(outputs, targets, indices, convertor.pseudo_labels)
                loss = loss_dict['total_loss']
                loss.backward()
                self.optimizer.step()

                if epoch >= self.parameters["pre_mip_epoch"] and epoch % self.parameters["mip_interval"] == 0:
                    trace_selector.update(indices, outputs['z'], outputs['a'], inits, goals)

                # Accumulate each loss component
                for key, value in loss_dict.items():
                    running_log[key] = running_log.get(key, 0.0) + value.item()
                for name, metric in self.metrics.items():
                    running_log[name] = running_log.get(name, 0.0) + metric(outputs, targets)

            # Compute average losses over the epoch
            avg_log = {k: running_log[k] / n_batches for k in running_log}
            # Update the progress bar description with running averages
            bar.set_description("  ".join([f"{k} {v:8.3g}" for k, v in sorted(avg_log.items())]))
            # Log the averaged training losses
            for k, v in avg_log.items():
                self.file_writer.add_scalar(k, v, epoch)

            if avg_log["total_loss"] < best_train_loss:
                best_train_loss = avg_log["total_loss"]
                torch.save(self.net.state_dict(), self.local("best_model.pth"))
                self.best_epoch = epoch

            # Perform validation every 10 epochs
            # Compute pseudo_label every 10 epochs
            if epoch % 9 == 0:
                avg_val_log = self.evaluate(val_data)
                bar.set_description("  ".join([f"{k} {v:8.3g}" for k, v in sorted(avg_val_log.items())]))
                self.save()
                self.dump_actions()

            # # Compute pseudo labels
            if epoch >= self.parameters["pre_mip_epoch"] and epoch % self.parameters["mip_interval"] == 0:
                mip_gt_dist = convertor.run_fixer(trace_selector, self.parameters["mip_time_limit"])
                if mip_gt_dist is not None:
                    self.file_writer.add_scalar("mip_gt_dist", mip_gt_dist, epoch)

            torch.cuda.empty_cache()
            self.epoch += epoch

        self.file_writer.close()
        self.save_at_end(logs)

    def train(self, train_data, val_data):
        input_shape = train_data[0].shape[1:]
        self.input_shape = input_shape
        self.build(input_shape)
        self.build_aux(input_shape)
        device = torch.device(self.parameters["device"])
        self.net.to(device)
        # Create a new optimizer (it will be reinitialized here)
        self.optimizer = getattr(optim, self.parameters["optimizer"])(
            self.net.parameters(), lr=self.parameters["lr"]
        )
        self._run_training(train_data, val_data)

    def resume(self, train_data, val_data, parameters):
        self.parameters = parameters
        self.load()
        # Update to a new subfolder when resume training
        self.path = self.local(str(self.parameters["hash"]))
        # self.perturb_model(self.net.domain_model, self.parameters["perturbation_scale"])
        self._run_training(train_data, val_data)

    def save_at_end(self, logs):
        self.save()
        self.loaded = True

    def compute_permutation(self):
        convertor = Convertor(self.net.domain_model, self.parameters["DL_to_MIP"], self.parameters["domain"], self.parameters["cp_type"])
        domain = convertor.domain
        rosame_model = convertor.translator.trans_rosame()

        name_mapping, args_mapping, highest_agree, lowest_error = model_permutation(domain, rosame_model, convertor.gt_am)
        original_actions = list(self.net.domain_model.actions.keys())
        action_mapping = []
        for action in original_actions:
            name = action.split(" ")[0]
            params = action.split(" ")[1:]
            new_name = name_mapping[name]
            new_params = [params[i - 1] for i in args_mapping[name]]  # Convert to 0-based index
            new_action = f"{new_name} {' '.join(new_params)}"
            action_mapping.append(original_actions.index(new_action))
        return name_mapping, args_mapping, highest_agree, action_mapping, lowest_error

    @torch.no_grad()
    def evaluate(self, val_data, **kwargs):
        name_mapping, args_mapping, highest_agree, action_mapping, lowest_error = self.compute_permutation()

        if "pddlgym" in self.parameters["data_loc"]:
            val_dataset = TraceDataset(val_data, transforms=TraceResize(size=64))
        else:
            val_dataset = TraceDataset(val_data)
        batch_size = kwargs.get("batch_size", self.parameters["batch_size"])
        batch_size = min(batch_size, len(val_dataset))
        val_loader = DataLoader(
            val_dataset, batch_size,
            shuffle=False, num_workers=4, prefetch_factor=2,
            persistent_workers=False
        )
        # Move the network to the desired device.
        device = torch.device(self.parameters["device"])
        self.net.to(device)
        self.net.eval()

        val_running_log = {}
        n_val_batches = 0
        with torch.no_grad():
            for img_traces, state_traces, action_traces, _ in val_loader:
                n_val_batches += 1
                img_traces, state_traces, action_traces = img_traces.to(device), state_traces.to(
                    device), action_traces.to(device)
                inits = state_traces[:, 0, :]
                goals = state_traces[:, -1, :]
                targets = [state_traces, action_traces]
                outputs = self.net(img_traces, action_traces, inits, goals)
                loss_dict = self.loss(outputs, targets)
                for key, value in loss_dict.items():
                    val_running_log[key] = val_running_log.get(key, 0.0) + value.item()
                for name, metric in self.metrics.items():
                    if name == "action_acc":
                        val_running_log[name] = val_running_log.get(name, 0.0) + metric(outputs, targets, action_mapping)
                    else:
                        val_running_log[name] = val_running_log.get(name, 0.0) + metric(outputs, targets)
            avg_val_log = {f"val_{k}": val_running_log[k] / n_val_batches for k in val_running_log}
        print(f"Highest agree: {highest_agree}")
        print(f"Lowest error: {lowest_error}")
        print(avg_val_log)
        return avg_val_log


    @torch.no_grad()
    def report(self, train_data,
               batch_size=1000,
               test_data=None,
               train_data_to=None,
               test_data_to=None):
        pass

    def save_array(self, name, data):
        print("Saving to", self.local(name))
        with open(self.local(name), "wb") as f:
            np.savetxt(f, data, "%d")

    def _plot(self, path, data, epoch=None, orientation="columns"):
        """
        Arranges images in rows or columns for plotting, saving, and logging to TensorBoard.

        Args:
            path (str): The file path to save the plot.
            data (list): A list of rows or columns, where each element is a list of tensors representing images.
            epoch (int, optional): Current epoch for TensorBoard logging. Defaults to None.
            orientation (str): "columns" to treat `data` as columns, "rows" to treat `data` as rows. Defaults to "columns".
        """
        if orientation not in {"columns", "rows"}:
            raise ValueError(f"Invalid orientation: {orientation}. Use 'columns' or 'rows'.")

        data = list(data)

        if epoch is None:
            if orientation == "columns":
                # Flatten data column-wise into rows for plotting
                rows = []
                for seq in zip(*data):  # Arrange into rows from columns
                    rows.extend(seq)
            elif orientation == "rows":
                # Flatten data row-wise for plotting
                rows = [item for row in data for item in row]
            plot_grid(rows, w=len(data) if orientation == "columns" else len(data[0]), path=path, verbose=True)
        else:
            # Log images to TensorBoard every 10 epochs
            if (epoch % 10) != 0:
                return

            _, basename = os.path.split(path)

            if orientation == "columns":
                columns = data
            elif orientation == "rows":
                # Transpose rows into columns for consistent internal handling
                columns = list(zip(*data))  # Convert rows to columns

            # Iterate over columns and save/log images
            for i, col in enumerate(columns):
                # Convert PyTorch tensor (C, H, W) to (H, W, C) for saving
                col = col.permute(0, 2, 3, 1).cpu().numpy()  # Change to (batch, H, W, C)
                col = np.clip(col, 0.0, 1.0)  # Ensure the pixel values are in [0, 1]

                for k, image in enumerate(col):
                    # Log the image to TensorBoard
                    self.file_writer.add_image(f"{basename}/col{i}_img{k}", torch.tensor(image).permute(2, 0, 1), epoch)

            # Ensure all data is written to TensorBoard
            self.file_writer.flush()