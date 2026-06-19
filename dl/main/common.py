import argparse
import sys
import dl.model
from dl.util.tuning import simple_genetic_search, parameters, nn_task, grid_search
from dl.util import curry
import dl.util.stacktrace

# --- Preliminary parsing: only parse the first argument ('mode') ---
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("mode", help="Mode string (e.g., learn_plot_dump or resume_plot_dump)")
# Parse just the mode from sys.argv[1:2]
pre_args, _ = pre_parser.parse_known_args(sys.argv[1:2])

# --- Build the main parser ---
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    "mode",
    help=(
        "A string which contains mode substrings."
        "\nRecognized modes are:"
        "\n" 
        "\n   learn     : perform the training with a hyperparameter tuner. Results are stored in samples/[experiment]/logs/[hyperparameter]."
        "\n               If 'learn' is not specified, it attempts to load the stored weights."
        "\n   plot      : produce visualizations"
        "\n   dump      : dump the csv files necessary for producing the PDDL models"
        "\n   summary   : perform extensive performance evaluations and collect the statistics, store the result in performance.json"
        "\n   debug     : debug training limited to epoch=2, batch_size=100. dataset is truncated to 200 samples"
        "\n   reproduce : train the best hyperparameter so far three times with different random seeds. store the best results."
        "\n   iterate   : iterate plot/dump/summary commands above over all hyperparmeters that are already trained and stored in logs/ directory."
        "\n"
        "\nFor example, learn_plot_dump contains 'learn', 'plot', 'dump' mode."
        "\nThe separater does not matter because its presense is tested by python's `in` directive, i.e., `if 'learn' in mode:` ."
        "\nTherefore, learnplotdump also works."))

# Conditionally add the resume_hash positional argument if mode indicates resume.
if "resume" in pre_args.mode:
    parser.add_argument(
        "resume_hash",
        help="Hash value indicating where to resume from."
    )

subparsers = parser.add_subparsers(
    title="subcommand",
    metavar="subcommand",
    required=True,
    description=(
        "\nA string which matches the name of one of the dataset functions in dl.main module."
        "\n"
        "\nEach task has a different set of parameters, e.g.,"
        "\n'puzzle' has 'type', 'width', 'height' where 'type' should be one of 'mnist', 'spider', 'mandrill', 'lenna',"
        "\nwhile 'lightsout' has 'type' being either 'digital' and 'twisted', and 'size' being an integer."
        "\nSee subcommand help."))

def add_common_arguments(subparser,task,objs=False):
    subparser.set_defaults(task=task)
    subparser.add_argument(
        "num_examples",
        default=5000,
        type=int,
        help=(
            "\nNumber of data points to use. 90%% of this number is used for training, and 5%% each for validation and testing."
            "\nIt is assumed that the user has already generated a dataset archive in dl/data/,"
            "\nwhich contains a larger number of data points using the setup-dataset script provided in the root of the repository."))
    subparser.add_argument(
        "aeclass",
        help=
        "A string which matches the name of the model class available in dl.model module.\n"+
        "It must be one of:\n"+
        "\n".join([ " "*4+name for name, cls in vars(dl.model).items()
                    if type(cls) is type and \
                    issubclass(cls, dl.network.Network) and \
                    cls is not dl.network.Network
                ])
    )
    if objs:
        subparser.add_argument("location_representation",
                               nargs='?',
                               choices=["bbox","coord","binary","sinusoidal","anchor"],
                               default="coord",
                               help="A string which specifies how to convert/encode the location in the dataset. See documentations for normalize_transitions_objects")
        subparser.add_argument("randomize_location",
                               nargs='?',
                               type=bool,
                               default=False,
                               help="A boolean which specifies whether we randomly translate the environment globally. See documentations for normalize_transitions_objects")
    subparser.add_argument("comment",
                           nargs='?',
                           default="",
                           help="A string which is appended to the directory name to label each experiment.")
    return

def main(parameters={}):
    dl.util.tuning.parameters.update(parameters)
    global args, sae_path
    args = parser.parse_args()
    task = args.task
    delattr(args,"task")
    print(vars(args))
    dl.util.tuning.parameters.update(vars(args))
    if 'resume' in args.mode:
        sae_path = "_".join(sys.argv[3:])
    else:
        sae_path = "_".join(sys.argv[2:])
    try:
        task(args)
    except:
        dl.util.stacktrace.format()


def train_val_test_split(x):
    train = x[:int(len(x)*0.9)]
    val   = x[int(len(x)*0.9):int(len(x)*0.95)]
    test  = x[int(len(x)*0.95):]
    return train, val, test

def show_summary(ae, train, test):
    if 'summary' in args.mode:
        ae.summary()
        ae.report(train, test_data = test)

def plot_autoencoding_image(ae, transitions, label):
    if 'plot' not in args.mode:
        return

    if hasattr(ae, "plot_transitions"):
        transitions = transitions[:3]
        ae.plot_transitions(transitions, ae.local(f"transitions_{label}"),verbose=True)
    else:
        transitions = transitions[:3]
        states = transitions.reshape((-1, *transitions.shape[2:]))
        ae.plot(states, ae.local(f"states_{label}"),verbose=True)

    return

def dump_actions(ae, transitions, name = "actions.csv", repeat=1):
    if 'dump' not in args.mode:
        return
    print(ae.local(name))
    ae.dump_actions(transitions, batch_size = 1000)


def run(path, img_traces, state_traces, action_traces, extra=None):
    train_img_traces, val_img_traces, test_img_traces = train_val_test_split(img_traces)
    train_state_traces, val_state_traces, _ = train_val_test_split(state_traces)
    train_action_traces, val_action_traces, _ = train_val_test_split(action_traces)
    train_all = (train_img_traces, train_state_traces, train_action_traces)
    val_all = (val_img_traces, val_state_traces, val_action_traces)


    def postprocess(ae):
        show_summary(ae, train_img_traces, test_img_traces)
        plot_autoencoding_image(ae, train_img_traces, "train")
        plot_autoencoding_image(ae, test_img_traces, "test")
        dump_actions(ae, img_traces)
        return ae


    def report(net,eval):
        try:
            postprocess(net)
            if extra:
                extra(net)
        except:
            dl.util.stacktrace.format()
        return


    if 'learn' in args.mode:
        grid_search(
            curry(nn_task,
                  dl.model.get(parameters["aeclass"]),
                  path,
                  train_all,
                  val_all),  # regular training task
            parameters,
            path,
            limit=100,
            report=report,
        )
    elif 'resume' in args.mode:
        grid_search(
            curry(nn_task,
                  dl.model.get(parameters["aeclass"]),
                  path,
                  train_all,
                  val_all),
            parameters,
            path,
            limit=100,
            report=report,
            resume_hash=args.resume_hash,
        )
