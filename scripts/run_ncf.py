import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt

from src.utils.hparam_search import param_comb
from src.data.ncf import NCFDataset
from src.models.ncf import NCFModel
from src.metrics.evaluator import collect_user_predictions, compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(MODEL_TYPE, PLOT, TUNE, VERBOSE):

    # ----------------------------------------------------------------------------------
    # ------ Global Parameters
    # ----------------------------------------------------------------------------------
    CONFIG = load_config(f"src/config/{MODEL_TYPE.lower()}.yml")

    DEVICE = CONFIG["info"]["device"]
    DATA_DIR = CONFIG["info"]["data_dir"]
    RANDOM_SEED = CONFIG["info"]["random_seed"]
    MODEL_DIR = CONFIG["info"]["model_dir"]

    if TUNE:
        MODEL_CONFIG = CONFIG["hparam_tune"]
    else:
        MODEL_CONFIG = CONFIG["hparam_optim"]

    # ----------------------------------------------------------------------------------
    # ------ Data
    # ----------------------------------------------------------------------------------

    train_file = MODEL_CONFIG["dataset_names"][0]
    test_file = MODEL_CONFIG["dataset_names"][1]
    full_file = MODEL_CONFIG["dataset_names"][2]

    data = NCFDataset(
        train_file_path=f"{DATA_DIR}/{train_file}.parquet",
        test_file_path=f"{DATA_DIR}/{test_file}.parquet",
        full_file_path=f"{DATA_DIR}/{full_file}.parquet",
    )

    # ----------------------------------------------------------------------------------
    # ------ MAIN: Tune OR evaluate
    # ----------------------------------------------------------------------------------

    hparam_combinations = param_comb(config=MODEL_CONFIG, is_tune=TUNE)

    for hparams in hparam_combinations:
        # MERGE: Combine fixed settings with current trial settings
        # This ensures 'step_size' and 'gamma' are available

        print(f"vebose: {hparams}")

        # ------------------------------------------------------------------------------
        # ------ Model Related Parameters
        # ------------------------------------------------------------------------------

        EPOCHS = hparams["epochs"]
        BATCH_SIZE = hparams["batch_size"]
        N_WORKERS = hparams["n_workers"]

        STEP_SIZE = hparams["step_size"]
        GAMMA = hparams["gamma"]

        LEARNING_RATE = hparams["learning_rate"]
        LAYERS = hparams["layers"]
        DROPOUT = hparams["dropout"]

        LOG_EVERY = hparams["log_every"]
        THRESHOLD = hparams["threshold"]

        # ------------------------------------------------------------------------------
        # ------ Train
        # ------------------------------------------------------------------------------

        ncf_model = NCFModel(
            n_users=data.n_users,
            n_items=data.n_items,
            epochs=EPOCHS,
            step_size=STEP_SIZE,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE,
            log_every=LOG_EVERY,
            threshold=THRESHOLD,
            layers=LAYERS,
            dropout=DROPOUT,
            model_type=MODEL_TYPE,
        )

        all_losses_list = ncf_model.train(
            data.train_loader(batch_size=BATCH_SIZE, n_workers=N_WORKERS, shuffle=True)
        )

        # Plot Loss
        if PLOT:
            plt.figure()
            plt.plot(all_losses_list)
            plt.show()

        print("Train Loss: {}\n".format(np.round(all_losses_list[-1], 4)))

        # ------------------------------------------------------------------------------
        # ------ Evaluation (Test set)
        # ------------------------------------------------------------------------------

        test_loader = data.test_loader(
            batch_size=BATCH_SIZE, n_workers=N_WORKERS, shuffle=False
        )

        test_loss = ncf_model.evaluate(test_loader)
        print("Test Loss: {}\n".format(np.round(test_loss, 4)))

        if not TUNE:

            K = [1, 3, 5, 10, 20, 50, 100]
            metrics_to_compute = ["precision", "recall", "hit_rate", "ndcg"]

            user_pred_true = collect_user_predictions(
                test_loader, ncf_model.model, DEVICE
            )

            for k in K:

                if "rmse" in metrics_to_compute and k != K[0]:
                    metrics_to_compute.remove("rmse")

                print(metrics_to_compute)

                metrics = compute_metrics(
                    user_pred_true=user_pred_true,
                    metrics=metrics_to_compute,
                    k=k,
                    threshold=THRESHOLD,
                )

                for metric in metrics_to_compute:
                    if metric != "rmse":
                        print(
                            "{} @ {}: {}\n".format(
                                metric.upper(), k, np.round(metrics[metric], 4)
                            )
                        )
                    else:
                        print(
                            "{}: {}\n".format(
                                metric.upper(), np.round(metrics[metric], 4)
                            )
                        )


if __name__ == "__main__":
    # setup Argument Parser
    parser = argparse.ArgumentParser(description="Train NCF models")

    # define the --model argument
    parser.add_argument(
        "--model",
        type=str,
        default="DeepNCF",
        choices=["SimpleNCF", "DeepNCF"],
        help="Model type to use: SimpleNCF or DeepNCF",
    )
    parser.add_argument(
        "--plot",
        action="store_true",  # Sets value to True if argument is present
        help="Enable plotting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",  # Sets value to True if argument is present
        help="Enable verbose",
    )
    parser.add_argument(
        "--tune",
        action="store_true",  # Sets value to True if argument is present
        help="Run hyperparameter tuning",
    )
    args = parser.parse_args()

    main(MODEL_TYPE=args.model, PLOT=args.plot, TUNE=args.tune, VERBOSE=args.verbose)
