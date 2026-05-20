# Libraries
import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join("results", ".cache", "matplotlib"))

import lightning as L
import matplotlib.pyplot as plt
import torch
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.data import DataLoader

from training.train import CurriculumLearning, DataModule, LightningTrainer, TestLossLogger
from utils.dataset import (
    create_model_dataset,
    get_temporal_test_dataset_parameters,
    to_temporal_dataset,
)
from utils.load import read_config
from utils.miscellaneous import (
    SpatialAnalysis,
    get_model,
    get_numerical_times,
    get_speed_up,
    normalize_config,
)
from utils.visualization import PlotRollout

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def get_aim_logger(cfg):
    aim_cfg = cfg.get("aim", {}) or {}
    enabled = aim_cfg.get("enabled", os.getenv("AIM_ENABLED", "true"))
    if isinstance(enabled, str):
        enabled = enabled.lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return False

    logger_kwargs = {
        "repo": aim_cfg.get("repo") or os.getenv("AIM_REPO") or "results/aim",
        "experiment": aim_cfg.get("experiment")
        or os.getenv("AIM_EXPERIMENT")
        or "mSWE-GNN",
    }

    run_name = aim_cfg.get("run_name") or os.getenv("AIM_RUN_NAME")
    if run_name:
        logger_kwargs["run_name"] = run_name

    return AimLogger(**logger_kwargs)


def main(config, logger):
    L.seed_everything(config.models["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_parameters = config.dataset_parameters
    scalers = config.scalers
    selected_node_features = config.selected_node_features
    selected_edge_features = config.selected_edge_features

    train_dataset, val_dataset, test_dataset, scalers = create_model_dataset(
        scalers=scalers,
        device=device,
        **dataset_parameters,
        **selected_node_features,
        **selected_edge_features,
    )

    temporal_dataset_parameters = config.temporal_dataset_parameters
    temporal_train_dataset = to_temporal_dataset(
        train_dataset, **temporal_dataset_parameters
    )

    print("Number of training simulations:\t", len(train_dataset))
    print("Number of training samples:\t", len(temporal_train_dataset))
    print("Number of node features:\t", temporal_train_dataset[0].x.shape[-1])
    print("Number of rollout steps:\t", temporal_train_dataset[0].y.shape[-1])

    num_node_features, num_edge_features = (
        temporal_train_dataset[0].x.size(-1),
        temporal_train_dataset[0].edge_attr.size(-1),
    )
    num_nodes, num_edges = (
        temporal_train_dataset[0].x.size(0),
        temporal_train_dataset[0].edge_attr.size(0),
    )

    previous_t = temporal_dataset_parameters["previous_t"]
    test_size = len(test_dataset)
    test_dataset_name = dataset_parameters["test_dataset_name"]
    temporal_res = dataset_parameters["temporal_res"]
    max_rollout_steps = temporal_dataset_parameters["rollout_steps"]

    print("Temporal resolution:\t", temporal_res, "min")

    model_parameters = config.models
    model_type = model_parameters.pop("model_type")

    if model_type == "MSGNN":
        num_scales = train_dataset[0].mesh.num_meshes
        model_parameters["num_scales"] = num_scales

    model = get_model(model_type)(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        previous_t=previous_t,
        device=device,
        **model_parameters,
    ).to(device)

    trainer_options = config.trainer_options
    type_loss = trainer_options["type_loss"]
    lr_info = config["lr_info"]

    # info for testing dataset
    temporal_test_dataset_parameters = get_temporal_test_dataset_parameters(
        config, temporal_dataset_parameters
    )

    temporal_val_dataset = to_temporal_dataset(
        val_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )

    temporal_test_dataset = to_temporal_dataset(
        test_dataset, rollout_steps=-1, **temporal_test_dataset_parameters
    )

    test_dataloader = DataLoader(
        temporal_test_dataset, batch_size=len(temporal_test_dataset), shuffle=False
    )

    plmodule = LightningTrainer(
        model, lr_info, trainer_options, temporal_test_dataset_parameters
    )

    pldatamodule = DataModule(
        temporal_train_dataset,
        temporal_val_dataset,
        batch_size=trainer_options["batch_size"],
    )

    # Number of parameters
    total_parameteres = sum(p.numel() for p in model.parameters())
    if logger:
        logger.log_metrics({"total parameters": total_parameteres}, step=0)

    # Training
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_logs/models", monitor="val_loss", mode="min", save_top_k=1
    )
    curriculum_callback = CurriculumLearning(max_rollout_steps, patience=5)
    test_loss_logger = TestLossLogger(test_dataloader)
    early_stopping = EarlyStopping(
        "val_CSI_005", mode="max", patience=trainer_options["patience"]
    )
    # Load trained model
    plmodule_kwargs = {
        "model": model,
        "lr_info": lr_info,
        "trainer_options": trainer_options,
        "temporal_test_dataset_parameters": temporal_test_dataset_parameters,
    }

    if "saved_model" in config:
        plmodule = LightningTrainer.load_from_checkpoint(
            config["saved_model"], map_location=device, **plmodule_kwargs
        )
        model = plmodule.model.to(device)

    # Define trainer
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=trainer_options["max_epochs"],
        gradient_clip_val=1,
        precision="16-mixed",
        enable_progress_bar=True,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            curriculum_callback,
            test_loss_logger,
            early_stopping,
        ],
    )

    # Train and get trained model
    trainer.fit(plmodule, pldatamodule)

    # Load the best model checkpoint
    plmodule = LightningTrainer.load_from_checkpoint(
        checkpoint_callback.best_model_path, map_location=device, **plmodule_kwargs
    )
    model = plmodule.model.to(device)

    # validate with trained model
    trainer.validate(plmodule, pldatamodule)

    # Numerical simulation times
    maximum_time = test_dataset[0].WD.shape[1]
    numerical_times = get_numerical_times(
        test_dataset_name + "_test",
        test_size,
        temporal_res,
        maximum_time,
        **temporal_test_dataset_parameters,
        overview_file="database/overview.csv",
    )

    # Rollout error and time
    start_time = time.time()
    predicted_rollout = trainer.predict(plmodule, dataloaders=test_dataloader)
    prediction_times = time.time() - start_time
    prediction_times = prediction_times / len(temporal_test_dataset)
    predicted_rollout = [item for roll in predicted_rollout for item in roll]

    spatial_analyser = SpatialAnalysis(
        predicted_rollout,
        prediction_times,
        test_dataset,
        **temporal_test_dataset_parameters,
    )

    rollout_loss = spatial_analyser._get_rollout_loss(type_loss=type_loss)
    model_times = spatial_analyser.prediction_times
    test_roll_loss_WD = rollout_loss.mean(0)[0].item()
    test_roll_loss_V = rollout_loss.mean(0)[1:].mean().item()
    test_loss_overall = rollout_loss.mean().item()

    print("test roll loss WD:", test_roll_loss_WD)
    print("test roll loss V:", test_roll_loss_V)

    # Speed up
    avg_speedup, std_speedup = get_speed_up(numerical_times, model_times)

    print(
        f"test CSI_005: {spatial_analyser._get_CSI(water_threshold=0.05).nanmean().item()}"
    )
    print(
        f"test CSI_03: {spatial_analyser._get_CSI(water_threshold=0.3).nanmean().item()}"
    )

    if logger:
        completed_epoch = trainer.current_epoch
        logger.log_metrics(
            {
                "speed-up": avg_speedup,
                "test_loss_overall": test_loss_overall,
                "test_roll_loss_WD_overall": test_roll_loss_WD,
                "test_roll_loss_V_overall": test_roll_loss_V,
                "test roll loss WD": test_roll_loss_WD,
                "test roll loss V": test_roll_loss_V,
                "test CSI_005": spatial_analyser._get_CSI(water_threshold=0.05)
                .nanmean()
                .item(),
                "test CSI_03": spatial_analyser._get_CSI(water_threshold=0.3)
                .nanmean()
                .item(),
            },
            step=completed_epoch,
        )

    fig, _ = spatial_analyser.plot_CSI_rollouts(water_thresholds=[0.05, 0.3])
    plt.savefig("results/CSI.png")

    best_id = rollout_loss.mean(1).argmin().item()
    worst_id = rollout_loss.mean(1).argmax().item()

    for id_dataset, name in zip([best_id, worst_id], ["best", "worst"]):
        rollout_plotter = PlotRollout(
            model.to(device),
            test_dataset[id_dataset].to(device),
            scalers=scalers,
            type_loss=type_loss,
            **temporal_test_dataset_parameters,
        )
        if model_type == "MSGNN":
            fig = rollout_plotter.explore_rollout(scale=0)
        else:
            fig = rollout_plotter.explore_rollout()
        plt.savefig(f"results/simulation_{name}.png")

    print("Training and testing finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mSWE-GNN model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Read configuration file with parameters
    cfg = read_config(args.config)

    logger = get_aim_logger(cfg)
    config = normalize_config(cfg)

    main(config, logger)
