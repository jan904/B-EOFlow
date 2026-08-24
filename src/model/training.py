from src.model.INN.INN_training import train_INN


def train_model(
    model,
    optimizer,
    losses,
    kwargs_data,
    kwargs_loss,
    save_model_path,
    model_config,
    val_metrics_loss=None,
    start_epoch=0,
    num_epochs=1,
    print_info=True,
    log_dir=None,
    continue_log_path=None,
    conditions=None,
    use_counts=False,
    save_on_validation=False,
    ood_probe=None,
    ood_probe_every=50,
):

    if model_config.model_type == "inn":
        return train_INN(
            model,
            optimizer,
            losses,
            kwargs_data,
            kwargs_loss,
            save_model_path,
            model_config,
            val_metrics_loss=val_metrics_loss,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            print_info=print_info,
            log_dir=log_dir,
            continue_log_path=continue_log_path,
            conditions=conditions,
            use_counts=use_counts,
            save_on_validation=save_on_validation,
            ood_probe=ood_probe,
            ood_probe_every=ood_probe_every,
        )
    elif model_config.model_type == "vae":
        pass
    else:
        print(f"Unknown model_type '{model_config.model_type}'. Falling back to INN.")
        return train_INN(
            model,
            optimizer,
            losses,
            kwargs_data,
            kwargs_loss,
            save_model_path,
            model_config,
            val_metrics_loss=val_metrics_loss,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            print_info=print_info,
            log_dir=log_dir,
            continue_log_path=continue_log_path,
            conditions=conditions,
            use_counts=use_counts,
            save_on_validation=save_on_validation,
            ood_probe=ood_probe,
            ood_probe_every=ood_probe_every,
        )
