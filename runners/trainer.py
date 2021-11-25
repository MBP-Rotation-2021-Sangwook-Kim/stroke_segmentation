import os


def trainer(model, 
            criterion, 
            metric, 
            train_loader, 
            valid_loader,
            writer, 
            log_dir, 
            device, 
            num_epochs, 
            amp: bool,
            scaler):

    # general training params
    epoch_num = 500   # max epochs 500
    early_stop = 125
    early_stop_counter = 0
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    val_loss_values = list()
    metric_values = list()
    epoch_times = list()
    total_start = time.time()

    post_pred = tf.AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
    post_label = tf.AsDiscrete(to_onehot=True, n_classes=2)

    dice_metric = DiceMetric(include_background=False)
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    print(f'Starting training over max {num_epochs} epochs...')
    for epoch in range(num_epochs):
        epoch_start = time.time()
        early_stop_counter += 1
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        step_start = time.time()
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            if amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}"
                f" step time: {(time.time() - step_start):.4f}")
            step_start = time.time()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"time consuming of epoch {epoch + 1} is: {epoch_time:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                step = 0
                metric_sum = 0
                metric_count = 0
                for val_data in val_loader:
                    step += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    if amp:
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                            # val_outputs = model(val_inputs)
                            loss = loss_function(val_outputs, val_labels)
                    else:
                        val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                        # val_outputs = model(val_inputs)
                        loss = loss_function(val_outputs, val_labels)
                    val_loss += loss.item()
                    # val_outputs = post_pred(val_outputs)
                    # val_labels = post_label(val_labels)
                    dice = compute_meandice(
                        y_pred=post_pred(val_outputs[0]).unsqueeze(0),
                        y=post_label(val_labels[0]).unsqueeze(0),
                        include_background=False,
                    ).item()
                    # val_labels_list = decollate_batch(val_labels)
                    # val_labels_convert = [
                    #     post_label(val_label_tensor) for val_label_tensor in val_labels_list
                    # ]
                    # val_outputs_list = decollate_batch(val_outputs)
                    # val_outputs_convert = [
                    #     post_pred(val_output_tensor) for val_output_tensor in val_outputs_list
                    # ]
                    # dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
                    # dice = dice_metric.aggregate().item()
                    metric_count += 1
                    metric_sum += dice

                val_loss /= step
                val_loss_values.append(val_loss)

                metric = metric_sum / metric_count
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                    print("saved new best metric model")
                    early_stop_counter = 0
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
                )
                if early_stop_counter >= early_stop:
                    print(f"No validation metric improvement in {early_stop} epochs. "
                        f"Early stopping triggered. Breaking training loop.")
                    break
                    
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
        f" total time: {(time.time() - total_start):.4f}")
    # save loss and validation metric lists
    with open(os.path.join(out_dir, "train_losses.txt"), "wb") as fp:
        pickle.dump(epoch_loss_values, fp)
    with open(os.path.join(out_dir, "val_losses.txt"), "wb") as fp:
        pickle.dump(val_loss_values, fp)
    with open(os.path.join(out_dir, "val_metrics.txt"), "wb") as fp:
        pickle.dump(metric_values, fp)
    with open(os.path.join(out_dir, "epoch_times.txt"), "wb") as fp:
        pickle.dump(epoch_times, fp)
    pass

if __name__=="__main__":
    pass
