

# --- ASYMMETRIC-LOSS SIBLING FUNCTION ----------------------------------------
# Mirrors train_and_val but applies per-example temporal weights to the
# classification loss. Set asym_lambda > 0 to upweight pre-onset examples
# and downweight post-onset examples; asym_lambda = 0 reproduces standard CE.
# Requires raw_data (the dict loaded from the StudentLife pkl) so compound
# keys can be parsed for timestamp extraction.

from datetime import datetime as _datetime_for_asym
from collections import defaultdict as _defaultdict_for_asym


def _asym_parse_key(key):
    parts = key.split("_")
    if len(parts) < 4:
        return None, None
    try:
        return parts[0], _datetime_for_asym(
            2012, int(parts[1]), int(parts[2]), int(parts[3])
        )
    except (ValueError, IndexError):
        return None, None


def _asym_ts_to_hours(ts):
    return (ts - _datetime_for_asym(2012, 1, 1)).total_seconds() / 3600.0


def _asym_compute_weights(train_keys, raw_data, asym_lambda, window_hours=72.0):
    """Return dict: compound_key -> float weight."""
    by_student = _defaultdict_for_asym(list)
    for k in train_keys:
        student, ts = _asym_parse_key(k)
        if student is None:
            continue
        label = int(raw_data['data'][k][5])
        by_student[student].append((_asym_ts_to_hours(ts), k, label))

    pre_w  = 1.0 + asym_lambda
    post_w = 1.0 / (1.0 + asym_lambda) if asym_lambda > 0 else 1.0

    weights = {}
    for student, examples in by_student.items():
        examples.sort(key=lambda x: x[0])
        hours_seq = [e[0] for e in examples]
        labels    = [e[2] for e in examples]
        onsets = [hours_seq[j] for j in range(1, len(labels))
                  if labels[j] > labels[j - 1]]
        for h, k, _ in examples:
            if not onsets:
                weights[k] = 1.0
                continue
            nearest = min(onsets, key=lambda o: abs(o - h))
            gap = h - nearest
            if -window_hours <= gap < 0:
                weights[k] = pre_w
            elif 0 <= gap <= window_hours:
                weights[k] = post_w
            else:
                weights[k] = 1.0
    return weights


def train_and_val_asym(
    data,
    model_params,
    training_params,
    raw_data,
    asym_lambda,
    pre_record=None,
    leaved_student=None,
    up_weight_k=None,
):
    """Asymmetric-loss variant of train_and_val. Same training loop and
    saved_records structure, but per-batch classification loss is
    multiplied by per-example temporal weights.

    Parameters
    ----------
    data : dict
        Already-tensorified data (same format train_and_val receives).
    raw_data : dict
        The raw dict loaded from the StudentLife pickle, used here only
        to map compound keys to their stress labels for onset detection.
    asym_lambda : float
        Asymmetry strength. 0 reproduces standard CE.
    """
    train_data, val_data = formatting_train_val_data(data, training_params)

    # Compute per-example temporal weights once before training begins.
    # train_compound_keys must align with train_data positions; data['train_ids']
    # gives us those keys in their original order.
    train_compound_keys = list(data['train_ids'])
    weight_dict = _asym_compute_weights(
        train_compound_keys, raw_data, asym_lambda
    )
    weight_per_position = torch.tensor(
        [weight_dict.get(k, 1.0) for k in train_compound_keys],
        dtype=torch.float32, device=training_params['device']
    )

    n_pre  = int((weight_per_position > 1.0).sum().item())
    n_post = int((weight_per_position < 1.0).sum().item())
    n_neu  = len(weight_per_position) - n_pre - n_post
    print(f"asymmetric weighting (lambda={asym_lambda}): "
          f"pre={n_pre}, post={n_post}, neutral={n_neu}")

    saved_records = {
        'model': None,
        'train_losses': list(),
        'val_losses': list(),
        'outputs': list(),
        'generic_outputs': list(),
        'confmats': list(),
        'val_f1': {'micro': list(), 'macro': list(), 'weighted': list()},
        'val_auc': {'micro': list(), 'macro': list(), 'weighted': list()},
        'labels': list(),
        'generic_records': {
            'outputs': list(),
            'confmats': list(),
            'val_f1': {'micro': list(), 'macro': list(), 'weighted': list()},
            'val_auc': {'micro': list(), 'macro': list(), 'weighted': list()},
        }
    }

    print('Initializing...')
    model = None
    if pre_record is None:
        model = MultitaskAutoencoder(model_params, training_params['use_covariates']).to(training_params['device'])
    else:
        model = pre_record['model'].to(training_params['device'])

    reconstruction_criterion = torch.nn.L1Loss(reduction="sum")

    # Two CE criteria: per-example (for asymmetric training) and mean (for validation).
    cls_per_ex = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(training_params['class_weights'], device=training_params['device']),
        reduction='none'
    )
    cls_mean = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(training_params['class_weights'], device=training_params['device'])
    )

    optimizer = torch.optim.Adam(
        [
            {'params': model.autoencoder.parameters()},
            {'params': model.out_heads.parameters()},
            {'params': model.branching.parameters(), 'lr': training_params['branching_lr']},
            {'params': model.branch_layer.parameters(), 'lr': training_params['branching_lr']},
        ],
        lr=training_params['global_lr'],
        weight_decay=training_params['weight_decay'],
    )

    print('Training...')
    for epoch in tqdm.tqdm(range(training_params['epochs'])):
        # training
        model.train()
        batchs = get_mini_batchs(training_params['batch_size'], train_data['inds'])
        train_loss = 0
        for batch in batchs:
            if not model.with_generic_head:
                final_out, AE_out = model(
                    x=train_data['samples'],
                    inds=batch,
                    ids=train_data['ids'][batch],
                    covariate_data=train_data['covariate_data'][batch]
                )
            else:
                final_out, AE_out, generic_out = model(
                    x=train_data['samples'],
                    inds=batch,
                    ids=train_data['ids'][batch],
                    covariate_data=train_data['covariate_data'][batch]
                )

            # Per-example temporal weighting (the asymmetric step)
            batch_weights = weight_per_position[batch]
            per_ex_loss = cls_per_ex(final_out, train_data['labels'][batch])
            classification_loss = (per_ex_loss * batch_weights).mean() \
                                  * training_params['loss_weight']['beta']
            total_loss = classification_loss

            if training_params['use_decoder']:
                reconstruction_loss = 0
                for i in range(len(AE_out)):
                    reconstruction_loss += reconstruction_criterion(
                        train_data['samples'][batch[i]], AE_out[i]
                    )
                reconstruction_loss *= training_params['loss_weight']['alpha']
                total_loss = reconstruction_loss + classification_loss

            if model.with_generic_head:
                # Generic head keeps standard mean CE
                total_loss += cls_mean(generic_out, train_data['labels'][batch]) \
                              * training_params['loss_weight']['theta']

            model.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.cpu().detach().item()

        # validation (standard mean CE, identical to train_and_val)
        model.eval()
        if not model.with_generic_head:
            final_out, AE_out = model(
                x=val_data['samples'],
                inds=list(range(len(val_data['samples']))),
                ids=val_data['ids'],
                covariate_data=val_data['covariate_data']
            )
        else:
            final_out, AE_out, generic_out = model(
                x=val_data['samples'],
                inds=list(range(len(val_data['samples']))),
                ids=val_data['ids'],
                covariate_data=val_data['covariate_data']
            )

        classification_loss = cls_mean(final_out, val_data['labels']) \
                              * training_params['loss_weight']['beta']
        val_loss = classification_loss

        if training_params['use_decoder']:
            reconstruction_loss = 0
            for i in range(len(AE_out)):
                reconstruction_loss += reconstruction_criterion(val_data['samples'][i], AE_out[i])
            reconstruction_loss *= training_params['loss_weight']['alpha']
            val_loss = reconstruction_loss + classification_loss

        saved_records['train_losses'].append(train_loss)
        saved_records['val_losses'].append(val_loss.cpu().detach().item())

        saved_records['outputs'].append(final_out.cpu().detach().numpy())
        y_pred = np.argmax(saved_records['outputs'][-1], axis=1)
        y_true = val_data['labels'].cpu().detach().numpy()
        labels = [[0], [1], [2]]
        saved_records['confmats'].append(
            metrics.confusion_matrix(y_true, y_pred, labels=[i[0] for i in labels])
        )
        for avg_type in ['micro', 'macro', 'weighted']:
            saved_records['val_auc'][avg_type].append(
                eval_auc_score(saved_records['outputs'][-1], y_true, labels, avg_type)
            )
            saved_records['val_f1'][avg_type].append(
                eval_f1_score(y_pred, y_true, avg_type)
            )

        if saved_records['val_f1']['weighted'][-1] == max(saved_records['val_f1']['weighted']):
            saved_records['model'] = copy.deepcopy(model).cpu()

        if model.with_generic_head:
            val_loss += cls_mean(generic_out, val_data['labels']) \
                        * training_params['loss_weight']['theta']
            saved_records['generic_records']['outputs'].append(
                generic_out.cpu().detach().numpy()
            )
            y_pred = np.argmax(saved_records['generic_records']['outputs'][-1], axis=1)
            y_true = val_data['labels'].cpu().detach().numpy()
            labels = [[0], [1], [2]]
            saved_records['generic_records']['confmats'].append(
                metrics.confusion_matrix(y_true, y_pred, labels=[i[0] for i in labels])
            )
            for avg_type in ['micro', 'macro', 'weighted']:
                saved_records['generic_records']['val_auc'][avg_type].append(
                    eval_auc_score(saved_records['generic_records']['outputs'][-1], y_true, labels, avg_type)
                )
                saved_records['generic_records']['val_f1'][avg_type].append(
                    eval_f1_score(y_pred, y_true, avg_type)
                )

        print("F1 Score This Epoch: {} Best Score: {}".format(
            saved_records['val_f1']['weighted'][-1],
            max(saved_records['val_f1']['weighted'])
        ))

    return saved_records

# --- END ASYMMETRIC-LOSS SIBLING --------------------------------------------
