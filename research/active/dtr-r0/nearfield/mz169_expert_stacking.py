"""Fixed four-expert score stacking; no file access or threshold selection."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERTS = ('static', 'unregistered', 'compensated', 'reflection')
CLIP_EPSILON = 1e-6
METHOD = dict(
    schema='MZ169_FIXED_FOUR_EXPERT_LOGIT_STACKING_V1',
    expert_order=list(EXPERTS),
    transform='LOGIT_OF_CLIPPED_PROBABILITY',
    clip_probability=[CLIP_EPSILON, 1.-CLIP_EPSILON],
    candidate=dict(pipeline=['StandardScaler', 'LogisticRegression'],
        C=1., solver='lbfgs', max_iter=1000, random_state=169016,
        class_weight=None, fit_intercept=True),
    candidate_score='CLASS_ONE_PROBABILITY',
    control_score='ARITHMETIC_MEAN_OF_THE_SAME_FOUR_SIGNED_LOGITS',
    fitting='ONLY_ROWS_AND_LABELS_EXPLICITLY_PASSED_TO_FIT_MODEL',
    selection='NO_INTERACTIONS_CLASS_WEIGHTING_OR_PARAMETER_SEARCH',
)


def transform_scores(scores):
    """Validate nonempty finite Nx4 probabilities and return float64 logits.

    Columns must follow ``EXPERTS``. No identities, labels or other metadata
    are accepted. Input arrays are never modified.
    """
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Scores must be a finite numeric N x 4 array') from exc
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError('Scores must be a nonempty N x 4 array')
    if not np.isfinite(values).all() or np.any((values < 0.) | (values > 1.)):
        raise ValueError('Scores must be finite probabilities in [0, 1]')
    clipped = np.clip(values, CLIP_EPSILON, 1.-CLIP_EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def fit_model(scores, labels):
    """Fit only the provided rows; return StandardScaler + logistic Pipeline.

    The pipeline consumes transformed logits internally to this module's API.
    Use ``predict_score`` for original probability inputs. Both binary classes
    must occur in the supplied labels; no fallback model or weighting is used.
    """
    features = transform_scores(scores)
    try:
        target = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Labels must be one-dimensional binary values') from exc
    if (target.shape != (features.shape[0],) or not np.isfinite(target).all()
            or not np.isin(target, [0., 1.]).all()):
        raise ValueError('Labels must contain one binary value per score row')
    if len(np.unique(target)) != 2:
        raise ValueError('Both binary classes are required for logistic fitting')
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1., solver='lbfgs', max_iter=1000,
            random_state=169016, class_weight=None)),
    ])
    return model.fit(features, target.astype(np.int64))


def predict_score(model, scores):
    """Return class-one probabilities without fitting or updating the scaler."""
    features = transform_scores(scores)
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError('Expected a fitted binary model with classes [0, 1]')
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def mean_score(scores):
    """Return the signed arithmetic mean of the same four clipped logits."""
    return transform_scores(scores).mean(axis=1)
