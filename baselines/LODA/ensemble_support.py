from loda_support import *


class ComputedModel(object):
    """
    Attributes:
        anoms: numpy.ndarray
        lbls: numpy.array
        topanomidxs: list of int
        proj_wts: numpy.array
        w: numpy.ndarray
        hpdfs: numpy.ndarray
        hists: list
        nlls: numpy.ndarray
        orderedprojs: numpy.array
        anom_score: numpy.array
        order_anom_idxs: numpy.array
    """
    def __init__(self, anoms=None, lbls=None, topanomidxs=None,
                 proj_wts=None,
                 w=None, hpdfs=None, hists=None, nlls=None,
                 orderedprojs=None, anom_score=None,
                 order_anom_idxs=None):
        self.anoms = anoms
        self.lbls = lbls
        self.topanomidxs = topanomidxs
        self.proj_wts = proj_wts
        self.w = w
        self.hpdfs = hpdfs
        self.hists = hists
        self.nlls = nlls
        self.orderedprojs = orderedprojs
        self.anom_score = anom_score
        self.order_anom_idxs = order_anom_idxs


def generate_model_from_loda_result(lodares, samples, labels):
    """Computes a descriptive model using the LODA information.

    Also orders the histogram projections based on the accuracy (cheats to do this.)
    The ordering of projections only helps in analyzing the results later. It
    does not affect algorithm performance.

    Args:
        lodares: LodaResult
        samples: numpy.ndarray
        labels: numpy.array
    Returns:
        ComputedModel
    """

    # get indexes of top anomalies
    n = nrow(samples)
    topanomidxs = order(lodares.nll, decreasing=True)[0:n]
    anoms = samples[topanomidxs,]
    samples = None  # to be safe; make sure this is not accidentally accessed.
    labels = labels[topanomidxs]  # consistent with order of anoms

    hists = lodares.pvh.pvh.hists
    w = lodares.pvh.pvh.w
    hpdfs = get_all_hist_pdfs(anoms, w, hists)
    m = ncol(w)

    orderedprojs = None
    if False:
        # Sort projections on the individual AUCs (by cheating)
        # This does not change any performance for ALAD, just helps
        # in visualizing performance based on individual histogram quality later.
        hprecs = []
        for k in range(m):
            hprec = fn_precision(d=cbind(labels, hpdfs[:, k]), k=[10])[1]  # APR
            hprecs.append(hprec)
        orderedprojs = order(hprecs)
        hists = [hists[op] for op in orderedprojs]
        w = w[:, orderedprojs]
        hpdfs = hpdfs[:, orderedprojs]

    # use the hist pdfs as anomaly scores
    nlls = -np.log(hpdfs)

    proj_wts = np.ones(m, dtype=float) * 1 / np.sqrt(m)
    anom_score = nlls.dot(proj_wts)
    order_anom_idxs = order(anom_score, decreasing=True)

    model = ComputedModel(anoms=anoms, lbls=labels, topanomidxs=topanomidxs,
                          proj_wts=proj_wts,
                          w=w, hpdfs=hpdfs, hists=hists, nlls=nlls,
                          orderedprojs=orderedprojs, anom_score=anom_score,
                          order_anom_idxs=order_anom_idxs)
    return model


def load_ensemble_scores(scoresfile, header=True):
    df = read_csv(scoresfile, header=header)
    # first (0-th) column is label
    scores = np.ndarray(shape=(df.shape[0], df.shape[1] - 1), dtype=float)
    scores[:, :] = df.iloc[:, 1:ncol(df)]
    strlabels = df.iloc[:, 0]
    labels = np.array([1 if label == "anomaly" else 0 for label in strlabels], dtype=int)
    return scores, labels


class Ensemble(object):
    """Stores all ensemble scores"""

    def __init__(self, samples, labels, scores, weights,
                 agg_scores=None, ordered_anom_idxs=None, original_indexes=None,
                 auc=0.0, model=None):
        self.samples = samples
        self.labels = labels
        self.scores = scores
        self.weights = weights
        self.agg_scores = agg_scores
        self.ordered_anom_idxs = ordered_anom_idxs
        self.original_indexes = original_indexes
        self.auc = auc
        self.model = model

        if original_indexes is None:
            self.original_indexes = np.arange(samples.shape[0])

        if agg_scores is not None and ordered_anom_idxs is None:
            self.ordered_anom_idxs = order(agg_scores, decreasing=True)


class EnsembleManager(object):
    """Load the scores from ensemble of algorithms"""

    def load_data(self, samples, labels, opts):
        pass

    @staticmethod
    def get_ensemble_manager(opts):
        if opts.ensembletype == "regular":
            logger.debug("Using PrecomputedEnsemble...")
            return PrecomputedEnsemble(opts)
        elif opts.ensembletype == "loda":
            logger.debug("Using LodaEnsemble...")
            return LodaEnsemble(opts)
        else:
            raise ValueError("Invalid ensemble type: %s" % (opts.ensembletype, ))


class LodaEnsemble(EnsembleManager):

    def __init__(self, opts):
        self.modelmanager = ModelManager.get_model_manager(opts.cachetype)

    @staticmethod
    def ensemble_from_lodares(lodares, samples, labels):
        model = generate_model_from_loda_result(lodares, samples, labels)
        anoms, lbls, _, _, _, detector_scores, detector_wts = (
            model.anoms, model.lbls,
            model.w, model.hists, model.hpdfs, model.nlls, model.proj_wts
        )
        auc = fn_auc(cbind(lbls, -model.anom_score))
        return Ensemble(anoms, lbls, detector_scores, detector_wts,
                        agg_scores=model.anom_score, ordered_anom_idxs=model.order_anom_idxs,
                        original_indexes=model.topanomidxs, auc=auc, model=model)

    def load_data(self, samples, labels, opts):
        algo_result = self.modelmanager.get_model(samples, opts)
        return LodaEnsemble.ensemble_from_lodares(algo_result, samples, labels)


class PrecomputedEnsemble(EnsembleManager):
    """Anomaly score for each instance is precomputed for each algorithm in ensemble

    The scores are stored in CSV format with one detector per column.
    First column has true label ('anomaly'|'nominal'). Default aggregate
    score is the average score across detectors for an instance.
    """

    def __init__(self, opts):
        self.modelmanager = ModelManager.get_model_manager(opts.cachetype)

    def load_data(self, samples, labels, opts, scores=None):
        if scores is None:
            scores, labels = load_ensemble_scores(scoresfile=opts.scoresfile, header=True)
        n, m = scores.shape
        detector_wts = rep(1./np.sqrt(m), m)
        agg_scores = scores.dot(detector_wts)
        auc = fn_auc(cbind(labels, -agg_scores))
        return Ensemble(samples, labels, scores, detector_wts,
                        agg_scores=agg_scores, auc=auc, model=None)


def get_loda_alad_ensembles(fids, runidxs, allsamples, opts):
    # load all models and pre-process the sorting of projections
    modelmanager = ModelManager.get_model_manager(opts.cachetype)
    ensembles = []
    for i in range(len(fids)):
        subensembles = []
        for j in range(len(runidxs)):
            opts.set_multi_run_options(fids[i], runidxs[j])

            a = allsamples[i].fmat
            lbls = allsamples[i].lbls

            lodares = modelmanager.load_model(opts)
            ensemble = LodaEnsemble.ensemble_from_lodares(lodares, a, lbls)
            subensembles.append(ensemble)

        ensembles.append(subensembles)
    return ensembles


