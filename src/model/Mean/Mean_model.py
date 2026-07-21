_SETUP_KEY_ = "mean_shifter_setup"


class MeanShifter:
    def __init__(self, adata):
        if _SETUP_KEY_ not in adata.uns:
            raise ValueError(f"adata.uns['{_SETUP_KEY_}'] not found. Please run setup_anndata first.")

        self.adata = adata
        self.setup_dict = adata.uns[_SETUP_KEY_]

        self.labels_key = self.setup_dict["labels_key"]
        self.control_label = self.setup_dict["control_label"]

        self.deltas_ = {}
        self.ctrl_mean_ = None

    @classmethod
    def setup_anndata(cls, adata, labels_key, control_label):
        adata.uns[_SETUP_KEY_] = {
            "labels_key": labels_key,
            "control_label": control_label,
        }

    def train(self):
        self.ctrl_mean_ = self.adata[self.adata.obs[self.labels_key] == self.control_label].X.mean(axis=0)

        for label in self.adata.obs[self.labels_key].unique():
            if label == self.control_label:
                continue
            label_mask = self.adata.obs[self.labels_key] == label
            label_mean = self.adata[label_mask].X.mean(axis=0)
            delta = label_mean - self.ctrl_mean_
            self.deltas_[label] = delta
