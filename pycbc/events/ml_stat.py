"""Machine-learning models for evaluating ranking-statistic densities."""

import copy
import logging
from typing import Type

import h5py
import numpy
import tqdm

try:
    import glasflow.flows as glasflow_flows
    import torch
except ImportError as exc:
    raise ImportError(
        "Glasflow and torch are required to use pycbc.events.ml_stat"
    ) from exc


logger = logging.getLogger("pycbc.events.ml_stat")


class MLModel:
    """Base class for machine-learning density models."""

    def fit(self, x, **kwargs):
        """Fit the model to a set of samples."""
        raise NotImplementedError

    def sample(self, n_samples=1):
        """Draw samples from the model."""
        raise NotImplementedError

    def log_prob(self, x):
        """Evaluate the log probability of a set of samples."""
        raise NotImplementedError

    def to_file(self, filename, **kwargs):
        """Save the model to a file."""
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Load the model from a file."""
        raise NotImplementedError


class DataTransform:
    """Map data between bounded parameter space and the unit hypercube."""

    def __init__(self, bounds, device):
        self.bounds = torch.as_tensor(
            bounds,
            dtype=torch.get_default_dtype(),
            device=device,
        )
        self.log_abs_det_jacobian = -torch.log(
            self.bounds[:, 1] - self.bounds[:, 0]
        ).sum()

    def forward(self, x):
        """Map samples to the unit hypercube."""
        transformed = (
            (x - self.bounds[:, 0])
            / (self.bounds[:, 1] - self.bounds[:, 0])
        )
        log_jacobian = self.log_abs_det_jacobian * torch.ones(
            x.shape[0], device=x.device, dtype=x.dtype
        )
        return transformed, log_jacobian

    def inverse(self, x):
        """Map samples from the unit hypercube to parameter space."""
        transformed = (
            x * (self.bounds[:, 1] - self.bounds[:, 0])
            + self.bounds[:, 0]
        )
        log_jacobian = -self.log_abs_det_jacobian * torch.ones(
            x.shape[0], device=x.device, dtype=x.dtype
        )
        return transformed, log_jacobian

    def in_bounds(self, x):
        """Return a mask identifying samples inside the transform bounds."""
        below_lower = (x < self.bounds[:, 0]).any(axis=-1)
        above_upper = (x > self.bounds[:, 1]).any(axis=-1)
        return ~(below_lower | above_upper)


class NormalizingFlow(MLModel):
    """Normalizing-flow model for density estimation.

    Parameters
    ----------
    dims : int
        Number of input dimensions.
    flow_class : str, optional
        Name of the normalizing-flow class in ``glasflow.flows``.
    device : str, optional
        Device used for training and evaluation.
    bounds : numpy.ndarray
        Lower and upper bounds for each input dimension. The expected shape is
        ``(dims, 2)``.
    metadata : dict, optional
        Metadata stored with the model.
    context_dims : int, optional
        Number of conditional (context) inputs. Use 0 (default) for an
        unconditional flow.
    **kwargs
        Additional keyword arguments passed to the flow class.
    """

    def __init__(
        self,
        dims,
        flow_class="CouplingNSF",
        device="cpu",
        bounds=None,
        metadata=None,
        context_dims=0,
        **kwargs,
    ):
        self.dims = dims
        self.context_dims = int(context_dims)
        self.device = torch.device(device)
        self.flow_class = flow_class
        self.metadata = {} if metadata is None else metadata
        self.kwargs = self._set_defaults(**kwargs)
        self.data_transform = DataTransform(bounds, self.device)

        flow_type = getattr(glasflow_flows, self.flow_class, None)
        if flow_type is None:
            raise ValueError(f"Unknown flow class: {self.flow_class}")

        self.flow = flow_type(
            n_inputs=dims,
            n_conditional_inputs=self.context_dims or None,
            **self.kwargs,
        )
        self.flow.to(self.device)
        logger.info("Initialized normalizing flow:\n%s", self.flow)

    @staticmethod
    def _set_defaults(**kwargs):
        """Set default flow parameters that were not supplied."""
        kwargs = copy.deepcopy(kwargs)
        kwargs.setdefault("n_transforms", 4)
        kwargs.setdefault("n_neurons", 180)
        kwargs.setdefault("batch_norm_between_transforms", False)
        kwargs.setdefault("distribution", "uniform")
        kwargs.setdefault("num_bins", 20)
        kwargs.setdefault("min_derivative", 1e-6)
        kwargs.setdefault("min_bin_width", 1e-6)
        kwargs.setdefault("min_bin_height", 1e-6)
        return kwargs

    def _conditional_tensor(self, conditional, n_rows):
        """Convert conditional inputs to a ``(n_rows, context_dims)`` tensor."""
        if conditional is None:
            if self.context_dims:
                raise ValueError("This flow requires conditional inputs")
            return None
        if not self.context_dims:
            raise ValueError("This flow was built without conditional inputs")
        conditional = torch.as_tensor(
            conditional,
            dtype=torch.get_default_dtype(),
            device=self.device,
        ).reshape(n_rows, self.context_dims)
        return conditional

    def _loss(self, x, conditional=None):
        """Return the negative mean log likelihood for a batch."""
        return -self.flow.log_prob(x, conditional=conditional).mean()

    def fit(
        self,
        x,
        n_epochs=200,
        lr=1e-3,
        batch_size=5000,
        validation_fraction=0.2,
        lr_annealing=False,
        n_samples=None,
        conditional=None,
    ):
        """Fit the normalizing flow to a set of samples.

        Parameters
        ----------
        x : numpy.ndarray
            Samples with shape ``(n_samples, dims)``.
        n_epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Optimizer learning rate.
        batch_size : int, optional
            Number of samples in each training batch.
        validation_fraction : float, optional
            Fraction of samples reserved for validation.
        lr_annealing : bool, optional
            Apply cosine learning-rate annealing during training.
        n_samples : int, optional
            Maximum number of shuffled samples to use.
        conditional : numpy.ndarray, optional
            Conditional inputs with shape ``(n_samples, context_dims)``.
            Required if the flow was built with ``context_dims > 0``.

        Returns
        -------
        dict
            Training and validation losses for each epoch.
        """
        x = torch.as_tensor(
            x,
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        conditional = self._conditional_tensor(conditional, x.shape[0])
        x = self.data_transform.forward(x)[0]
        perm = torch.randperm(x.shape[0], device=self.device)
        x = x[perm]
        if conditional is not None:
            conditional = conditional[perm]

        if n_samples is not None:
            x = x[:n_samples]
            if conditional is not None:
                conditional = conditional[:n_samples]

        for name, tensor in (("Data", x), ("Conditional data", conditional)):
            if tensor is None:
                continue
            if torch.isnan(tensor).any():
                raise ValueError(f"{name} contains NaN values")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains infinite values")

        n_validation = int(validation_fraction * x.shape[0])
        if n_validation == 0:
            raise ValueError("Validation set is empty")

        train_tensors = [x[:-n_validation]]
        validation_tensors = [x[-n_validation:]]
        if conditional is not None:
            train_tensors.append(conditional[:-n_validation])
            validation_tensors.append(conditional[-n_validation:])
        logger.info(
            "Training on %d samples and validating on %d samples",
            train_tensors[0].shape[0],
            validation_tensors[0].shape[0],
        )

        training_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*train_tensors),
            batch_size=batch_size,
            shuffle=True,
        )
        validation_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*validation_tensors),
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = None
        if lr_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, n_epochs
            )

        history = {"training_loss": [], "validation_loss": []}
        for _ in tqdm.tqdm(range(n_epochs)):
            self.flow.train()
            training_loss = 0.0
            for batch in training_loader:
                optimizer.zero_grad()
                loss = self._loss(*batch)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            history["training_loss"].append(
                training_loss / len(training_loader)
            )

            self.flow.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for batch in validation_loader:
                    validation_loss += self._loss(*batch).item()
            history["validation_loss"].append(
                validation_loss / len(validation_loader)
            )

        self.flow.eval()
        return history

    def sample(self, n_samples=1, conditional=None):
        """Draw samples from the normalizing flow.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw.
        conditional : numpy.ndarray, optional
            Conditional inputs. Either a single row, which is used for all
            ``n_samples`` samples, or one row per sample with shape
            ``(n_samples, context_dims)``.

        Returns
        -------
        numpy.ndarray
            Samples with shape ``(n_samples, dims)``.
        """
        self.flow.eval()
        with torch.no_grad():
            if conditional is None:
                samples = self.flow.sample(n_samples)
            else:
                conditional = numpy.asarray(conditional).reshape(
                    -1, self.context_dims
                )
                if conditional.shape[0] == 1:
                    conditional = numpy.repeat(conditional, n_samples, axis=0)
                conditional = self._conditional_tensor(
                    conditional, conditional.shape[0]
                )
                # glasflow pairs noise row i with conditional row i, so
                # draw one noise sample per conditional row.
                samples = self.flow.sample(
                    conditional.shape[0], conditional=conditional
                )
            samples = self.data_transform.inverse(samples)[0]
        return samples.cpu().numpy()

    def log_prob(self, x, conditional=None):
        """Evaluate the log probability of a set of samples.

        Parameters
        ----------
        x : numpy.ndarray
            Samples with shape ``(n_samples, dims)``.
        conditional : numpy.ndarray, optional
            Conditional inputs with shape ``(n_samples, context_dims)``.
            Required if the flow was built with ``context_dims > 0``.

        Returns
        -------
        numpy.ndarray
            Log probabilities for the supplied samples.
        """
        self.flow.eval()
        with torch.no_grad():
            x = torch.as_tensor(
                x,
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
            conditional = self._conditional_tensor(conditional, x.shape[0])
            log_prob = torch.full(
                (x.shape[0],),
                -torch.inf,
                dtype=x.dtype,
                device=self.device,
            )
            in_bounds = self.data_transform.in_bounds(x)
            if not torch.any(in_bounds):
                return log_prob.cpu().numpy()

            if conditional is not None:
                conditional = conditional[in_bounds]
            transformed, log_jacobian = self.data_transform.forward(
                x[in_bounds]
            )
            log_prob[in_bounds] = (
                self.flow.log_prob(transformed, conditional=conditional)
                + log_jacobian
            )
        return log_prob.cpu().numpy()

    def prob(self, x, conditional=None):
        """Evaluate the probability density of a set of samples."""
        return numpy.exp(self.log_prob(x, conditional=conditional))

    @classmethod
    def from_file(cls, filename, group_name="model", load_weights=True):
        """Load a normalizing flow from an HDF file.

        Parameters
        ----------
        filename : str
            Name of the HDF file to read.
        group_name : str, optional
            Name of the group containing the model.
        load_weights : bool, optional
            Load the saved model weights.

        Returns
        -------
        NormalizingFlow
            Loaded normalizing-flow model.
        """
        with h5py.File(filename, "r") as hdf_file:
            model_group = hdf_file[group_name]
            model_name = model_group.attrs["model_class"]
            if model_name.lower() != cls.__name__.lower():
                raise ValueError(
                    f"File contains {model_name}, not {cls.__name__}"
                )

            config = dict(model_group["config"].attrs)
            metadata = {}
            if "metadata" in hdf_file:
                metadata = dict(hdf_file["metadata"].attrs)

            logger.info("Loading model with config: %s", config)
            model = cls(metadata=metadata, **config)

            if load_weights:
                logger.info("Loading model weights")
                weights = model_group["weights"]
                state_dict = {
                    key: torch.from_numpy(value[()]).to(model.device)
                    for key, value in weights.items()
                }
                model.flow.load_state_dict(state_dict)

        return model

    def to_file(self, filename, group_name="model", metadata=None):
        """Save the normalizing flow and metadata to an HDF file.

        Parameters
        ----------
        filename : str
            Name of the HDF file to create.
        group_name : str, optional
            Name of the group used to store the model.
        metadata : dict, optional
            Metadata to save instead of ``self.metadata``.
        """
        if metadata is None:
            metadata = self.metadata

        with h5py.File(filename, "w") as hdf_file:
            hdf_file.attrs["stat"] = metadata.get(
                "stat", self.__class__.__name__
            )

            metadata_group = hdf_file.create_group("metadata")
            for key, value in metadata.items():
                metadata_group.attrs[key] = value

            model_group = hdf_file.create_group(group_name)
            model_group.attrs["model_class"] = self.__class__.__name__

            config_group = model_group.create_group("config")
            config_group.attrs["dims"] = self.dims
            config_group.attrs["context_dims"] = self.context_dims
            config_group.attrs["flow_class"] = self.flow_class
            config_group.attrs["bounds"] = (
                self.data_transform.bounds.cpu().numpy()
            )
            config_group.attrs["device"] = str(self.device)
            for key, value in sorted(self.kwargs.items()):
                config_group.attrs[key] = value

            weights_group = model_group.create_group("weights")
            for key, value in self.flow.state_dict().items():
                weights_group.create_dataset(
                    key,
                    data=numpy.atleast_1d(value.detach().cpu().numpy()),
                )



KNOWN_CLASSES = {
    "normalizingflow": NormalizingFlow,
}


class MLStatistic:
    """Handle a machine-learning model used as a ranking statistic.

    Parameters
    ----------
    filename : str, optional
        File containing a saved statistic. When supplied, ``model`` and
        ``model_class`` are ignored.
    model : MLModel, optional
        Initialized machine-learning model.
    model_class : type, optional
        Model class to initialize when neither ``filename`` nor ``model`` is
        supplied.
    metadata : dict, optional
        Metadata stored with the statistic.
    **kwargs
        Arguments passed to the model constructor or file loader.
    """

    def __init__(
        self,
        filename=None,
        model=None,
        model_class: Type[MLModel] = NormalizingFlow,
        metadata=None,
        **kwargs,
    ):
        if filename is not None:
            loaded = self.from_file(filename, **kwargs)
            self.model = loaded.model
            self.metadata = loaded.metadata
            return

        if model is not None:
            self.model = model
        else:
            if isinstance(model_class, str):
                model_name = model_class.lower()
                try:
                    model_class = KNOWN_CLASSES[model_name]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown model class: {model_class}"
                    ) from exc
            self.model = model_class(**kwargs)

        self.metadata = {} if metadata is None else metadata
        self.model.metadata = self.metadata

    @classmethod
    def from_file(cls, filename, group_name="model", load_weights=True):
        """Load a machine-learning statistic from an HDF file.

        Parameters
        ----------
        filename : str
            Name of the HDF file to read.
        group_name : str, optional
            Name of the group containing the model.
        load_weights : bool, optional
            Load the saved model weights.

        Returns
        -------
        MLStatistic
            Loaded machine-learning statistic.
        """
        with h5py.File(filename, "r") as hdf_file:
            model_name = hdf_file[group_name].attrs["model_class"].lower()

        try:
            model_class = KNOWN_CLASSES[model_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model class: {model_name}"
            ) from exc

        model = model_class.from_file(
            filename,
            group_name=group_name,
            load_weights=load_weights,
        )
        return cls(model=model, metadata=model.metadata)

    def to_file(self, filename, group_name="model"):
        """Save the machine-learning statistic to an HDF file.

        Parameters
        ----------
        filename : str
            Name of the HDF file to create.
        group_name : str, optional
            Name of the group used to store the model.
        """
        self.model.to_file(
            filename,
            group_name=group_name,
            metadata=self.metadata,
        )

    def log_prob(self, x, conditional=None):
        """Evaluate the log probability of a set of samples."""
        return self.model.log_prob(x, conditional=conditional)

    def fit(self, x, **kwargs):
        """Fit the machine-learning model to a set of samples."""
        return self.model.fit(x, **kwargs)

    def prob(self, x, conditional=None):
        """Evaluate the probability density of a set of samples."""
        return self.model.prob(x, conditional=conditional)