import copy
import numpy as np
import logging
from typing import Type
import h5py
import torch
import tqdm

logger = logging.getLogger("pycbc.events.ml_stat")


class MLModel:
    """Base class for machine learning models.
    This class defines the interface for machine learning models used in MLStatistic.
    """

    def fit(self, x, **kwargs):
        raise NotImplementedError

    def sample(self, n_samples: int = 1):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def to_file(self, filename: str):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        raise NotImplementedError


class DataTransform:
    """Data transformation to map data to [0, 1]^d"""

    def __init__(self, bounds: np.ndarray, device):
        self.bounds = torch.from_numpy(bounds).to(device)
        self.device = device
        self.log_abs_det_jacobian = -torch.log(
            self.bounds[:, 1] - self.bounds[:, 0]
        ).sum()

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the transform to the data.
        For now, this just applies the transform.
        """
        return self.forward(x)[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (x - self.bounds[..., 0]) / (self.bounds[..., 1] - self.bounds[..., 0]),
            self.log_abs_det_jacobian
            * torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
        )

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            x * (self.bounds[..., 1] - self.bounds[..., 0]) + self.bounds[..., 0],
            -self.log_abs_det_jacobian
            * torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
        )

    def in_bounds(self, x):
        below_lower = (x < self.bounds[:,0]).any(-1)
        above_upper = (x > self.bounds[:,1]).any(-1)
        out = below_lower | above_upper
        return ~ out


class NormalizingFlow(MLModel):
    """Normalizing flow model for density estimation.
    Parameters
    ----------
    dims: int
        The number of dimensions of the input data.
    flow_class: str
        The class of normalizing flow to use. Options are "UNSF" (default)
        or any class in zuko.flows (e.g. "MAF", "NSF", etc).
    device: str
        The device to use for training. Default is "cpu".
    bounds: np.ndarray
        The bounds for each dimension of the input data. Shape should be (dims, 2).
    kwargs: dict
        Additional arguments to pass to the flow class.
    """

    def __init__(
        self,
        dims: int,
        flow_class: str = "CouplingNSF",
        device: str = "cpu",
        bounds: np.ndarray = None,
        **kwargs,
    ) -> None:

        import glasflow.flows as gf

        self.dims = dims
        self.device = torch.device(device)
        self.data_transform = DataTransform(bounds, device=self.device)
        self.flow_class = flow_class

        self.kwargs = self.set_defaults(**kwargs)

        FlowClass = getattr(gf, self.flow_class, None)
        if FlowClass is None:
            raise ValueError(f"Unknown flow class: {self.flow_class}")

        self.flow = FlowClass(
            n_inputs=dims,
            **self.kwargs,
        )
        self.flow.to(self.device)
        # self.flow.compile()
        logger.info(f"Initialized normalizing flow: \n {self.flow}\n")

    def set_defaults(self, **kwargs) -> dict:
        """Set default parameters for the flow if not provided."""
        # This could include defaults based on e.g. the number of dimensions
        kwargs = copy.deepcopy(kwargs)
        kwargs.setdefault("n_transforms", 4)
        kwargs.setdefault("n_neurons", 128)
        kwargs.setdefault("batch_norm_between_transforms", False)
        kwargs.setdefault("distribution", "uniform")
        kwargs.setdefault("num_bins", 20)
        kwargs.setdefault("min_derivative", 1e-6)
        kwargs.setdefault("min_bin_width", 1e-6)
        kwargs.setdefault("min_bin_height", 1e-6)
        return kwargs

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Loss function to minimize."""
        return -self.flow.log_prob(x).mean()

    def fit_data_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the data transform to the data."""
        return self.data_transform.fit(x)

    def fit(
        self,
        x,
        n_epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 5000,
        validation_fraction: float = 0.2,
        lr_annealing: bool = False,
        n_samples: int = None,
    ) -> dict:
        """Fit the normalizing flow to the data.
        This assumes all the data fits in memory.
        Parameters
        ----------
        x: np.ndarray
            The data to fit the model to. Shape should be (n_samples, dims).
        n_epochs: int
            The number of epochs to train for. Default is 100.
        lr: float
            The learning rate to use. Default is 1e-3.
        batch_size: int
            The batch size to use. Default is 500.
        validation_fraction: float
            The fraction of the data to use for validation. Default is 0.2.
        lr_annealing: bool
            Whether to use learning rate annealing. Default is False.
        """
        x = torch.tensor(x,dtype=torch.get_default_dtype()).to(self.device)
        # Transform data to unit hypercube
        x_prime = self.fit_data_transform(x)
        # Shuffle data

        indices = torch.randperm(x_prime.shape[0])
        x_prime = x_prime[indices, ...]
        if n_samples is not None:
            x_prime = x_prime[:n_samples]
        # Check for NaNs or infs
        if torch.isnan(x_prime).any():
            raise ValueError("Data contains NaN values.")
        if not torch.isfinite(x_prime).all():
            raise ValueError("Data contains infinite values.")
        # Split into training and validation sets
        n = x_prime.shape[0]
        x_train = torch.as_tensor(
            x_prime[: -int(validation_fraction * n)],
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        logger.info(
            f"Training on {x_train.shape[0]} samples, "
            f"validating on {x_prime.shape[0] - x_train.shape[0]} samples."
        )
        # Validation set
        x_val = torch.as_tensor(
            x_prime[-int(validation_fraction * n) :],
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        # Define data loaders for training
        dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train),
            shuffle=True,
            batch_size=batch_size,
        )
        val_dataset = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_val),
            shuffle=False,
            batch_size=batch_size,
        )
        # Optimizer
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        # Learning rate annealing
        # Decay the learning rate to zero over the course of training
        if lr_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
        history = {"training_loss": [], "validation_loss": []}
        # Training loop
        for _ in tqdm.tqdm(range(n_epochs)):
            self.flow.train()
            loss_epoch = 0.0
            for (x_batch,) in dataset:
                loss = self.loss_fn(x_batch)
                optimizer.zero_grad()
                loss.backward()
                # We may want to clip gradients in some cases
                # torch.nn.utils.clip_grad_norm_(flow.parameters(), 2.0)
                optimizer.step()
                loss_epoch += loss.item()
            if lr_annealing:
                scheduler.step()
            history["training_loss"].append(loss_epoch / len(dataset))
            self.flow.eval()
            val_loss = 0.0
            for (x_batch,) in val_dataset:
                with torch.no_grad():
                    val_loss += self.loss_fn(x_batch).item()
            history["validation_loss"].append(val_loss / len(val_dataset))
        self.flow.eval()
        return history

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the model.
        Parameters
        ----------
        n_samples: int
            The number of samples to draw. Default is 1.
        
        Returns
        -------
        samples: np.ndarray
            The samples drawn from the model. Shape is (n_samples, dims).
        """
        # Make sure the model is in eval mode
        if self.flow.training:
            self.flow.eval()
        # Disable gradients to speed up sampling
        with torch.no_grad():
            x_prime = self.flow.sample(
                n_samples,
            )
        # Transform back to original space
        x = self.data_transform.inverse(x_prime)[0]
        # Return as numpy array
        return x.cpu().numpy()

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute the log probability of the data under the model.
        Parameters
        ----------
        x: np.ndarray
            The data to compute the log probability for. Shape should be (n_samples, dims).
        Returns
        -------
        log_prob: np.ndarray
            The log probability of the data under the model. Shape is (n_samples,).
        """
        # Make sure the model is in eval mode
        if self.flow.training:
            self.flow.eval()
        # Disable gradients to speed up computation
        with torch.no_grad():
            # Convert to torch tensor and move to device
            x = torch.tensor(x,dtype=torch.get_default_dtype()).to(self.device)
            log_prob = torch.full((len(x),),-torch.inf)
            in_bounds = self.data_transform.in_bounds(x)
            if not torch.any(in_bounds):
                return log_prob.cpu().numpy()
            # Transform to unit hypercube
            x_prime, log_abs_det_jacobian = self.data_transform.forward(x[in_bounds])
            # Compute log probability
            log_prob[in_bounds] = self.flow.log_prob(x_prime) + log_abs_det_jacobian
    
        # Return as numpy array
        return log_prob.cpu().numpy()

    def prob(self, x):
        return np.exp(self.log_prob(x))

    def to_file(self, filename: str, group_name: str = "model") -> None:
        """Save to file.
        Saves the model architecture and weights to an h5 file. This file has
        three groups: the top level has the class name, the config group has all
        the configuration parameters, and the weights group has all the model
        weights.
        Parameters
        ----------
        filename: str
            The file to save the model to.
        """
        with h5py.File(filename, "w") as f:
            self.to_hdf5(f, group_name=group_name)

    def to_hdf5(self, h5py_file: h5py.File, group_name: str = "model") -> None:
        """Save to file.
        Saves the model architecture and weights to an h5 file. This file has
        three groups: the top level has the class name, the config group has all
        the configuration parameters, and the weights group has all the model
        weights.
        Parameters
        ----------
        h5py_file: h5py.File
            The h5py file to save the model to.
        group_name: str
            The name of the group to save the model under.
        """
        base_group = h5py_file.create_group(group_name)
        # top level has class name, config and weights
        base_group.attrs["model_class"] = self.__class__.__name__
        # group called config has all the config info
        g = base_group.create_group("config")
        g.attrs["dims"] = self.dims
        g.attrs["flow_class"] = self.flow_class
        g.attrs["bounds"] = self.data_transform.bounds.numpy()
        g.attrs["device"] = str(self.device)
        # Sort kwargs
        for k, v in sorted(self.kwargs.items()):
            g.attrs[k] = v
        w = base_group.create_group("weights")
        for k, v in self.flow.state_dict().items():
            w.create_dataset(k, data=np.atleast_1d(v.cpu().numpy()))

    @classmethod
    def from_hdf5(cls, h5py_file: h5py.File, group_name: str = "model", load_weights: bool = True) -> "NormalizingFlow":
        """Load from a file.
        Load a model from an h5 file. See `to_file` for file structure.
        Parameters
        ----------
        h5py_file: h5py.File
            The h5py file to load the model from.
        group_name: str
            The name of the group to load the model from.
        load_weights: bool
            Whether to load the model weights. Default is True.
        Returns
        -------
        NormalizingFlow
            The loaded model.
        """
        f = h5py_file[group_name]
        assert f.attrs["model_class"] == cls.__name__
        flow_kwargs = dict(f["config"].attrs)
        logger.info(f"Loading model with config: {flow_kwargs}")
        model = cls(**flow_kwargs)
        if load_weights:
            logger.info("Loading model weights")
            w = f["weights"]
            state_dict = {
                k: torch.from_numpy(v[()]).to(model.device) for k, v in w.items()
            }
            model.flow.load_state_dict(state_dict)
        return model

    @classmethod
    def from_file(cls, filename: str, load_weights: bool = True) -> "NormalizingFlow":
        """Load from a file.
        Load a model from an h5 file. See `to_file` for file structure.
        Parameters
        ----------
        filename: str
            The file to load the model from.
        load_weights: bool
            Whether to load the model weights. Default is True.
        Returns
        -------
        NormalizingFlow
            The loaded model.
        """
        import h5py

        with h5py.File(filename, "r") as f:
            assert f.attrs["model_class"] == cls.__name__
            flow_kwargs = dict(f["config"].attrs)
            logger.info(f"Loading model with config: {flow_kwargs}")
            model = cls(**flow_kwargs)
            if load_weights:
                logger.info("Loading model weights")
                w = f["weights"]
                state_dict = {
                    k: torch.from_numpy(v[()]).to(model.device) for k, v in w.items()
                }
                model.flow.load_state_dict(state_dict)
        return model


# Registry of known classes
KNOWN_CLASSES = {
    "normalizingflow": NormalizingFlow,
}


class MLStatistic:
    """
    Class to handle machine learning based statistics.
    Parameters
    ----------
    filename: str
        The file to load the model from. If provided, `model` and `model_class`
        are ignored.
    model: MLModel
        An instance of a MLModel to use. If provided, `filename` and `model_class`
        are ignored.
    model_class: Type[MLModel]
        The class of MLModel to use. Default is NormalizingFlow.
    kwargs: dict
        Additional arguments to pass to the model class or to load the model from file.
    """

    def __init__(
        self,
        filename: str = None,
        model: MLModel = None,
        model_class: Type[MLModel] = NormalizingFlow,
        metadata: dict = None,
        **kwargs,
    ) -> None:
        if filename:
            self.model = self.from_file(filename, **kwargs)
        elif model:
            self.model = model
        else:
            # For now this is hardcoded, but we can add more options later
            if isinstance(model_class, str):
                model_class = KNOWN_CLASSES.get(model_class.lower())
                if model_class is None:
                    raise ValueError(f"Unknown model class: {model_class}")
            self.model = model_class(**kwargs)
        self.metadata = metadata if metadata is not None else {}

    @classmethod
    def from_file(cls, filename: str, group_name: str = "model", **kwargs) -> "MLStatistic":
        """
        Load a MLStatistic from a file.
        Parameters
        ----------
        filename: str
            The file to load the model from.
        Returns
        -------
        MLStatistic
            The loaded MLStatistic.
        """
        with h5py.File(filename, "r") as f:
            ModelClass = KNOWN_CLASSES.get(f[group_name].attrs["model_class"].lower())
            model = ModelClass.from_hdf5(f, group_name=group_name, **kwargs)
            metadata = dict(f["metadata"].attrs)
        return cls(model=model, metadata=metadata)

    def to_file(self, filename: str, **kwargs) -> None:
        """
        Save the MLStatistic to a file.
        Parameters
        ----------
        filename: str
            The file to save the model to.
        """
        with h5py.File(filename, "w") as f:
            f.attrs['stat'] = self.metadata.get('stat', 'MLStatistic')
            metadata_group = f.create_group("metadata")
            for k, v in self.metadata.items():
                metadata_group.attrs[k] = v
            self.model.to_hdf5(f, **kwargs)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log probability of the data under the model.
        Parameters
        ----------
        x: np.ndarray
            The data to compute the log probability for.
        Returns
        -------
        log_prob: np.ndarray
            The log probability of the data.
        """
        return self.model.log_prob(x)

    def fit(self, x: np.ndarray, **kwargs) -> dict:
        """
        Fit the model to the data.
        Parameters
        ----------
        x: np.ndarray
            The data to fit the model to.
        Returns
        -------
        history: dict
            The training history.
        """
        return self.model.fit(x, **kwargs)

    def prob(self, x):
        return self.model.prob(x)


if __name__ == "__main__":
    import numpy as np

    data = 8 * np.random.rand(10000, 2).astype(np.float32) - 4
    ml_stat = MLStatistic(
        dims=2,
        bounds=np.array([[-4, 4], [-4, 4]]),
        n_neurons=64,
    )
    ml_stat.model.fit(data, n_epochs=10)
    samples = ml_stat.model.sample(10)
    log_probs = ml_stat.model.log_prob(data[:10])
    # Test saving and loading
    ml_stat.to_file("ml_stat.h5")
    ml_stat2 = MLStatistic.from_file("ml_stat.h5")
    log_probs2 = ml_stat2.model.log_prob(data[:10])
    assert np.allclose(log_probs, log_probs2)
    ml_stat3 = MLStatistic.from_file("ml_stat.h5", load_weights=False)
    log_probs3 = ml_stat3.model.log_prob(data[:10])
    assert not np.allclose(log_probs, log_probs3)
    # Time the log_prob method
    import time

    n = 1000
    # Need to decide how to handle float32 vs float64
    x_np = np.random.randn(n, 2).astype(np.float32)
    start = time.time_ns()
    for _ in range(100):
        ml_stat.log_prob(x_np)
    end = time.time_ns()
    print(f"Average time to evaluate log_prob on 1000 samples: {(end - start) / 100 / 1e6} ms")

