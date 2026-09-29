import torch
import torch.nn as nn
import torch.nn.functional as F


def early_exit_detector(
    losses, start_after=100, lookback=600, p_cutoff=0.05, on_validation=True
):
    """# Early Exit Detection

    Detect when a linear regression on the most recent `lookback` loss values turn over
    Only start after `start_after` iterations have progressed
    Assume a confidence of `p_cutoff`
    If `on_validation` only process the validation losses
    Assume that the losses are in the form of [(t_loss_0, v_loss_0),...]
    """
    from scipy.stats import linregress

    t_losses, v_losses = list(zip(*losses))
    on_loss = v_losses if on_validation else t_losses
    if len(v_losses) > start_after + lookback:
        x = range(lookback)
        y = on_loss[-lookback:]
        reg_output = linregress(x, y)
        if reg_output.pvalue > p_cutoff:
            return True
    return False


def rescale(x, m, s):
    from numpy import array

    m = array(m)
    s = array(s)
    return (x - m) / s


def load_raw_data(
    base_path,
    include_image=False,
    include_metrics=False,
    include_energetics=False,
    include_metricenergetics=False,
    include_latent=False,
    include_nuclei=False,
    include_nuclei_only=False,
    include_random=[False, 0, 0],
):
    import json
    from numpy import array, hstack

    metric_means = [
        3502.971142073042,
        85.00468020299446,
        53.464603472173415,
        0.731229006178522,
        3845.033671099852,
        64.71664786018896,
        0.9183397951608545,
        0.6103669853421727,
        236.42698767152342,
        233.2487797762296,
        100.589953051465,
        23.846228548049524,
        34.359612821138484,
    ]
    metric_stdev = [
        1.85780382e03,
        2.46159747e01,
        1.55011250e01,
        1.46940730e-01,
        2.10144664e03,
        1.64886892e01,
        5.45856614e-02,
        9.06750294e-02,
        6.45156794e01,
        6.17974799e01,
        2.93619892e01,
        6.94590854e00,
        1.92704459e01,
    ]
    nuclei_means = [
        429.29368823387625,
        450.62477187424673,
        76.27442670437136,
        27.91251776669599,
        18.498285917806736,
        0.7022634386169229,
        0.7163965729747223,
        0.9515115374290878,
        22.34525824686273,
    ]
    nuclei_stdev = [
        3.20191002e02,
        3.39974166e02,
        2.51591279e01,
        9.05870226e00,
        6.33460309e00,
        1.54504718e-01,
        7.60154292e-02,
        2.61432521e-02,
        6.87627327e00,
    ]
    energy_means = [3502.46273207, 211.98549232, 69.32343727, 144.4087803]
    energy_stdev = [1856.88233949, 58.47988665, 59.3040725, 67.68173487]

    return_objects = []
    assert (
        include_image
        or include_metrics
        or include_nuclei
        or include_nuclei_only
        or include_energetics
        or include_metricenergetics
        or include_latent
        or include_random[0]
    ), "Must have resulting data"
    #     assert not (include_image and include_metrics), "Must have only one resulting data"
    with open(base_path + "/metadata.json") as fhandle:
        meta = json.load(fhandle)
        meta["assignment"] = (
            "" if "assignment" not in meta.keys() else meta["assignment"]
        )
        return_objects.append(meta)
    # desired_data_columns = [
    #    'CentroidX',
    #    'CentroidY',
    # ]
    desired_data_columns = []
    if include_image:
        from matplotlib import pyplot as plt

        return_objects.append(
            plt.imread(base_path + "/image.png")[..., :3].transpose(2, 0, 1)
        )
    if include_metrics:
        import pandas as pd

        scaled_data_columns = [
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ]
        df = pd.read_csv(base_path + "/Metrics.csv")
        ratio = 10 / return_objects[0]["magnification"]
        prescale_powers = array([2, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1])
        prescale = ratio**prescale_powers
        scaled_data = df[scaled_data_columns].to_numpy() * prescale
        max_scale = array(
            [
                6.18800000e03,
                1.52125653e02,
                8.19416135e01,
                9.99175882e-01,
                8.24700000e03,
                8.87626402e01,
                1.00000000e00,
                9.27272727e-01,
                5.78900000e02,
                3.58986000e02,
                1.72137177e02,
                3.04138126e01,
                1.54225449e02,
            ]
        )
        max_scale = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        scaled_data = scaled_data / max_scale
        # New Scaling
        scaled_data = rescale(
            df[scaled_data_columns].to_numpy(), metric_means, metric_stdev
        )
        data = hstack([scaled_data, df[desired_data_columns].to_numpy()])
        return_objects.append(data)
    if include_nuclei:
        import pandas as pd

        scaled_data_columns = [
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ]
        nuclei_columns = [
            "area",
            "area_convex",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "extent",
            "solidity",
            "equivalent_diameter_area",
        ]
        df = pd.read_csv(base_path + "/CNJoinedMetrics.csv")
        ratio = 10 / return_objects[0]["magnification"]
        prescale_powers = array([2, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1])
        prescale = ratio**prescale_powers
        scaled_data = df[scaled_data_columns].to_numpy() * prescale
        max_scale = array(
            [
                6.18800000e03,
                1.52125653e02,
                8.19416135e01,
                9.99175882e-01,
                8.24700000e03,
                8.87626402e01,
                1.00000000e00,
                9.27272727e-01,
                5.78900000e02,
                3.58986000e02,
                1.72137177e02,
                3.04138126e01,
                1.54225449e02,
            ]
        )
        max_scale = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        scaled_data = scaled_data / max_scale
        # New Scaling
        scaled_data = rescale(
            df[scaled_data_columns].to_numpy(), metric_means, metric_stdev
        )
        nuclei_data = rescale(df[nuclei_columns].to_numpy(), nuclei_means, nuclei_stdev)
        data = hstack(
            [
                scaled_data,
                df[desired_data_columns].to_numpy(),
                nuclei_data,
            ]
        )
        return_objects.append(data)
    if include_nuclei_only:
        import pandas as pd

        df = pd.read_csv(base_path + "/CNJoinedMetrics.csv")
        nuclei_columns = [
            "area",
            "area_convex",
            "perimeter",
            "axis_major_length",
            "axis_minor_length",
            "eccentricity",
            "extent",
            "solidity",
            "equivalent_diameter_area",
        ]
        data = hstack(
            [
                df[desired_data_columns].to_numpy(),
                df[nuclei_columns].to_numpy(),
            ]
        )
        nuclei_data = rescale(df[nuclei_columns].to_numpy(), nuclei_means, nuclei_stdev)
        data = hstack(
            [
                df[desired_data_columns].to_numpy(),
                nuclei_data,
            ]
        )
        return_objects.append(data)
    if include_latent:
        import pandas as pd

        scaled_data_columns = [
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ]
        df = pd.read_csv(base_path + "/MetricsAndLatent.csv")
        data = df[["Latent-0", "Latent-1"]].to_numpy()
        return_objects.append(data)
    if include_energetics:
        from numpy import load as nload

        data = nload(base_path + "/Energetics.npy")
        max_scale = array([26_000, 700, 550, 650])
        scaled_data = data / max_scale
        scaled_data = rescale(data, energy_means, energy_stdev)
        return_objects.append(scaled_data)
    if include_metricenergetics:
        from numpy import load as nload

        data = nload(base_path + "/Energetics.npy")
        max_scale = array([26_000, 700, 550, 650])
        nrg_data = data / max_scale
        nrg_data = rescale(data, energy_means, energy_stdev)
        import pandas as pd

        scaled_data_columns = [
            "Area",
            "MjrAxisLength",
            "MnrAxisLength",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Solidity",
            "Extent",
            "Perimeter",
            "ConvexPerim",
            "FibLen",
            "InscribeR",
            "BlebLen",
        ]
        df = pd.read_csv(base_path + "/Metrics.csv")
        ratio = 10 / return_objects[0]["magnification"]
        prescale_powers = array([2, 1, 1, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1])
        prescale = ratio**prescale_powers
        scaled_data = df[scaled_data_columns].to_numpy() * prescale
        max_scale = array(
            [
                6.18800000e03,
                1.52125653e02,
                8.19416135e01,
                9.99175882e-01,
                8.24700000e03,
                8.87626402e01,
                1.00000000e00,
                9.27272727e-01,
                5.78900000e02,
                3.58986000e02,
                1.72137177e02,
                3.04138126e01,
                1.54225449e02,
            ]
        )
        max_scale = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        scaled_data = scaled_data / max_scale
        # print(base_path, scaled_data.shape, nrg_data.shape)
        data = hstack(
            [
                scaled_data,
                df[desired_data_columns].to_numpy(),
                nrg_data[: scaled_data.shape[0], ...],
            ]
        )
        return_objects.append(data)
    if include_random[0]:
        assert isinstance(include_random[1], int)
        assert include_random[1] > 0
        assert isinstance(include_random[2], int)
        assert include_random[2] > 0
        from numpy.random import rand

        data = rand(include_random[1], include_random[2])
        return_objects.append(data)
    return return_objects


def load_data(base_path, meta_type, data_type):
    """# Load a histology data source
    data_type: 'image' or 'metrics' or 'energetics' or 'random'
    meta_type: 'tn' or 'stage_basic' or 'stage_advanced'"""
    data_type_possibilities = [
        "image",
        "metrics",
        "nuclei",
        "nuclei_only",
        "energetics",
        "metricenergetics",
        "latent",
        "random",
        "image_vt",
    ]
    meta_type_possibilities = [
        "tn",
        "stage_basic",
        "stage_advanced",
        "grade",
    ]
    assert data_type in data_type_possibilities
    assert meta_type in meta_type_possibilities
    print(base_path)
    if (data_type == "image") or (data_type == "image_vt"):
        meta, data = load_raw_data(base_path, include_image=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "metrics":
        meta, data = load_raw_data(base_path, include_metrics=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "nuclei":
        meta, data = load_raw_data(base_path, include_nuclei=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "nuclei_only":
        meta, data = load_raw_data(base_path, include_nuclei_only=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "energetics":
        meta, data = load_raw_data(base_path, include_energetics=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "metricenergetics":
        meta, data = load_raw_data(base_path, include_metricenergetics=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "latent":
        meta, data = load_raw_data(base_path, include_latent=True)
        return meta[meta_type + "_i"], data, meta["assignment"]
    elif data_type == "random":
        meta, data = load_raw_data(base_path, include_random=[True, 200, 6])
        return meta[meta_type + "_i"], data, meta["assignment"]
    assert False, "Should never reach here"


def load_data_bank(base_paths, meta_type, data_type):
    data = []
    for base_path in base_paths:
        m, d, a = load_data(base_path, meta_type, data_type)
        if m < 0:
            continue
        if len(d) == 0:
            continue
        data.append((m, d, base_path, a))
    return zip(*data)


class HistologyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta,
        data_base,
        min_seq_len=10,
        seq_len=20,
        mask_p=0.2,
        name="",
        assignment="",
    ):
        self.meta = torch.Tensor([meta])
        self.data_base = torch.Tensor(data_base)
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.mask_p = mask_p
        self.name = name
        self.assignment = assignment
        assert isinstance(seq_len, int)
        assert seq_len > 0

    def to(self, *args):
        self.meta = self.meta.to(*args)
        self.data_base = self.data_base.to(*args)
        return self

    def __getitem__(self, idx):
        output_data = torch.zeros(
            self.seq_len, *self.data_base.shape[1:], device=self.data_base.device
        )
        r_len = torch.randint(self.min_seq_len, self.seq_len, (1,)).item()
        n = self.seq_len
        if torch.rand((1)).item() < self.mask_p:
            n = r_len
        n = min(self.seq_len, self.data_base.shape[0], n)
        output_data[:n, ...] = self.data_base[
            torch.randperm(self.data_base.size()[0]), ...
        ][:n, ...]
        return output_data, self.meta

    def __len__(self):
        return 1


class HistologyImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, meta, data_base, name="", assignment="", height=2560, width=1920
    ):
        self.meta = torch.Tensor([meta])
        self.data_base = torch.Tensor(data_base)
        self.data_base = torch.nn.functional.pad(
            self.data_base,
            (
                0,
                height - self.data_base.shape[2],
                0,
                width - self.data_base.shape[1],
                0,
                0,
            ),
        )
        self.name = name
        self.assignment = assignment

    def to(self, *args):
        self.meta = self.meta.to(*args)
        self.data_base = self.data_base.to(*args)
        return self

    def __getitem__(self, idx):
        return self.data_base, self.meta

    def __len__(self):
        return 1


class AttentionBlock(nn.Module):
    """# Attention Block

    FF(MHA(x))"""

    def __init__(
        self, embed_dim, num_heads, dim_feedforward, dropout=0.1, activation=F.elu
    ):
        super(AttentionBlock, self).__init__()
        self.attention_layer = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feedforward_layer = nn.Linear(embed_dim, dim_feedforward)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dim_feedforward, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        """# Forward pass

        x: [N_batch, N_seq, N_dim]"""
        dx, _ = self.attention_layer(x, x, x)
        X_hat = x + dx
        X_norm = self.attention_norm(X_hat)

        z = self.feedforward_dropout(self.activation(self.feedforward_layer(X_norm)))
        z = self.output_dropout(self.activation(self.output_layer(z)))
        return z


class CloudFlatten(nn.Module):
    """# Flatten Point Cloud

    Useful to feed entire cloud to classifier model after attending"""

    def __init__(self, embed_dim, max_points):
        super(CloudFlatten, self).__init__()

    def forward(self, x):
        x_shape = (*x.shape[:-2], x.shape[-2] * x.shape[-1])
        return x.reshape(x_shape)


class CloudFill(nn.Module):
    """# Pad Cloud

    Useful to pad cloud to appropriate size to subsequent model layers"""

    def __init__(self, max_points, min_points=0):
        """# Initialization

        max_points: The desired size of the point cloud
        min_points: The minimal size of the starting point cloud"""
        super(CloudFill, self).__init__()
        self.max_points = max_points
        self.min_points = min_points

    def forward(self, x):
        """# Reshape given cloud to desired length"""
        # Ensure cloud size is maintained
        assert (
            x.shape[-2] >= self.min_points
        ), f"Cloud must at least be of size {self.min_points}"
        assert x.shape[-2] <= self.max_points, "Point Count Exceeded"
        # Quick return for correctly sized clouds
        if x.shape[-2] == self.max_points:
            return x
        # Otherwise pad with zeros
        z_shape = (*x.shape[:-2], self.max_points - x.shape[-2], x.shape[-1])
        x = torch.cat((x, torch.zeros(*z_shape).to(x.device)), dim=-2)
        return x


class SimpleHiddenLayer(nn.Module):
    """# Simple Hidden Layer"""

    def __init__(self, dim_in, dim_out, dropout=0.1, activation=F.elu):
        super(SimpleHiddenLayer, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class PointCloudClassifier(nn.Module):
    """# Point Cloud Classifier

    Classify continuous point cloud with at most `max_points` points.
    """

    def __init__(
        self,
        embed_dim,
        max_points,
        num_heads,
        dim_feedforward,
        N_attn=1,
        min_points=0,
        dropout=0.1,
        n_classes=1,
        activation=F.elu,
        dropout_output=False,
        n_classifier_layers=1,
    ):
        """# Model initialization

        embed_dim: The number of dimensions for each point
        max_points: The maximum number of points in the cloud
        num_heads: Transformer head count
        dim_feedforward: Dimensionality of feed-forward network subnetwork
        min_points: Minimum point count required
        dropout: Dropout for model overfitting mitigation
        """
        super(PointCloudClassifier, self).__init__()
        self.activation = activation
        self.cloud_fill = CloudFill(max_points, min_points)
        self.attention_layers = nn.ModuleList(
            [
                AttentionBlock(
                    embed_dim,
                    num_heads,
                    dim_feedforward,
                    dropout=dropout,
                    activation=self.activation,
                )
                for _ in range(N_attn)
            ]
        )
        self.cloud_flatten = CloudFlatten(embed_dim, max_points)
        classifier_layers = []
        assert isinstance(n_classifier_layers, int)
        assert n_classifier_layers > 0
        if n_classifier_layers == 1:
            classifier_layers.append(
                SimpleHiddenLayer(
                    embed_dim * max_points,
                    n_classes,
                    dropout=dropout if dropout_output else 0,
                    activation=lambda x: x,
                )
            )
        else:
            classifier_layers.append(
                SimpleHiddenLayer(
                    embed_dim * max_points, dim_feedforward, dropout=dropout
                )
            )
            for _ in range(n_classifier_layers - 2):
                classifier_layers.append(
                    SimpleHiddenLayer(dim_feedforward, dim_feedforward, dropout=dropout)
                )
            classifier_layers.append(
                SimpleHiddenLayer(
                    dim_feedforward,
                    n_classes,
                    dropout=dropout if dropout_output else 0,
                    activation=lambda x: x,
                )
            )
        self.classifier = nn.Sequential(*classifier_layers)
        #         self.classifier = nn.Sequential(
        #             SimpleHiddenLayer(embed_dim*max_points, dim_feedforward, dropout=dropout),
        # #             SimpleHiddenLayer(dim_feedforward, dim_feedforward, dropout=dropout),
        #             SimpleHiddenLayer(dim_feedforward, n_classes, dropout=dropout if dropout_output else 0, activation=lambda x: x)
        #         )
        self.transfer = F.sigmoid

    def forward(self, x):
        """# Forward pass of classifier

        Classify point cloud.
        Automatically pads to model point cloud size.
        """
        # Ensure cloud size is maintained
        x = self.cloud_fill(x)

        # Attention Layer
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        X_t = x

        # Flatten cloud for classification
        X_t = self.cloud_flatten(X_t)

        # Final classification
        classified = self.classifier(X_t)
        return self.transfer(classified)


class BasicPointCloudClassifier(nn.Module):
    """# Point Cloud Classifier

    Classify continuous point cloud with at most `max_points` points.
    """

    def __init__(
        self,
        embed_dim,
        max_points,
        num_heads,
        dim_feedforward,
        min_points=0,
        dropout=0.1,
        n_classes=3,
    ):
        """# Model initialization

        embed_dim: The number of dimensions for each point
        max_points: The maximum number of points in the cloud
        num_heads: Transformer head count
        dim_feedforward: Dimensionality of feed-forward network subnetwork
        min_points: Minimum point count required
        dropout: Dropout for model overfitting mitigation
        """
        super(PointCloudClassifier, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_linear = nn.Linear(embed_dim, dim_feedforward)
        self.class_linear = nn.Linear(dim_feedforward * max_points, n_classes)
        self.activation = F.elu
        self.transfer = F.sigmoid
        self.seq_length = max_points
        self.min_points = min_points

    def forward(self, x):
        """# Forward pass of classifier

        Classify point cloud.
        Automatically pads to model point sequence count.
        """
        # Ensure sequence length is maintained
        assert (
            x.shape[-2] >= self.min_points
        ), f"Sequence must at lest be of length {self.min_points}"
        assert x.shape[-2] <= self.seq_length, "Sequence Length Exceeded"
        z_shape = (*x.shape[:-2], self.seq_length - x.shape[-2], x.shape[-1])
        x = torch.cat((x, torch.zeros(*z_shape).to(x.device)), dim=-2)

        # Attention Layer
        attn, _ = self.attn_layer(x, x, x)
        attn += x
        attn = self.attn_norm(attn)
        # Feed forward
        ff = self.ff_dropout(self.activation(self.ff_linear(attn)))

        # Final classification
        ff_shape = (*ff.shape[:-2], ff.shape[-2] * ff.shape[-1])
        classified = self.class_linear(ff.reshape(ff_shape))
        return self.transfer(classified)


class SimpleConvBlock(nn.Module):
    """# Simple Hidden Layer"""

    def __init__(self, n, m, dropout=0.1, activation=F.elu):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n, m, kernel_size=3, padding=1),
            nn.BatchNorm2d(m),
        )
        self.activation = activation
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            SimpleConvBlock(3, 8),
            SimpleConvBlock(8, 16),
            SimpleConvBlock(16, 32),
            # Adaptive pooling to handle arbitrary input sizes
            nn.AdaptiveAvgPool2d((1, 1)),  # Output feature map size is 1x1
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)  # Input to linear layer is 128 * 1 * 1
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


def get_datestring():
    from datetime import datetime

    now = datetime.now()
    return f"{now.year:04}{now.month:02}{now.day:02}-{now.hour:02}{now.minute:02}{now.second:02}-{now.microsecond:06}"


def accuracy_from_matrix(matrix):
    from torch import eye

    correct = (matrix * eye(matrix.shape[0])).sum().item()
    total = matrix.sum().item()
    return correct / total


def train_epoch_ohe(
    model, data_loader, optimizer, num_classes, device="cuda", loss_fn=F.mse_loss
):
    from torch import int64, float32, zeros
    from torch.nn.functional import one_hot, softmax

    model.to(device)
    model.train()
    total_loss = 0.0
    loss_count = 0
    accuracy_matrix = zeros((num_classes, num_classes), dtype=int)
    for data, label in data_loader:
        data = data.to(device)
        label = label.to(int64)
        label_ohe = one_hot(label.flatten(), num_classes).to(data.dtype).to(device)
        model.zero_grad()
        forward = softmax(model(data), dim=-1)
        loss = loss_fn(forward, label_ohe)
        loss.backward()
        optimizer.step()
        label_pred = forward.argmax(dim=-1)
        for actual, pred in zip(label, label_pred):
            actual = actual.to(int).item()
            pred = pred.to(int).item()
            accuracy_matrix[actual, pred] += 1
        total_loss += loss.item()
        loss_count += 1
    return (
        total_loss / loss_count,
        accuracy_from_matrix(accuracy_matrix),
        accuracy_matrix,
    )


def eval_epoch_ohe(model, data_loader, num_classes, device="cuda", loss_fn=F.mse_loss):
    from torch import int64, float32, no_grad, zeros
    from torch.nn.functional import one_hot, softmax

    model.to(device)
    model.eval()
    total_loss = 0.0
    loss_count = 0
    accuracy_matrix = zeros((num_classes, num_classes), dtype=int)
    with no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(int64)
            label_ohe = one_hot(label.flatten(), num_classes).to(data.dtype).to(device)
            forward = softmax(model(data), dim=-1)
            # print(forward.shape, label_ohe.shape)
            loss = loss_fn(forward, label_ohe)
            label_pred = forward.argmax(dim=-1)
            for actual, pred in zip(label, label_pred):
                actual = actual.to(int).item()
                pred = pred.to(int).item()
                accuracy_matrix[actual, pred] += 1
            total_loss += loss.item()
            loss_count += 1
    return (
        total_loss / loss_count,
        accuracy_from_matrix(accuracy_matrix),
        accuracy_matrix,
    )


def analyze_model_performance_on_datasets(model, datasets, n_tests=1):
    from torch import no_grad
    from numpy import array

    model.eval()
    output = []
    with no_grad():
        subslice = [None, ...]
        for subdataset in datasets:
            dataset_results = []
            name = subdataset.name
            for _ in range(n_tests):
                raw = subdataset[0]
                data = raw[0]
                label = raw[1]
                dataset_results.append(
                    (model(data[subslice]).argmax(dim=-1) == label.item()).item()
                )
            dataset_results = array(dataset_results)
            p_pass = float(dataset_results.mean())
            output.append((name, p_pass > 0.5, p_pass))
    return output


def get_num_classes(data_bank):
    return max(data_bank[0]) + 1


def get_data_dim(data_bank):
    return (data_bank[1][0]).shape[-1]


def get_dataset(data_sources, config):
    from torch.utils.data import ConcatDataset

    is_image = config["input_data_type"][: len("image")] == "image"
    data_bank = list(
        load_data_bank(
            data_sources, config["output_data_type"], config["input_data_type"]
        )
    )
    kwargs = {}
    if not is_image:
        kwargs["seq_len"] = config["pointcloud_values"]["max_points"]
        kwargs["min_seq_len"] = config["pointcloud_values"]["min_points"]
    DatasetClass = HistologyImageDataset if is_image else HistologyDataset
    datasets = []
    for index in range(len(data_bank[0])):
        label = data_bank[0][index]
        data = data_bank[1][index]
        name = data_bank[2][index]
        assignment = data_bank[3][index]
        kwargs["name"] = name
        kwargs["assignment"] = assignment
        if is_image:
            kwargs["height"] = config["image_values"]["window_height"]
            kwargs["width"] = config["image_values"]["window_width"]
        datasets.append(DatasetClass(label, data, **kwargs).to(config["device"]))
    dataset = ConcatDataset(datasets)
    num_classes = get_num_classes(data_bank)
    num_data = get_data_dim(data_bank)
    return dataset, num_data, num_classes, datasets


def train_eval_split(dataset, config, regen=False):
    from torch import Tensor, bincount
    from torch.utils.data import WeightedRandomSampler, random_split, DataLoader, Subset

    labels = Tensor([dataset[idx][1] for idx in range(len(dataset))]).to(int)
    label_bincounts = bincount(labels)
    label_weights_values = 1.0 / label_bincounts
    label_weights = Tensor(
        [label_weights_values[labels[idx]] for idx in range(len(labels))]
    )

    sampler = WeightedRandomSampler(
        label_weights.type("torch.DoubleTensor"),  # Ensure weights are a DoubleTensor
        len(label_weights),  # Number of samples to draw
        replacement=True,
    )

    t_split = config["train_val_split"]
    t_len = int(t_split * len(dataset))
    v_len = len(dataset) - t_len
    if regen:
        while True:
            t_dataset, v_dataset = random_split(dataset, [t_len, v_len])

            t_labels = Tensor([t_dataset[idx][1] for idx in range(len(t_dataset))]).to(
                int
            )
            t_label_bincounts = bincount(t_labels)
            t_label_weights_values = 1.0 / t_label_bincounts
            t_label_weights = Tensor(
                [t_label_weights_values[t_labels[idx]] for idx in range(len(t_labels))]
            )
            if (
                t_label_bincounts > 0
            ).all():  # Ensure all labels are present in training dataset
                break
        for idx in t_dataset.indices:
            dataset.datasets[idx].assignment = "training"
        for idx in v_dataset.indices:
            dataset.datasets[idx].assignment = "validation"

    t_indicies = [
        idx
        for idx in range(len(dataset))
        if dataset.datasets[idx].assignment == "training"
    ]
    v_indicies = [
        idx
        for idx in range(len(dataset))
        if dataset.datasets[idx].assignment == "validation"
    ]
    t_dataset = Subset(dataset, t_indicies)
    v_dataset = Subset(dataset, v_indicies)

    t_labels = Tensor([t_dataset[idx][1] for idx in range(len(t_dataset))]).to(int)
    t_label_bincounts = bincount(t_labels)
    t_label_weights_values = 1.0 / t_label_bincounts
    t_label_weights = Tensor(
        [t_label_weights_values[t_labels[idx]] for idx in range(len(t_labels))]
    )

    v_labels = torch.Tensor([v_dataset[idx][1] for idx in range(len(v_dataset))]).to(
        int
    )
    v_label_bincounts = torch.bincount(v_labels)
    v_label_weights_values = 1.0 / v_label_bincounts
    v_label_weights = torch.Tensor(
        [v_label_weights_values[v_labels[idx]] for idx in range(len(v_labels))]
    )

    t_sampler = WeightedRandomSampler(
        t_label_weights.type("torch.DoubleTensor"),  # Ensure weights are a DoubleTensor
        len(t_label_weights),  # Number of samples to draw
        replacement=True,
    )

    v_sampler = WeightedRandomSampler(
        v_label_weights.type("torch.DoubleTensor"),  # Ensure weights are a DoubleTensor
        len(v_label_weights),  # Number of samples to draw
        replacement=True,
    )

    train_loader = DataLoader(
        t_dataset, batch_size=config["batch_size"], sampler=t_sampler
    )
    eval_loader = DataLoader(
        v_dataset, batch_size=config["batch_size"], sampler=v_sampler
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler)
    return train_loader, eval_loader, loader


def get_model(config, data_dim, num_classes):
    is_image = config["input_data_type"][: len("image")] == "image"
    model = None
    if is_image:
        if config["input_data_type"] == "image":
            sub_config = config["image_values"]
            model = SimpleCNN(num_classes).to(config["device"])
        elif config["input_data_type"] == "image_vt":
            from torchvision.models.vision_transformer import VisionTransformer

            sub_config = config["vision_transformer_config"]
            sub_config["num_classes"] = num_classes
            model = VisionTransformer(**sub_config).to(config["device"])
    else:
        sub_config = config["pointcloud_values"]
        max_points = sub_config["max_points"]
        num_embed = data_dim
        num_head = num_embed
        dropout = sub_config["dropout"]
        dropout_output = sub_config["dropout_output"]
        n_ff_nodes = sub_config["n_ff_nodes"]
        n_ff_layers = sub_config["n_ff_layers"]
        model = PointCloudClassifier(
            num_embed,
            max_points,
            num_head,
            n_ff_nodes,
            n_classes=num_classes,
            min_points=1,
            dropout=dropout,
            dropout_output=dropout_output,
            n_classifier_layers=n_ff_layers,
        ).to(config["device"])
    return model


class ModelWrapper:
    def __init__(self):
        self.optimizer = None
        self.loss_fn = None
        self.device = "cpu"
        self.dtype = torch.float
        raise NotImplementedError("Implement Me!")

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def train_epoch(self):
        raise NotImplementedError("Implement Me!")

    def eval_epoch(self):
        raise NotImplementedError("Implement Me!")

    def forward(self, x):
        return self.model.forward(x)

    def to(self, device=None, dtype=None):
        self.device = self.device if device is None else device
        self.dtype = self.dtype if dtype is None else dtype
        self.model.to(device=device, dtype=dtype)


class PointCloudClassifierWrapper(ModelWrapper):
    def init(
        self,
        num_embed,
        max_points,
        num_head,
        n_ff_nodes,
        n_classes,
        min_points=1,
        dropout=0,
        dropout_output=0,
        n_classifier_layers=1,
    ):
        self.model = PointCloudClassifier(
            num_embed,
            max_points,
            num_head,
            n_ff_nodes,
            n_classes=n_classes,
            min_points=min_points,
            dropout=dropout,
            dropout_output=dropout_output,
            n_classifier_layers=n_classifier_layers,
        )
        self.num_classes = n_classes

    def train_epoch(self, data_loader, num_classes, device="cuda"):
        return train_epoch_ohe(
            self.model,
            data_loader,
            self.optimizer,
            self.num_classes,
            device=self.device,
            loss_fn=self.loss_fn,
        )

    def eval_epoch(self, data_loader, num_classes, device="cuda", loss_fn=F.mse_loss):
        return eval_epoch_ohe(
            self.model,
            data_loader,
            self.num_classes,
            device=self.device,
            loss_fn=loss_fn,
        )
